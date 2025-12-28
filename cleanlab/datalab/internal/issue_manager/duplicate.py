from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


from cleanlab.datalab.internal.issue_manager import IssueManager
from cleanlab.datalab.internal.issue_manager.knn_graph_helpers import set_knn_graph
from cleanlab.internal.constants import EPSILON

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from cleanlab.datalab.datalab import Datalab


class NearDuplicateIssueManager(IssueManager):
    """Manages issues related to near-duplicate and exact duplicate examples.

    This issue manager can detect both exact duplicates (identical examples) and
    near duplicates (highly similar examples) based on distance or similarity metrics.
    """

    description: ClassVar[
        str
    ] = """A (near) duplicate issue refers to two or more examples in
    a dataset that are extremely similar to each other, relative
    to the rest of the dataset.  The examples flagged with this issue
    may be exactly duplicated (assigned a score of 0), or lie atypically 
    close together when represented as vectors (i.e. feature embeddings).
    
    For exact duplicates, the distance between examples is 0 and they are
    assigned a duplicate score of 0. For near duplicates, the score is
    close to 0 based on their similarity.
    """
    issue_name: ClassVar[str] = "near_duplicate"
    verbosity_levels = {
        0: [],
        1: ["num_duplicate_sets"],
        2: ["threshold", "metric", "exact_duplicates_only", "similarity_threshold"],
        3: [
            "threshold",
            "metric",
            "exact_duplicates_only",
            "similarity_threshold",
            "near_duplicate_sets",
        ],
    }

    def __init__(
        self,
        datalab: Datalab,
        metric: Optional[Union[str, Callable]] = None,
        threshold: float = 0.13,
        similarity_threshold: Optional[float] = None,
        k: int = 10,
        exact_duplicates_only: bool = False,
        **_,
    ):
        """
        Parameters
        ----------
        datalab : Datalab
            The Datalab instance.
        metric : Optional[Union[str, Callable]]
            Distance metric to use. If 'cosine', similarity threshold can be used.
        threshold : float
            Distance threshold for near duplicates (as fraction of median distance).
        similarity_threshold : Optional[float]
            Similarity threshold (0-1) for cosine similarity. If provided with
            metric='cosine', overrides distance threshold.
        k : int
            Number of nearest neighbors to consider.
        exact_duplicates_only : bool
            If True, only flag exact duplicates (distance=0).
        """
        super().__init__(datalab)

        # Validate parameters
        self._validate_parameters(
            metric=metric,
            threshold=threshold,
            similarity_threshold=similarity_threshold,
            k=k,
            exact_duplicates_only=exact_duplicates_only,
        )

        self.metric = metric
        self.threshold = self._set_threshold(threshold)
        self.similarity_threshold = similarity_threshold
        self.k = k
        self.exact_duplicates_only = exact_duplicates_only
        self.near_duplicate_sets: List[List[int]] = []

    def find_issues(
        self,
        features: Optional[npt.NDArray] = None,
        **kwargs,
    ) -> None:
        # Early validation for empty datasets
        if features is not None and len(features) == 0:
            warnings.warn("Empty dataset provided - no duplicates can be detected.")
            self.near_duplicate_sets = []
            self.issues = pd.DataFrame(
                {
                    f"is_{self.issue_name}_issue": [],
                    self.issue_score_key: [],
                }
            )
            self.summary = self.make_summary(score=1.0)
            self.info = {"num_duplicate_sets": 0, "num_near_duplicates": 0}
            return

        # Check memory requirements for large datasets
        if features is not None:
            self._check_memory_requirements(features=features)

        # Early validation and error handling
        try:
            knn_graph, self.metric, _ = set_knn_graph(
                features=features,
                find_issues_kwargs=kwargs,
                metric=self.metric,
                k=self.k,
                statistics=self.datalab.get_info("statistics"),
            )
        except (ValueError, AssertionError) as e:
            if "Features must be provided" in str(e):
                raise ValueError(
                    "No features provided for duplicate detection. "
                    "Please provide either 'features' or 'knn_graph' parameter. "
                    "For more details, see the documentation at: "
                    "https://docs.cleanlab.ai/stable/cleanlab/datalab/guide/issue_type_description.html#near-duplicate-issue"
                ) from e
            elif "Number of nearest neighbors k=" in str(e):
                dataset_size = len(features) if features is not None else "unknown"
                raise ValueError(
                    f"k={self.k} is too large for dataset with {dataset_size} examples. "
                    f"k must be less than the number of examples in the dataset. "
                    f"Try reducing k to a smaller value (e.g., k={min(10, max(1, int(dataset_size) - 1)) if dataset_size != 'unknown' else 5})"
                ) from e
            else:
                raise

        N = knn_graph.shape[0]
        nn_distances = knn_graph.data.reshape(N, -1)[:, 0]
        median_nn_distance = max(np.median(nn_distances), EPSILON)  # avoid threshold = 0

        # Handle similarity threshold for cosine metric
        if self.metric == "cosine" and self.similarity_threshold is not None:
            # Convert similarity threshold to distance threshold
            # cosine distance = 1 - cosine similarity
            distance_threshold = 1 - self.similarity_threshold
        else:
            distance_threshold = self.threshold * median_nn_distance

        # Find duplicates based on threshold
        if self.exact_duplicates_only:
            # Only flag exact duplicates (distance = 0)
            self.near_duplicate_sets = self._find_exact_duplicates(knn_graph)
        else:
            # Find all near duplicates within threshold
            self.near_duplicate_sets = self._neighbors_within_radius(
                knn_graph, distance_threshold, median_nn_distance
            )

        # Flag every example in a near-duplicate set as a near-duplicate issue
        # Handle case where no duplicates are found
        if self.near_duplicate_sets and any(len(s) > 0 for s in self.near_duplicate_sets):
            all_near_duplicates = np.unique(
                np.concatenate([s for s in self.near_duplicate_sets if len(s) > 0])
            )
            all_near_duplicates = all_near_duplicates.astype(int)  # Ensure integer indices
        else:
            all_near_duplicates = np.array([], dtype=int)

        is_issue_column = np.zeros(N, dtype=bool)
        if len(all_near_duplicates) > 0:
            is_issue_column[all_near_duplicates] = True
        temperature = 1.0 / median_nn_distance
        scores = _compute_scores_with_exp_transform(nn_distances, temperature=temperature)
        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": is_issue_column,
                self.issue_score_key: scores,
            },
        )

        self.summary = self.make_summary(score=scores.mean())
        self.info = self.collect_info(knn_graph=knn_graph, median_nn_distance=median_nn_distance)

    @staticmethod
    def _neighbors_within_radius(knn_graph: csr_matrix, threshold: float, median: float):
        """Returns a list of lists of indices of near-duplicate examples.

        Each list of indices represents a set of near-duplicate examples.

        If the list is empty for a given example, then that example is not
        a near-duplicate of any other example.

        Parameters
        ----------
        knn_graph : csr_matrix
            Sparse matrix of k-nearest neighbors
        threshold : float
            Absolute distance threshold (not relative to median)
        median : float
            Median distance (kept for compatibility)
        """

        N = knn_graph.shape[0]
        distances = knn_graph.data.reshape(N, -1)
        # Create a mask for the threshold (now using absolute threshold)
        mask = distances < threshold

        # Update the indptr to reflect the new number of neighbors
        indptr = np.zeros(knn_graph.indptr.shape, dtype=knn_graph.indptr.dtype)
        indptr[1:] = np.cumsum(mask.sum(axis=1))

        # Filter the knn_graph based on the threshold
        indices = knn_graph.indices[mask.ravel()]
        near_duplicate_sets = [indices[indptr[i] : indptr[i + 1]] for i in range(N)]

        # Second pass over the data is required to ensure each item is included in the near-duplicate sets of its own near-duplicates.
        # This is important because a "near-duplicate" relationship is reciprocal.
        # For example, if item A is a near-duplicate of item B, then item B should also be considered a near-duplicate of item A.
        # NOTE: This approach does not assure that the sets are ordered by increasing distance.
        for i, near_duplicates in enumerate(near_duplicate_sets):
            for j in near_duplicates:
                if i not in near_duplicate_sets[j]:
                    near_duplicate_sets[j] = np.append(near_duplicate_sets[j], i)

        return near_duplicate_sets

    @staticmethod
    def _find_exact_duplicates(knn_graph: csr_matrix) -> List[List[int]]:
        """Find only exact duplicates (distance = 0) in the dataset.

        Parameters
        ----------
        knn_graph : csr_matrix
            Sparse matrix of k-nearest neighbors

        Returns
        -------
        exact_duplicate_sets : List[List[int]]
            List of lists containing indices of exact duplicates
        """
        N = knn_graph.shape[0]
        distances = knn_graph.data.reshape(N, -1)

        # Find exact duplicates (distance very close to 0)
        exact_mask = np.isclose(distances, 0, atol=1e-10)

        # Create duplicate sets for exact duplicates only
        exact_duplicate_sets = []
        for i in range(N):
            row_mask = exact_mask[i]
            if np.any(row_mask):
                # Get indices of exact duplicates for this example
                start_idx = knn_graph.indptr[i]
                end_idx = knn_graph.indptr[i + 1]
                neighbor_indices = knn_graph.indices[start_idx:end_idx]
                exact_neighbors = neighbor_indices[row_mask[: len(neighbor_indices)]]

                if len(exact_neighbors) > 0:
                    exact_duplicate_sets.append(exact_neighbors.tolist())
                else:
                    exact_duplicate_sets.append([])
            else:
                exact_duplicate_sets.append([])

        # Make relationships reciprocal
        for i, duplicates in enumerate(exact_duplicate_sets):
            for j in duplicates:
                if i not in exact_duplicate_sets[j]:
                    exact_duplicate_sets[j].append(i)

        return exact_duplicate_sets

    def collect_info(self, knn_graph: csr_matrix, median_nn_distance: float) -> dict:
        # Count duplicate sets and exact duplicates
        num_duplicate_sets = len([s for s in self.near_duplicate_sets if len(s) > 0])
        num_exact_duplicates = 0
        if hasattr(self, "_exact_duplicate_count"):
            num_exact_duplicates = self._exact_duplicate_count

        # Handle case where no duplicates are found
        if self.near_duplicate_sets and any(len(s) > 0 for s in self.near_duplicate_sets):
            all_duplicates = np.unique(
                np.concatenate([s for s in self.near_duplicate_sets if len(s) > 0])
            )
            num_near_duplicates = len(all_duplicates)
        else:
            num_near_duplicates = 0

        issues_dict = {
            "average_near_duplicate_score": self.issues[self.issue_score_key].mean(),
            "near_duplicate_sets": self.near_duplicate_sets,
            "num_duplicate_sets": num_duplicate_sets,
            "num_near_duplicates": num_near_duplicates,
        }

        params_dict = {
            "metric": self.metric,
            "k": self.k,
            "threshold": self.threshold,
            "similarity_threshold": self.similarity_threshold,
            "exact_duplicates_only": self.exact_duplicates_only,
        }

        N = knn_graph.shape[0]
        dists = knn_graph.data.reshape(N, -1)[:, 0]
        nn_ids = knn_graph.indices.reshape(N, -1)[:, 0]

        knn_info_dict = {
            "nearest_neighbor": nn_ids.tolist(),
            "distance_to_nearest_neighbor": dists.tolist(),
            "median_distance_to_nearest_neighbor": median_nn_distance,
        }

        statistics_dict = self._build_statistics_dictionary(knn_graph=knn_graph)

        info_dict = {
            **issues_dict,
            **params_dict,
            **knn_info_dict,
            **statistics_dict,
        }
        return info_dict

    def _build_statistics_dictionary(self, knn_graph: csr_matrix) -> Dict[str, Dict[str, Any]]:
        statistics_dict: Dict[str, Dict[str, Any]] = {"statistics": {}}

        # Add the knn graph as a statistic if necessary
        graph_key = "weighted_knn_graph"
        old_knn_graph = self.datalab.get_info("statistics").get(graph_key, None)
        old_graph_exists = old_knn_graph is not None
        prefer_new_graph = (
            not old_graph_exists
            or (old_knn_graph is not None and knn_graph.nnz > old_knn_graph.nnz)
            or self.metric != self.datalab.get_info("statistics").get("knn_metric", None)
        )
        if prefer_new_graph:
            statistics_dict["statistics"][graph_key] = knn_graph
            if self.metric is not None:
                statistics_dict["statistics"]["knn_metric"] = self.metric

        return statistics_dict

    def _set_threshold(
        self,
        threshold: float,
    ) -> float:
        """Computes nearest-neighbors thresholding for near-duplicate detection."""
        if threshold < 0:
            warnings.warn(
                f"Computed threshold {threshold} is less than 0. "
                "Setting threshold to 0."
                "This may indicate that either the only a few examples are in the dataset, "
                "or the data is heavily skewed."
            )
            threshold = 0
        return threshold

    def _validate_parameters(
        self,
        metric: Optional[Union[str, Callable]],
        threshold: float,
        similarity_threshold: Optional[float],
        k: int,
        exact_duplicates_only: bool,
    ) -> None:
        """Validate parameters for NearDuplicateIssueManager.

        Parameters
        ----------
        metric : Optional[Union[str, Callable]]
            Distance metric to validate.
        threshold : float
            Distance threshold to validate.
        similarity_threshold : Optional[float]
            Similarity threshold to validate.
        k : int
            Number of neighbors to validate.
        exact_duplicates_only : bool
            Exact duplicates flag to validate.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        TypeError
            If any parameter has the wrong type.
        """
        # Validate similarity_threshold
        if similarity_threshold is not None:
            if not isinstance(similarity_threshold, (int, float)):
                raise TypeError(
                    f"similarity_threshold must be a numeric value, got {type(similarity_threshold)}"
                )
            if not (0 <= similarity_threshold <= 1):
                raise ValueError(
                    f"similarity_threshold must be between 0 and 1, got {similarity_threshold}"
                )

        # Validate threshold
        if not isinstance(threshold, (int, float)):
            raise TypeError(f"threshold must be a numeric value, got {type(threshold)}")

        # Validate k
        if not isinstance(k, int):
            raise TypeError(f"k must be an integer, got {type(k)}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        # Validate exact_duplicates_only
        if not isinstance(exact_duplicates_only, bool):
            raise TypeError(
                f"exact_duplicates_only must be a boolean, got {type(exact_duplicates_only)}"
            )

        # Validate metric and similarity_threshold combination
        if similarity_threshold is not None and metric != "cosine":
            warnings.warn(
                f"similarity_threshold is provided but metric is '{metric}'. "
                "similarity_threshold is only meaningful with metric='cosine'. "
                "Consider setting metric='cosine' or using threshold parameter instead."
            )

        # Warn about conflicting parameters
        if exact_duplicates_only and similarity_threshold is not None:
            warnings.warn(
                "Both exact_duplicates_only=True and similarity_threshold are specified. "
                "exact_duplicates_only will take precedence and only exact matches (distance=0) will be found."
            )

    def _estimate_memory_usage(
        self,
        n_samples: int,
        n_features: int,
        k: int,
        dtype_size: int = 8,
    ) -> Dict[str, float]:
        """Estimate memory usage for duplicate detection.

        Parameters
        ----------
        n_samples : int
            Number of samples in the dataset.
        n_features : int
            Number of features per sample.
        k : int
            Number of nearest neighbors.
        dtype_size : int
            Size of data type in bytes (default: 8 for float64).

        Returns
        -------
        memory_estimates : Dict[str, float]
            Dictionary containing memory estimates in MB for different components.
        """

        # Estimate feature matrix memory (n_samples × n_features)
        features_memory_mb = (n_samples * n_features * dtype_size) / (1024 * 1024)

        # Estimate k-NN graph memory (sparse matrix with n_samples × k non-zero entries)
        # Each entry needs: data (float), indices (int), indptr (int)
        knn_graph_memory_mb = (n_samples * k * (dtype_size + 4 + 4)) / (1024 * 1024)

        # Estimate distance computation memory (temporary arrays)
        # This can be significant for large datasets during k-NN computation
        distance_computation_mb = (n_samples * n_samples * dtype_size) / (1024 * 1024)

        # Estimate total working memory (peak usage)
        # This includes features + k-NN graph + temporary computation arrays
        total_memory_mb = features_memory_mb + knn_graph_memory_mb + (distance_computation_mb * 0.1)

        return {
            "features_memory_mb": features_memory_mb,
            "knn_graph_memory_mb": knn_graph_memory_mb,
            "distance_computation_mb": distance_computation_mb,
            "total_memory_mb": total_memory_mb,
        }

    def _check_memory_requirements(
        self,
        features: Optional[npt.NDArray] = None,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
    ) -> None:
        """Check memory requirements and issue warnings for large datasets.

        Parameters
        ----------
        features : Optional[npt.NDArray]
            Feature matrix (if available).
        n_samples : Optional[int]
            Number of samples (if features not provided).
        n_features : Optional[int]
            Number of features (if features not provided).
        """

        # Get dataset dimensions
        if features is not None:
            n_samples, n_features = features.shape
            dtype_size = features.dtype.itemsize
        else:
            if n_samples is None or n_features is None:
                return  # Cannot estimate without dimensions
            dtype_size = 8  # Assume float64

        # Estimate memory usage
        memory_estimates = self._estimate_memory_usage(n_samples, n_features, self.k, dtype_size)

        # Define memory warning thresholds (in MB)
        WARNING_THRESHOLD_MB = 1000  # 1 GB
        CRITICAL_THRESHOLD_MB = 4000  # 4 GB

        total_memory = memory_estimates["total_memory_mb"]

        if total_memory > CRITICAL_THRESHOLD_MB:
            warnings.warn(
                f"Large dataset detected: estimated memory usage is {total_memory:.1f} MB "
                f"({total_memory/1024:.1f} GB). This may cause memory issues.\n"
                f"Recommendations:\n"
                f"  • Reduce k parameter (current: {self.k}, try: {min(5, max(1, self.k//2))})\n"
                f"  • Use exact_duplicates_only=True for preprocessing\n"
                f"  • Process data in smaller batches\n"
                f"  • Consider using a more powerful machine with more RAM\n"
                f"Dataset size: {n_samples:,} samples × {n_features:,} features",
                UserWarning,
                stacklevel=3,
            )
        elif total_memory > WARNING_THRESHOLD_MB:
            warnings.warn(
                f"Medium-large dataset detected: estimated memory usage is {total_memory:.1f} MB "
                f"({total_memory/1024:.1f} GB). Monitor memory usage during processing.\n"
                f"If you encounter memory issues, consider:\n"
                f"  • Reducing k parameter (current: {self.k})\n"
                f"  • Using exact_duplicates_only=True if only exact duplicates are needed\n"
                f"Dataset size: {n_samples:,} samples × {n_features:,} features",
                UserWarning,
                stacklevel=3,
            )

        # Additional warnings for very large k values
        if self.k > 50:
            warnings.warn(
                f"Large k value detected (k={self.k}). This increases memory usage and computation time. "
                f"For most duplicate detection tasks, k=5-20 is sufficient. "
                f"Consider reducing k for better performance.",
                UserWarning,
                stacklevel=3,
            )


def _compute_scores_with_exp_transform(nn_distances: np.ndarray, temperature: float) -> np.ndarray:
    r"""Compute near-duplicate scores from nearest neighbor distances.

    This is a non-linear transformation of the nearest neighbor distances that
    maps distances to scores in the range [0, 1].

    Note
    ----

    This transformation is given by the following formula:

    .. math::

        \text{score}(d, t) = 1 - e^{-dt}

    where :math:`d` is the nearest neighbor distance and :math:`t > 0` is a temperature parameter.

    Parameters
    ----------
    nn_distances :
        The nearest neighbor distances for each example.

    Returns
    -------
    scores :
        The near-duplicate scores for each example. The scores are in the range [0, 1].
        A lower score indicates that an example is more likely to be a near-duplicate than
        an example with a higher score.
        A score of 0 indicates that an example has an exact duplicate.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")

    scores = 1 - np.exp(-temperature * nn_distances)

    # Ensure that for nn_distances approximately equal to 0, the score is set to 0
    inds = np.isclose(nn_distances, 0)
    scores[inds] = 0

    return scores

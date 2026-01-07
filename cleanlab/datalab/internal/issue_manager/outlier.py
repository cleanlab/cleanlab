from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Tuple

from scipy.sparse import csr_matrix
from scipy.stats import iqr
import numpy as np
import pandas as pd

from cleanlab.datalab.internal.issue_manager import IssueManager
from cleanlab.datalab.internal.issue_manager.knn_graph_helpers import knn_exists, set_knn_graph
from cleanlab.internal.outlier import correct_precision_errors
from cleanlab.outlier import OutOfDistribution, transform_distances_to_scores

if TYPE_CHECKING:  # pragma: no cover
    from sklearn.neighbors import NearestNeighbors
    import numpy.typing as npt
    from cleanlab.datalab.datalab import Datalab
    from cleanlab.typing import Metric


class OutlierIssueManager(IssueManager):
    """Manages issues related to out-of-distribution examples."""

    description: ClassVar[
        str
    ] = """Examples that are very different from the rest of the dataset
    (i.e. potentially out-of-distribution or rare/anomalous instances).
    """
    issue_name: ClassVar[str] = "outlier"
    verbosity_levels = {
        0: [],
        1: [],
        2: ["average_ood_score"],
        3: [],
    }

    DEFAULT_THRESHOLDS = {
        "features": 0.37037,
        "pred_probs": 0.13,
    }
    """Default thresholds for outlier detection.

    If outlier detection is performed on the features, an example whose average
    distance to their k nearest neighbors is greater than
    Q3_avg_dist + (1 / threshold - 1) * IQR_avg_dist is considered an outlier.

    If outlier detection is performed on the predicted probabilities, an example
    whose average score is lower than threshold * median_outlier_score is
    considered an outlier.
    """

    def __init__(
        self,
        datalab: Datalab,
        k: int = 10,
        t: int = 1,
        metric: Optional[Metric] = None,
        scaling_factor: Optional[float] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(datalab)

        ood_kwargs = kwargs.get("ood_kwargs", {})

        valid_ood_params = OutOfDistribution.DEFAULT_PARAM_DICT.keys()
        params = {
            key: value
            for key, value in ((k, kwargs.get(k, None)) for k in valid_ood_params)
            if value is not None
        }

        # Simplified API: directly specify k and metric instead of NearestNeighbors object
        # This reduces dependency on OutOfDistribution and aligns with Datalab's approach
        params["k"] = k
        self.k = k
        self.t = t
        self.metric: Optional[Metric] = metric
        self.scaling_factor = scaling_factor

        if params:
            ood_kwargs["params"] = params

        # OutOfDistribution still used for pred-prob based outlier detection
        self.ood: OutOfDistribution = OutOfDistribution(**ood_kwargs)

        self._find_issues_inputs: Dict[str, bool] = {
            "features": False,
            "pred_probs": False,
            "knn_graph": False,
        }

        # Used for both methods of outlier detection
        self.threshold = threshold

    def find_issues(
        self,
        features: Optional[npt.NDArray] = None,
        pred_probs: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        statistics = self.datalab.get_info("statistics")

        # Determine if we can use kNN-based outlier detection
        knn_graph_works: bool = self._knn_graph_works(features, kwargs, statistics, self.k)
        knn_graph = None
        knn = None
        if knn_graph_works:
            # Set up or retrieve the kNN graph
            knn_graph, self.metric, knn = set_knn_graph(
                features=features,
                find_issues_kwargs=kwargs,
                metric=self.metric,
                k=self.k,
                statistics=statistics,
            )

            # Compute distances and thresholds for outlier detection
            distances = knn_graph.data.reshape(knn_graph.shape[0], -1)
            assert isinstance(distances, np.ndarray)
            (
                self.threshold,
                issue_threshold,  # Useful info for detecting issues in test data
                is_issue_column,
            ) = self._compute_threshold_and_issue_column_from_distances(distances, self.threshold)

            # Calculate outlier scores based on average distances
            avg_distances = distances.mean(axis=1)
            median_avg_distance = np.median(avg_distances)
            self._find_issues_inputs.update({"knn_graph": True})

            # Ensure scaling factor is not too small to avoid numerical issues
            if self.scaling_factor is None:
                self.scaling_factor = float(
                    max(median_avg_distance, 100 * np.finfo(np.float64).eps)
                )
            scores = transform_distances_to_scores(
                avg_distances, t=self.t, scaling_factor=self.scaling_factor
            )

            # Apply precision error correction if metric is available
            _metric = self.metric
            if _metric is not None:
                _metric = _metric if isinstance(_metric, str) else _metric.__name__
                scores = correct_precision_errors(scores, avg_distances, _metric)
        elif pred_probs is not None:
            # Fallback to prediction probabilities-based outlier detection
            scores = self._score_with_pred_probs(pred_probs, **kwargs)
            self._find_issues_inputs.update({"pred_probs": True})

            # Set threshold for pred_probs-based detection
            if self.threshold is None:
                self.threshold = self.DEFAULT_THRESHOLDS["pred_probs"]
            if not 0 <= self.threshold:
                raise ValueError(f"threshold must be non-negative, but got {self.threshold}.")
            issue_threshold = float(
                self.threshold * np.median(scores)
            )  # Useful info for detecting issues in test data
            is_issue_column = scores < issue_threshold

        else:
            # Handle case where neither kNN nor pred_probs-based detection is possible
            if (
                kwargs.get("knn_graph", None) is not None
                or statistics.get("weighted_knn_graph", None) is not None
            ):
                raise ValueError(
                    "knn_graph is provided, but not sufficiently large to compute the scores based on the provided hyperparameters."
                )
            raise ValueError(f"Either features pred_probs must be provided.")

        # Store results
        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": is_issue_column,
                self.issue_score_key: scores,
            },
        )

        self.summary = self.make_summary(score=scores.mean())

        self.info = self.collect_info(issue_threshold=issue_threshold, knn_graph=knn_graph, knn=knn)

    def _knn_graph_works(self, features, kwargs, statistics, k: int) -> bool:
        """Decide whether to skip the knn-based outlier detection and rely on pred_probs instead."""
        sufficient_knn_graph_available = knn_exists(kwargs, statistics, k)
        return (features is not None) or sufficient_knn_graph_available

    def _compute_threshold_and_issue_column_from_distances(
        self, distances: np.ndarray, threshold: Optional[float] = None
    ) -> Tuple[float, float, np.ndarray]:
        avg_distances = distances.mean(axis=1)
        if threshold:
            if not (isinstance(threshold, (int, float)) and 0 <= threshold <= 1):
                raise ValueError(
                    f"threshold must be a number between 0 and 1, got {threshold} of type {type(threshold)}."
                )
        if threshold is None:
            threshold = OutlierIssueManager.DEFAULT_THRESHOLDS["features"]

        def compute_issue_threshold(avg_distances: np.ndarray, threshold: float) -> float:
            q3_distance = np.percentile(avg_distances, 75)
            iqr_scale = 1 / threshold - 1 if threshold != 0 else np.inf
            issue_threshold = q3_distance + iqr_scale * iqr(avg_distances)
            return float(issue_threshold)

        issue_threshold = compute_issue_threshold(avg_distances, threshold)
        return threshold, issue_threshold, avg_distances > issue_threshold

    def collect_info(
        self,
        *,
        issue_threshold: float,
        knn_graph: Optional[csr_matrix],
        knn: Optional["NearestNeighbors"],
    ) -> dict:
        issues_dict = {
            "average_ood_score": self.issues[self.issue_score_key].mean(),
            "threshold": self.threshold,
            "issue_threshold": issue_threshold,
        }
        pred_probs_issues_dict: Dict[str, Any] = {}
        feature_issues_dict = {}

        if knn_graph is not None:
            N = knn_graph.shape[0]
            k = knn_graph.nnz // N
            dists = knn_graph.data.reshape(N, -1)[:, 0]
            nn_ids = knn_graph.indices.reshape(N, -1)[:, 0]

            feature_issues_dict.update(
                {
                    "k": self.k,  # type: ignore[union-attr]
                    "nearest_neighbor": nn_ids.tolist(),
                    "distance_to_nearest_neighbor": dists.tolist(),
                    "metric": self.metric,  # type: ignore[union-attr]
                    "scaling_factor": self.scaling_factor,
                    "t": self.t,
                    "knn": knn,
                }
            )

        if self.ood.params["confident_thresholds"] is not None:
            pass  #
        statistics_dict = self._build_statistics_dictionary(knn_graph=knn_graph)
        ood_params_dict = {
            "ood": self.ood,
            **self.ood.params,
        }
        knn_dict = {
            **pred_probs_issues_dict,
            **feature_issues_dict,
        }
        info_dict: Dict[str, Any] = {
            **issues_dict,
            **ood_params_dict,  # type: ignore[arg-type]
            **knn_dict,
            **statistics_dict,
            "find_issues_inputs": self._find_issues_inputs,
        }
        return info_dict

    def _build_statistics_dictionary(
        self, *, knn_graph: Optional[csr_matrix]
    ) -> Dict[str, Dict[str, Any]]:
        statistics_dict: Dict[str, Dict[str, Any]] = {"statistics": {}}

        # Add the knn graph as a statistic if necessary
        graph_key = "weighted_knn_graph"
        old_knn_graph = self.datalab.get_info("statistics").get(graph_key, None)
        old_graph_exists = old_knn_graph is not None
        prefer_new_graph = (
            not old_graph_exists
            or (
                isinstance(knn_graph, csr_matrix)
                and old_knn_graph is not None
                and knn_graph.nnz > old_knn_graph.nnz
            )
            or self.metric != self.datalab.get_info("statistics").get("knn_metric", None)
        )
        if prefer_new_graph:
            if knn_graph is not None:
                statistics_dict["statistics"][graph_key] = knn_graph
        if self.metric is not None:
            statistics_dict["statistics"]["knn_metric"] = self.metric

        return statistics_dict

    def _score_with_pred_probs(self, pred_probs: np.ndarray, **kwargs) -> np.ndarray:
        # Remove "threshold" from kwargs if it exists
        kwargs.pop("threshold", None)
        labels = self.datalab.labels
        if not isinstance(labels, np.ndarray):
            error_msg = (
                f"labels must be a numpy array of shape (n_samples,) to use the OutlierIssueManager "
                f"with pred_probs, but got {type(labels)}."
            )
            raise TypeError(error_msg)
        scores = self.ood.fit_score(pred_probs=pred_probs, labels=labels, **kwargs)
        return scores

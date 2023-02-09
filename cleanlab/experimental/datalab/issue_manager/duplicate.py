from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from cleanlab.experimental.datalab import data as datalab_data
from cleanlab.experimental.datalab.issue_manager import IssueManager

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab import Datalab


class NearDuplicateIssueManager(IssueManager):
    """Manages issues realted to near-duplicate examples."""

    issue_name: str = "near_duplicate"

    def __init__(
        self,
        datalab: Datalab,
        metric: Optional[str] = None,
        threshold: Optional[float] = None,
        k: int = 10,
        **_,
    ):
        super().__init__(datalab)
        self.metric = metric
        self.threshold = threshold
        self.k = k
        self.knn: Optional[NearestNeighbors] = None
        self.near_duplicate_sets: List[List[int]] = []
        self._embeddings: Optional[npt.NDArray] = None
        self.distances: Optional[npt.NDArray] = None

    def find_issues(
        self,
        features: npt.NDArray,
        **_,
    ) -> None:

        self._embeddings = features
        if self.knn is None:
            if self.metric is None:
                self.metric = "cosine" if features.shape[1] > 3 else "euclidean"
            self.knn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)

        if self.metric and self.metric != self.knn.metric:
            warnings.warn(
                f"Metric {self.metric} does not match metric {self.knn.metric} used to fit knn. "
                "Most likely an existing NearestNeighbors object was passed in, but a different "
                "metric was specified."
            )
        self.metric = self.knn.metric

        try:
            check_is_fitted(self.knn)
        except:
            self.knn.fit(self._embeddings)

        scores, self.distances = self._score_features(self._embeddings)
        self.radius, self.threshold = self._compute_threshold_and_radius()

        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": scores < self.threshold,
                self.issue_score_key: scores,
            },
        )

        indices = self.knn.radius_neighbors(self._embeddings, self.radius, return_distance=False)
        self.near_duplicate_sets = [
            duplicates[duplicates != idx] for idx, duplicates in enumerate(indices)
        ]

        self.summary = self.get_summary(score=scores.mean())
        self.info = self.collect_info()

    def collect_info(self) -> dict:
        issues_dict = {
            "num_near_duplicate_issues": len(self.near_duplicate_sets),
            "average_near_duplicate_score": self.issues[self.issue_score_key].mean(),
            "near_duplicate_sets": self.near_duplicate_sets,
            "radius": self.radius,
        }

        params_dict = {
            "metric": self.metric,
            "k": self.k,
            "threshold": self.threshold,
        }

        weighted_knn_graph = self.knn.kneighbors_graph(mode="distance")  # type: ignore[union-attr]
        dists, nn_ids = self._query_knn_graph(weighted_knn_graph)

        knn_info_dict = {
            "nearest_neighbor": nn_ids.tolist(),
            "distance_to_nearest_neighbor": dists.tolist(),
            "weighted_knn_graph": weighted_knn_graph.toarray().tolist(),
        }

        info_dict = {
            **issues_dict,
            **params_dict,
            **knn_info_dict,
        }
        return info_dict

    def _query_knn_graph(self, weighted_knn_graph) -> Tuple[np.ndarray, np.ndarray]:
        """Find the nearest neighbor for each example and the distance to it in the weighted
        knn graph.

        Parameters:
        -----------
        weighted_knn_graph: scipy.sparse.csr_matrix
            The weighted knn graph is an NxN matrix in a sparse CSR format, where N is
            the number of examples. Each row has `self.k` non-zero entries on the off-diagonal,
            where `self.k` is the number of nearest neighbors. The non-zero entries are the
            distances to the closest neighbors.

        Returns:
        --------
        dists: np.ndarray
            The distances to the nearest neighbors for each example.

        nn_ids: np.ndarray
            The indices of the nearest neighbors for each example.
        """
        indices = weighted_knn_graph.indices.reshape(-1, self.k)
        reshaped_distances = weighted_knn_graph.data.reshape(weighted_knn_graph.shape[0], -1)
        nn_ids = np.take_along_axis(
            indices, np.argmin(reshaped_distances, axis=1)[:, None], axis=1
        ).flatten()
        dists = np.min(reshaped_distances, axis=1)
        return dists, nn_ids

    def _score_features(self, feature_array) -> Tuple[np.ndarray, np.ndarray]:
        """Computes nearest-neighbor distances and near-duplicate scores for input features"""
        knn = cast(NearestNeighbors, self.knn)
        distances, _ = knn.kneighbors(feature_array)
        distances = distances[:, 1]  # nearest neighbor is always itself

        scores = np.tanh(distances)
        return scores, distances

    def _compute_threshold_and_radius(self) -> Tuple[float, float]:
        """Computes the radius for nearest-neighbors thresholding"""
        if self.threshold is None:
            no_exact_duplicates = self.distances[self.distances != 0]
            median_nonzero_distance = np.median(
                no_exact_duplicates
            )  # get median nonzero nearest-neighbor distance
            radius = median_nonzero_distance * 0.1
            threshold = np.tanh(radius)
        else:
            threshold = self.threshold
            radius = np.arctanh(self.threshold)
        return radius, threshold

    @property
    def verbosity_levels(self) -> Dict[int, Any]:
        return {
            0: {
                "issue": ["near_duplicate_sets"]
            },  # This is important information, but the output could be very large. Maybe it shouldn't be default
            1: {"summary": ["num_near_duplicate_issues"]},
            2: {"issue": ["nearest_neighbor", "distance_to_nearest_neighbor"]},
        }

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from cleanlab.experimental.datalab.issue_manager import IssueManager

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab import Datalab


class NearDuplicateIssueManager(IssueManager):
    """Manages issues realted to near-duplicate examples."""

    issue_name: str = "near_duplicate"

    def __init__(
        self,
        datalab: Datalab,
        metric: Optional[str] = "cosine",
        threshold: Optional[float] = None,
        k: Optional[int] = 10,
        **_,
    ):
        super().__init__(datalab)
        self.metric = metric
        self.threshold = threshold
        self.k = k
        self.knn = None
        self.near_duplicate_sets: List[List[int]] = []

    def find_issues(
        self,
        features: List[str],
        **_,
    ) -> None:

        feature_array = self._extract_embeddings(features)
        if self.knn is None:
            self.knn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)

        try:
            check_is_fitted(self.knn)
        except:
            self.knn.fit(feature_array)

        scores, distances = self._score_features(feature_array)
        self.radius, self.threshold = self._compute_threshold_and_radius()

        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": scores < self.threshold,
                self.issue_score_key: scores,
            },
        )

        indices = self.knn.radius_neighbors(feature_array, self.radius, return_distance=False)
        self.near_duplicate_sets = [
            duplicates[duplicates != idx] for idx, duplicates in enumerate(indices)
        ]
        self.distances = distances

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

        knn = self.knn
        dists, nn_ids = [array[:, 0] for array in knn.kneighbors()]  # type: ignore[union-attr]
        weighted_knn_graph = knn.kneighbors_graph(mode="distance").toarray()  # type: ignore[union-attr]

        # TODO: Reverse the order of the calls to knn.kneighbors() and knn.kneighbors_graph()
        #   to avoid computing the (distance, id) pairs twice.
        knn_info_dict = {
            "nearest_neighbour": nn_ids.tolist(),
            "distance_to_nearest_neighbour": dists.tolist(),
            # TODO Check scipy-dependency
            "weighted_knn_graph": weighted_knn_graph.tolist(),
        }

        info_dict = {
            **issues_dict,
            **params_dict,
            **knn_info_dict,
        }
        return info_dict

    def _extract_embeddings(self, columns: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Extracts embeddings for the given columns."""

        if isinstance(columns, list):
            raise NotImplementedError("TODO: Support list of columns.")

        format_kwargs = kwargs.get("format_kwargs", {})

        return self.datalab.data.with_format("numpy", **format_kwargs)[columns]

    def _score_features(self, feature_array) -> Tuple[np.ndarray, np.ndarray]:
        """Computes nearest-neighbor distances and near-duplicate scores for input features"""
        distances, neighbor_indices = self.knn.kneighbors(feature_array)
        distances = distances[:, 1]  # nearest neighbor is always itself

        self.distances = distances

        scores = np.tanh(distances)
        return scores, distances

    def _compute_threshold_and_radius(self) -> float:
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
            2: {"issue": ["nearest_neighbor", "distance_to_nearest_neighbour"]},
        }

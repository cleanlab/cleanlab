# Copyright (C) 2017-2023  Cleanlab Inc.
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with cleanlab.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from cleanlab.experimental.datalab.issue_manager import IssueManager
from cleanlab.experimental.datalab.knn import KNN

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab.experimental.datalab.datalab import Datalab


class NearDuplicateIssueManager(IssueManager):
    """Manages issues related to near-duplicate examples."""

    description: ClassVar[
        str
    ] = """A near-duplicate issue refers to two or more examples in
        a dataset that are extremely similar to each other, relative
        to the rest of the dataset.

        This may be reflected in the high similarity of the examples'
        feature embeddings if the examples are represented as vectors.

        Including near-duplicate examples in a dataset may negatively impact
        a model's generalization performance and may lead to overfitting.

        Near duplicated examples may record the same information with different:
            - Abbreviations, misspellings, typos, etc. in text data.
            - Text formatting, such as bold, italics, etc. in text data.
            - Compression formats in audio, image, and video data.
            - Sampling rates in audio and video data.
            - Resolutions in image and video data.
        """
    issue_name: ClassVar[str] = "near_duplicate"
    verbosity_levels = {
        0: [],
        1: ["threshold"],
        2: [],
    }

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

    def find_issues(
        self,
        features: Optional[npt.NDArray] = None,
        **kwargs,
    ) -> None:
        knn_graph = self._process_knn_graph_from_inputs(kwargs)
        old_knn_metric = self.datalab.get_info("statistics").get("knn_metric")
        metric_changes = self.metric and self.metric != old_knn_metric

        if knn_graph is None or metric_changes:
            if features is None:
                raise ValueError(
                    "If a knn_graph is not provided, features must be provided to fit a new knn."
                )
            if self.knn is None:
                if self.metric is None:
                    self.metric = "cosine" if features.shape[1] > 3 else "euclidean"
                self.knn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
            knn = KNN(n_neighbors=self.k, knn=self.knn)

            if self.metric and self.metric != knn.metric:
                warnings.warn(
                    f"Metric {self.metric} does not match metric {knn.metric} used to fit knn. "
                    "Most likely an existing NearestNeighbors object was passed in, but a different "
                    "metric was specified."
                )
            self.metric = knn.metric

            try:
                check_is_fitted(self.knn)
            except:
                knn.fit(features)

            knn_graph = knn.kneighbors_graph()
        N = knn_graph.shape[0]
        nn_distances = knn_graph.data.reshape(N, -1)[:, 0]
        scores = np.tanh(nn_distances)

        self.threshold = self._compute_threshold(nn_distances)

        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": nn_distances < self.threshold,
                self.issue_score_key: scores,
            },
        )

        self.near_duplicate_sets = self._neighbors_within_radius(knn_graph, self.threshold)

        self.summary = self.make_summary(score=scores.mean())
        self.info = self.collect_info(knn_graph=knn_graph)

    @staticmethod
    def _neighbors_within_radius(knn_graph: csr_matrix, radius: float):
        """Returns a list of lists of indices of near-duplicate examples.

        Each list of indices represents a set of near-duplicate examples.

        If the list is empty for a given example, then that example is not
        a near-duplicate of any other example.
        """

        N = knn_graph.shape[0]
        distances = knn_graph.data.reshape(N, -1)
        # Create a mask for the threshold
        mask = distances < radius

        # Update the indptr to reflect the new number of neighbors
        indptr = np.zeros(knn_graph.indptr.shape, dtype=knn_graph.indptr.dtype)
        indptr[1:] = np.cumsum(mask.sum(axis=1))

        # Filter the knn_graph based on the threshold
        indices = knn_graph.indices[mask.ravel()]
        near_duplicate_sets = [indices[indptr[i] : indptr[i + 1]] for i in range(N)]

        return near_duplicate_sets

    def _process_knn_graph_from_inputs(self, kwargs: Dict[str, Any]) -> Union[csr_matrix, None]:
        """Determine if a knn_graph is provided in the kwargs or if one is already stored in the associated Datalab instance."""
        knn_graph_kwargs: Optional[csr_matrix] = kwargs.get("knn_graph", None)
        knn_graph_stats = self.datalab.get_info("statistics").get("weighted_knn_graph", None)

        knn_graph: Optional[csr_matrix] = None
        if knn_graph_kwargs is not None:
            knn_graph = knn_graph_kwargs
        elif knn_graph_stats is not None:
            knn_graph = knn_graph_stats

        if isinstance(knn_graph, csr_matrix) and kwargs.get("k", 0) > (
            knn_graph.nnz // knn_graph.shape[0]
        ):
            # If the provided knn graph is insufficient, then we need to recompute the knn graph
            # with the provided features
            knn_graph = None
        return knn_graph

    def collect_info(self, knn_graph: csr_matrix) -> dict:
        issues_dict = {
            "average_near_duplicate_score": self.issues[self.issue_score_key].mean(),
            "near_duplicate_sets": self.near_duplicate_sets,
        }

        params_dict = {
            "metric": self.metric,
            "k": self.k,
            "threshold": self.threshold,
        }

        N = knn_graph.shape[0]
        dists = knn_graph.data.reshape(N, -1)[:, 0]
        nn_ids = knn_graph.indices.reshape(N, -1)[:, 0]

        knn_info_dict = {
            "nearest_neighbor": nn_ids.tolist(),
            "distance_to_nearest_neighbor": dists.tolist(),
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
            or knn_graph.nnz > old_knn_graph.nnz
            or self.metric != self.datalab.get_info("statistics").get("knn_metric", None)
        )
        if prefer_new_graph:
            statistics_dict["statistics"][graph_key] = knn_graph
            if self.metric is not None:
                statistics_dict["statistics"]["knn_metric"] = self.metric

        return statistics_dict

    def _compute_threshold(self, distances: npt.NDArray) -> float:
        """Computes nearest-neighbors thresholding for near-duplicate detection."""
        threshold: float  # Declare type for mypy
        if self.threshold is None:
            # Threshold based on nearest-neighbor distance/radius. Smaller radius means
            # more examples are considered near-duplicates.

            threshold = np.median(distances) * 0.05

            if threshold < 0:
                warnings.warn(
                    f"Computed threshold {threshold} is less than 0. "
                    "Setting threshold to 0."
                    "This may indicate that either the only a few examples are in the dataset, "
                    "or the data is heavily skewed."
                )
                threshold = 0
        else:
            threshold = self.threshold
        return threshold

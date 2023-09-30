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

from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Union, Tuple
import warnings
import inspect

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import DBSCAN

from cleanlab.datalab.internal.issue_manager import IssueManager
from cleanlab.rank import get_self_confidence_for_each_label

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from cleanlab.datalab.datalab import Datalab


class UnderperformingGroupIssueManager(IssueManager):
    """Manages issues related to underperforming group examples."""

    description: ClassVar[
        str
    ] = """An underperforming group refers to a collection of “hard” examples
    for which the model predictions are poor. The quality of predictions is
    computed using the :py:func:`get_self_confidence_for_each_label <cleanlab.rank.get_self_confidence_for_each_label>` function.
    """
    issue_name: ClassVar[str] = "underperforming_group"
    verbosity_levels = {
        0: [],
        1: [],
        2: ["threshold"],
    }
    OUTLIER_LABELS = (-1,)

    def __init__(
        self,
        datalab: Datalab,
        metric: Optional[str] = None,
        threshold: float = 0.1,
        k: int = 10,
        clustering_kwargs: Dict[str, Any] = {},
        **_: Any,
    ):
        super().__init__(datalab)
        self.metric = metric
        self.threshold = self._set_threshold(threshold)
        self.k = k
        self.clustering_kwargs = clustering_kwargs

    def find_issues(
        self,
        features: npt.NDArray,
        pred_probs: npt.NDArray,
        cluster_labels: npt.NDArray[np.int_] = None,
        **kwargs: Any,
    ) -> None:
        labels = self.datalab.labels
        knn_graph = self.set_knn_graph(features, kwargs)
        if cluster_labels is None:
            cluster_labels = self.perform_clustering(knn_graph)
        else:
            if self.clustering_kwargs:
                warnings.warn(
                    "`clustering_kwargs` will not be used since `cluster_labels` have been passed."
                )
        unique_cluster_labels = self.filter_cluster_labels(cluster_labels)
        if not unique_cluster_labels.size:
            raise ValueError(
                "No meaningful clusters were generated for determining underperforming group."
            )
        n_clusters = len(unique_cluster_labels)
        worst_cluster_id, worst_cluster_ratio = self.get_worst_cluster(
            cluster_labels, unique_cluster_labels, labels, pred_probs
        )
        is_issue_column = cluster_labels == worst_cluster_id
        scores = np.where(is_issue_column, worst_cluster_ratio, 1)
        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": is_issue_column,
                self.issue_score_key: scores,
            },
        )
        self.summary = self.make_summary(score=worst_cluster_ratio)
        self.info = self.collect_info(
            knn_graph=knn_graph,
            n_clusters=n_clusters,
            cluster_labels=cluster_labels,
        )

    def set_knn_graph(
        self, features: npt.NDArray, find_issues_kwargs: Dict[str, Any]
    ) -> csr_matrix:
        knn_graph = self._process_knn_graph_from_inputs(find_issues_kwargs)
        old_knn_metric = self.datalab.get_info("statistics").get("knn_metric")
        metric_changes = self.metric and self.metric != old_knn_metric

        if knn_graph is None or metric_changes:
            if features is None:
                raise ValueError(
                    "If a knn_graph is not provided, features must be provided to fit a new knn."
                )
            if self.metric is None:
                self.metric = "cosine" if features.shape[1] > 3 else "euclidean"
            knn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)

            if self.metric and self.metric != knn.metric:
                warnings.warn(
                    f"Metric {self.metric} does not match metric {knn.metric} used to fit knn. "
                    "Most likely an existing NearestNeighbors object was passed in, but a different "
                    "metric was specified."
                )
            self.metric = knn.metric

            try:
                check_is_fitted(knn)
            except NotFittedError:
                knn.fit(features)
            knn_graph = knn.kneighbors_graph(mode="distance")
        return knn_graph

    def perform_clustering(self, knn_graph: csr_matrix) -> npt.NDArray[np.int_]:
        DBSCAN_VALID_KEYS = inspect.signature(DBSCAN).parameters.keys()
        dbscan_params = {
            key: value
            for key, value in ((k, self.clustering_kwargs.get(k, None)) for k in DBSCAN_VALID_KEYS)
            if value is not None
        }
        dbscan_params["metric"] = "precomputed"
        clusterer = DBSCAN(**dbscan_params)
        clusterer.fit(knn_graph)
        knn_graph.eliminate_zeros()  # Reset KNN Graph
        cluster_labels = clusterer.labels_
        return cluster_labels

    def filter_cluster_labels(self, cluster_labels: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        unique_cluster_labels = np.array(
            [label for label in set(cluster_labels) if label not in self.OUTLIER_LABELS]
        )
        return unique_cluster_labels

    def get_worst_cluster(
        self,
        cluster_labels: npt.NDArray[np.int_],
        unique_cluster_labels: npt.NDArray[np.int_],
        labels: npt.NDArray,
        pred_probs: npt.NDArray,
    ) -> Tuple[int, float]:
        worst_cluster_performance = 1  # Largest possible probability value
        worst_cluster_id = min(unique_cluster_labels) - 1
        for cluster_id in unique_cluster_labels:
            cluster_mask = cluster_labels == cluster_id
            cur_cluster_labels = labels[cluster_mask]
            cur_cluster_pred_probs = pred_probs[cluster_mask]
            cluster_performance = get_self_confidence_for_each_label(
                cur_cluster_labels, cur_cluster_pred_probs
            ).mean()
            if cluster_performance < worst_cluster_performance:
                worst_cluster_performance = cluster_performance
                worst_cluster_id = cluster_id
        mean_performance = get_self_confidence_for_each_label(labels, pred_probs).mean()
        worst_cluster_ratio = min(worst_cluster_performance / mean_performance, 1.0)
        worst_cluster_id = (
            worst_cluster_id
            if worst_cluster_ratio < self.threshold
            else max(unique_cluster_labels) + 1
        )
        return worst_cluster_id, worst_cluster_ratio

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

    def collect_info(
        self,
        knn_graph: csr_matrix,
        n_clusters: int,
        cluster_labels: npt.NDArray[np.int_],
    ) -> Dict[str, Any]:
        params_dict = {
            "metric": self.metric,
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

        cluster_stat_dict = self._get_cluster_statistics(
            n_clusters=n_clusters,
            cluster_labels=cluster_labels,
        )
        info_dict = {**params_dict, **knn_info_dict, **statistics_dict, **cluster_stat_dict}

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

    def _get_cluster_statistics(
        self,
        n_clusters: int,
        cluster_labels: npt.NDArray[np.int_],
    ) -> Dict[str, Dict[str, Any]]:
        cluster_stats: Dict[str, Dict[str, Any]] = {
            "clustering": {
                "algorithm": "DBSCAN",
                "params": {"metric": "precomputed"},
                "stats": {"n_clusters": n_clusters, "cluster_labels": cluster_labels},
            }
        }
        return cluster_stats

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

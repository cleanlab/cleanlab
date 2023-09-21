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

from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import HDBSCAN
from sklearn.base import ClusterMixin
from sklearn.metrics import silhouette_score

from cleanlab.datalab.internal.issue_manager import IssueManager
from cleanlab.rank import get_self_confidence_for_each_label
import scipy.sparse as sp

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
    OUTLIER_LABELS = (-1, -2, -3)

    def __init__(
        self,
        datalab: Datalab,
        metric: Optional[str] = None,
        threshold: float = 0.1,
        k: int = 10,
        **_,
    ):
        super().__init__(datalab)
        self.metric = metric
        self.threshold = self._set_threshold(threshold)
        self.k = k

    def find_issues(
        self,
        features: npt.NDArray,
        pred_probs: npt.NDArray,
        **kwargs,
    ) -> None:
        labels = self.datalab.labels
        knn_graph, original_knn_graph = self.set_knn_graph(features, kwargs)
        cluster_labels = self.perform_clustering(knn_graph)
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
            knn_graph=original_knn_graph,
            n_clusters=n_clusters,
            cluster_labels=cluster_labels,
        )

    def set_knn_graph(self, features: npt.NDArray, find_issues_kwargs: Dict) -> csr_matrix:
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
            symmetric_knn_graph = self._symmetrize_knn_graph(knn_graph)
        return symmetric_knn_graph, knn_graph

    def _symmetrize_knn_graph(self, knn_graph: csr_matrix) -> csr_matrix:
        binary_adjacency = (knn_graph > 0).astype(int)
        symmetric_adjacency = binary_adjacency.maximum(binary_adjacency.T)
        symmetric_distances = knn_graph + knn_graph.T.multiply(
            symmetric_adjacency - binary_adjacency
        )
        return symmetric_distances

    def perform_clustering(self, knn_graph: csr_matrix) -> npt.NDArray[np.int_]:
        # hdb = HDBSCAN(metric="precomputed")
        # hdb.fit(knn_graph)
        # cluster_labels = hdb.labels_
        # ALTERNATIVE FORMULATION: Connect the graph using a dummy node before HDBSCAN.
        # Adapted from - https://github.com/scikit-learn-contrib/hdbscan/issues/82#issuecomment-1242604639
        # knn_graph = self.connect_knn_graph(knn_graph)
        cluster_labels = self.run_hdbscan_on_components(knn_graph)

        return cluster_labels[:200]  # Slicing applicable if connect_knn_graph is called

    def run_hdbscan_on_components(
        self, knn_graph: csr_matrix, min_cluster_size: int = 5, **hdbscan_params: Dict[str, Any]
    ) -> npt.NDArray[np.int_]:
        n_components, component_labels = sp.csgraph.connected_components(
            csgraph=knn_graph, directed=False, return_labels=True
        )
        all_clusters = np.empty(knn_graph.shape[0], dtype=object)
        offset = 0
        for component_idx in range(n_components):
            component_mask = component_labels == component_idx
            component_subgraph = knn_graph[component_mask, :][:, component_mask]
            knn_subgraph = self._symmetrize_knn_graph(component_subgraph)
            clusterer = HDBSCAN(metric="precomputed")
            component_clusters = clusterer.fit_predict(knn_subgraph)
            outliers_mask = np.in1d(component_clusters, self.OUTLIER_LABELS)
            component_clusters[~outliers_mask] += offset
            all_clusters[component_mask] = component_clusters
            offset = np.max(component_clusters) + 1
        return all_clusters

    def connect_knn_graph(
        self, knn_graph: csr_matrix, new_node_dist: Optional[float] = None
    ) -> csr_matrix:
        """
        This function takes in a sparse graph (csr_matrix) that has more than
        one component (multiple unconnected subgraphs) and appends another
        node to the graph that is weakly connected to all other nodes.
        RH 2022

        Args:
            d (scipy.sparse.csr_matrix):
                Sparse graph with multiple components.
                See scipy.sparse.csgraph.connected_components
            dist_fullyConnectedNode (float):
                Value to use for the connection strengh to all other nodes.
                Value will be appended as elements in a new row and column at
                the ends of the 'd' matrix.

        Returns:
            d2 (scipy.sparse.csr_matrix):
                Sparse graph with only one component.
        """
        N = knn_graph.shape[0]
        if new_node_dist is None:
            new_node_dist = (knn_graph.max() - knn_graph.min()) * 1000
        connected_knn_graph = knn_graph.copy()
        connected_knn_graph = sp.vstack((connected_knn_graph, np.full((1, N), new_node_dist)))
        connected_knn_graph = sp.hstack((connected_knn_graph, np.full((N + 1, 1), new_node_dist)))
        return connected_knn_graph.tocsr()

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

        return knn_graph

    def collect_info(
        self,
        knn_graph: csr_matrix,
        n_clusters: int,
        cluster_labels: npt.NDArray[np.int_],
    ) -> dict:
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
            knn_graph=knn_graph,
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
        knn_graph: csr_matrix,
    ) -> Dict[str, Dict[str, Any]]:
        cluster_stats = {
            "clustering": {
                "algorithm": "HDBSCAN",
                "params": {"metric": "precomputed"},
                "stats": {"n_clusters": n_clusters, "cluster_labels": cluster_labels},
            }
        }
        try:
            sc = silhouette_score(knn_graph, cluster_labels, metric="precomputed")
            cluster_stats["clustering"]["stats"]["silhouette_score"] = sc
            if sc < 0:
                warnings.warn(
                    f"A negative silhoutte score ({sc}) could indicate that some samples have been assigned to the wrong cluster."
                )
        except ValueError as err:
            warnings.warn(
                f"Error computing Silhouette Score - {err}. \n `silhouette_score` will not be present in info."
            )
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

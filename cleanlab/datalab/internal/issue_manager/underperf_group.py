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
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import HDBSCAN
from sklearn.metrics import calinski_harabasz_score

from cleanlab.datalab.internal.issue_manager import IssueManager
from cleanlab.rank import get_self_confidence_for_each_label

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from cleanlab.datalab.datalab import Datalab


class UnderperfGroupIssueManager(IssueManager):
    """Manages issues related to near-duplicate examples."""

    description: ClassVar[
        str
    ] = """A (near) duplicate issue refers to two or more examples in
    a dataset that are extremely similar to each other, relative
    to the rest of the dataset.  The examples flagged with this issue
    may be exactly duplicated, or lie atypically close together when
    represented as vectors (i.e. feature embeddings).
    """
    issue_name: ClassVar[str] = "underperf_group"
    verbosity_levels = {
        0: [],
        1: [],
        2: ["threshold"],
    }

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
        self.near_duplicate_sets: List[List[int]] = []

    def find_issues(
        self,
        features,
        pred_probs,
        **kwargs,
    ) -> None:
        labels = self.datalab.labels
        knn_graph = self._process_knn_graph_from_inputs(kwargs)
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

            knn_graph = knn.kneighbors_graph(mode="distance", n_neighbors=features.shape[0] - 1)
        full_loss = get_self_confidence_for_each_label(labels, pred_probs).mean()
        hdb = HDBSCAN(metric="precomputed")
        hdb.fit(knn_graph)
        cluster_labels = hdb.labels_

        unique_labels = set(cluster_labels)
        n_outlier_clusters = sum(l in (-1, -2, -3) for l in unique_labels)
        n_clusters = len(unique_labels) - n_outlier_clusters

        min_loss = 10  # Assign some other value
        min_loss_clusterid = -1
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cur_cluster_labels = labels[cluster_mask]
            cur_cluster_pred_probs = pred_probs[cluster_mask]
            cluster_loss = get_self_confidence_for_each_label(
                cur_cluster_labels, cur_cluster_pred_probs
            ).mean()
            if cluster_loss < min_loss:
                min_loss = cluster_loss
                min_loss_clusterid = i
        loss_ratio = min(min_loss / full_loss, 1.0)
        print(min_loss, full_loss)
        min_loss_clusterid = min_loss_clusterid if loss_ratio < self.threshold else n_clusters
        is_issue_column = cluster_labels == min_loss_clusterid
        scores = np.where(is_issue_column, loss_ratio, 1)
        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": is_issue_column,
                self.issue_score_key: scores,
            },
        )
        self.summary = self.make_summary(score=loss_ratio)
        self.info = self.collect_info(
            knn_graph=knn_graph, n_clusters=n_clusters, cluster_labels=cluster_labels
        )

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
        self, knn_graph: csr_matrix, n_clusters: int, cluster_labels: np.ndarray
    ) -> dict:

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

        cluster_stat_dict = self._get_cluster_statistics(n_clusters=n_clusters)
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

    def _get_cluster_statistics(self, n_clusters: int) -> Dict[str, Dict[str, Any]]:
        cluster_stats = {"HDBSCAN": {"n_clusters": n_clusters}}
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

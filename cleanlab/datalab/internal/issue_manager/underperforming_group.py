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


CLUSTERING_ALGO = "DBSCAN"
CLUSTERING_PARAMS_DEFAULT = {"metric": "precomputed"}


class UnderperformingGroupIssueManager(IssueManager):
    """
    Manages issues related to underperforming group examples.

    Note: The `min_cluster_samples` argument should not be confused with the
    `min_samples` argument of sklearn.cluster.DBSCAN.

    Examples
    --------
    >>> from cleanlab import Datalab
    >>> import numpy as np
    >>> X = np.random.normal(size=(50, 2))
    >>> y = np.random.randint(2, size=50)
    >>> pred_probs = X / X.sum(axis=1, keepdims=True)
    >>> data = {"X": X, "y": y}
    >>> lab = Datalab(data, label_name="y")
    >>> issue_types={"underperforming_group": {"clustering_kwargs": {"eps": 0.5}}}
    >>> lab.find_issues(pred_probs=pred_probs, features=X, issue_types=issue_types)
    """

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
    OUTLIER_CLUSTER_LABELS: ClassVar[Tuple[int]] = (-1,)
    """Specifies labels considered as outliers by the clustering algorithm."""
    NO_UNDERPERFORMING_CLUSTER_ID: ClassVar[int] = min(OUTLIER_CLUSTER_LABELS) - 1
    """Constant to signify absence of any underperforming cluster."""

    def __init__(
        self,
        datalab: Datalab,
        metric: Optional[str] = None,
        threshold: float = 0.1,
        k: int = 10,
        clustering_kwargs: Dict[str, Any] = {},
        min_cluster_samples: int = 5,
        **_: Any,
    ):
        super().__init__(datalab)
        self.metric = metric
        self.threshold = self._set_threshold(threshold)
        self.k = k
        self.clustering_kwargs = clustering_kwargs
        self.min_cluster_samples = min_cluster_samples

    def find_issues(
        self,
        pred_probs: npt.NDArray,
        features: Optional[npt.NDArray] = None,
        cluster_ids: Optional[npt.NDArray[np.int_]] = None,
        **kwargs: Any,
    ) -> None:
        labels = self.datalab.labels
        if not isinstance(labels, np.ndarray):
            error_msg = (
                f"Labels must be a numpy array of shape (n_samples,) for UnderperformingGroupIssueManager. "
                f"Got {type(labels)} instead."
            )
            raise TypeError(error_msg)
        if cluster_ids is None:
            knn_graph = self.set_knn_graph(features, kwargs)
            cluster_ids = self.perform_clustering(knn_graph)
            performed_clustering = True
        else:
            if self.clustering_kwargs:
                warnings.warn(
                    "`clustering_kwargs` will not be used since `cluster_ids` have been passed."
                )
            performed_clustering = False
            knn_graph = None
        unique_cluster_ids = self.filter_cluster_ids(cluster_ids)
        if not unique_cluster_ids.size:
            raise ValueError(
                "No meaningful clusters were generated for determining underperforming group."
            )
        n_clusters = len(unique_cluster_ids)
        worst_cluster_id, worst_cluster_ratio = self.get_worst_cluster(
            cluster_ids, unique_cluster_ids, labels, pred_probs
        )
        is_issue_column = cluster_ids == worst_cluster_id
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
            cluster_ids=cluster_ids,
            performed_clustering=performed_clustering,
            worst_cluster_id=worst_cluster_id,
        )

    def set_knn_graph(
        self, features: Optional[npt.NDArray], find_issues_kwargs: Dict[str, Any]
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
        """Perform clustering of datapoints using a knn graph as distance matrix.

        Args:
            knn_graph (csr_matrix): Sparse Distance Matrix.

        Returns:
            cluster_ids (npt.NDArray[np.int_]): Cluster IDs for each datapoint.
        """
        DBSCAN_VALID_KEYS = inspect.signature(DBSCAN).parameters.keys()
        dbscan_params = {
            key: value
            for key, value in ((k, self.clustering_kwargs.get(k, None)) for k in DBSCAN_VALID_KEYS)
            if value is not None
        }
        dbscan_params["metric"] = "precomputed"
        clusterer = DBSCAN(**dbscan_params)
        cluster_ids = clusterer.fit_predict(
            knn_graph.copy()
        )  # Copy to avoid modification by DBSCAN
        return cluster_ids

    def filter_cluster_ids(self, cluster_ids: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Remove outlier clusters and return IDs of clusters with at least `self.min_cluster_samples` number of datapoints.


        Args:
            cluster_ids (npt.NDArray[np.int_]): Cluster IDs for each datapoint.

        Returns:
            unique_cluster_ids (npt.NDArray[np.int_]):  List of unique cluster IDs after
            removing outlier clusters and clusters with less than `self.min_cluster_samples`
            number of datapoints.
        """
        unique_cluster_ids = np.array(
            [label for label in set(cluster_ids) if label not in self.OUTLIER_CLUSTER_LABELS]
        )
        frequencies = np.bincount(cluster_ids[~np.isin(cluster_ids, self.OUTLIER_CLUSTER_LABELS)])
        unique_cluster_ids = np.array(
            [
                cluster_id
                for cluster_id in unique_cluster_ids
                if frequencies[cluster_id] >= self.min_cluster_samples
            ]
        )
        return unique_cluster_ids

    def get_worst_cluster(
        self,
        cluster_ids: npt.NDArray[np.int_],
        unique_cluster_ids: npt.NDArray[np.int_],
        labels: npt.NDArray,
        pred_probs: npt.NDArray,
    ) -> Tuple[int, float]:
        """Get ID and quality score of underperforming cluster.

        Args:
            cluster_ids (npt.NDArray[np.int_]): _description_
            unique_cluster_ids (npt.NDArray[np.int_]): _description_
            labels (npt.NDArray): _description_
            pred_probs (npt.NDArray): _description_

        Returns:
            Tuple[int, float]: (Underperforming Cluster ID, Cluster Quality Score)
        """
        worst_cluster_performance = 1  # Largest possible probability value
        worst_cluster_id = min(unique_cluster_ids) - 1
        for cluster_id in unique_cluster_ids:
            cluster_mask = cluster_ids == cluster_id
            cur_cluster_ids = labels[cluster_mask]
            cur_cluster_pred_probs = pred_probs[cluster_mask]
            cluster_performance = get_self_confidence_for_each_label(
                cur_cluster_ids, cur_cluster_pred_probs
            ).mean()
            if cluster_performance < worst_cluster_performance:
                worst_cluster_performance = cluster_performance
                worst_cluster_id = cluster_id
        mean_performance = get_self_confidence_for_each_label(labels, pred_probs).mean()
        worst_cluster_ratio = min(worst_cluster_performance / mean_performance, 1.0)
        worst_cluster_id = (
            worst_cluster_id
            if worst_cluster_ratio < self.threshold
            else self.NO_UNDERPERFORMING_CLUSTER_ID
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
        cluster_ids: npt.NDArray[np.int_],
        performed_clustering: bool,
        worst_cluster_id: int,
    ) -> Dict[str, Any]:
        params_dict = {
            "k": self.k,
            "metric": self.metric,
            "threshold": self.threshold,
        }

        knn_info_dict = {}
        if knn_graph is not None:
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
            cluster_ids=cluster_ids,
            performed_clustering=performed_clustering,
            worst_cluster_id=worst_cluster_id,
        )
        info_dict = {
            **params_dict,
            **knn_info_dict,
            **statistics_dict,
            **cluster_stat_dict,
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
            or (isinstance(knn_graph, csr_matrix) and knn_graph.nnz > old_knn_graph.nnz)
            or self.metric != self.datalab.get_info("statistics").get("knn_metric", None)
        )
        if prefer_new_graph:
            if knn_graph is not None:
                statistics_dict["statistics"][graph_key] = knn_graph
                if self.metric is not None:
                    statistics_dict["statistics"]["knn_metric"] = self.metric

        return statistics_dict

    def _get_cluster_statistics(
        self,
        n_clusters: int,
        cluster_ids: npt.NDArray[np.int_],
        performed_clustering: bool,
        worst_cluster_id: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Get relevant cluster statistics.

        Args:
            n_clusters (int): Number of clusters
            cluster_ids (npt.NDArray[np.int_]): Cluster IDs for each datapoint.
            performed_clustering (bool): Set to True to indicate that clustering was performed on
            `features` passed to `find_issues`. Set to False to suggest that `cluster_ids` were explicitly
            passed to `find_issues`.
            worst_cluster_id (int): Uderperforming cluster ID.

        Returns:
            cluster_stats (Dict[str, Dict[str, Any]]): Cluster Statistics
        """
        cluster_stats: Dict[str, Dict[str, Any]] = {
            "clustering": {
                "algorithm": None,
                "params": {},
                "stats": {
                    "n_clusters": n_clusters,
                    "cluster_ids": cluster_ids,
                    "underperforming_cluster_id": worst_cluster_id,
                },
            }
        }
        if performed_clustering:
            cluster_stats["clustering"].update(
                {"algorithm": CLUSTERING_ALGO, "params": CLUSTERING_PARAMS_DEFAULT}
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

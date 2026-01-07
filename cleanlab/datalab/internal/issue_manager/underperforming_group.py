from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Optional, Union, Tuple
import warnings
import inspect

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN

from cleanlab.datalab.internal.issue_manager import IssueManager
from cleanlab.datalab.internal.issue_manager.knn_graph_helpers import set_knn_graph
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
    ] = """An underperforming group refers to a cluster of similar examples
    (i.e. a slice) in the dataset for which the ML model predictions
    are particularly poor (loss evaluation over this subpopulation is high).
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
        metric: Optional[Union[str, Callable]] = None,
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
            statistics = self.datalab.get_info("statistics")
            knn_graph, self.metric, _ = set_knn_graph(
                features, kwargs, self.metric, self.k, statistics
            )
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
        cluster_id_to_score, worst_cluster_id, worst_cluster_ratio = (
            self.get_underperforming_clusters(cluster_ids, unique_cluster_ids, labels, pred_probs)
        )
        is_issue_column = cluster_ids == worst_cluster_id
        scores = np.ones(is_issue_column.shape[0])
        for cluster_id, cluster_score in cluster_id_to_score.items():
            scores[cluster_ids == cluster_id] = cluster_score
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

    def get_underperforming_clusters(
        self,
        cluster_ids: npt.NDArray[np.int_],
        unique_cluster_ids: npt.NDArray[np.int_],
        labels: npt.NDArray,
        pred_probs: npt.NDArray,
    ) -> Tuple[Dict[int, float], int, float]:
        """Get ID and quality score of each underperforming cluster.

        Args:
            cluster_ids (npt.NDArray[np.int_]): Cluster IDs corresponding to each sample
            unique_cluster_ids (npt.NDArray[np.int_]): Unique cluster IDs excluding noisy clusters
            labels (npt.NDArray): Label of each sample
            pred_probs (npt.NDArray): Prediction probability

        Returns:
            Tuple[Dict[int, float], int, float]: (Cluster IDs and their scores, Worst Cluster ID, Worst Cluster Quality Score)
        """
        worst_cluster_ratio = 1.0  # Largest possible probability value
        worst_cluster_id = min(unique_cluster_ids) - 1
        # For calculating mean_performance of the dataset, choose labels and pred-probs of samples belonging to non-noisy clusters
        filtered_cluster_id_mask = np.isin(cluster_ids, unique_cluster_ids)
        filtered_labels = labels[filtered_cluster_id_mask]
        filtered_pred_probs = pred_probs[filtered_cluster_id_mask]
        mean_performance = get_self_confidence_for_each_label(
            filtered_labels, filtered_pred_probs
        ).mean()
        cluster_ids_to_score = {}
        for cluster_id in unique_cluster_ids:
            cluster_mask = cluster_ids == cluster_id
            cur_cluster_ids = labels[cluster_mask]
            cur_cluster_pred_probs = pred_probs[cluster_mask]
            cluster_performance = get_self_confidence_for_each_label(
                cur_cluster_ids, cur_cluster_pred_probs
            ).mean()
            if cluster_performance < mean_performance:
                cluster_ids_to_score[cluster_id] = cluster_performance / mean_performance
                if cluster_performance < worst_cluster_ratio:
                    worst_cluster_ratio = cluster_ids_to_score[cluster_id]
                    worst_cluster_id = cluster_id
        worst_cluster_id = (
            worst_cluster_id
            if worst_cluster_ratio < self.threshold
            else self.NO_UNDERPERFORMING_CLUSTER_ID
        )
        return cluster_ids_to_score, worst_cluster_id, worst_cluster_ratio

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

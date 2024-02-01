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

from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Tuple, Union, cast

from scipy.sparse import csr_matrix
from scipy.stats import iqr
import numpy as np
import pandas as pd

from cleanlab.datalab.internal.issue_manager import IssueManager
from cleanlab.outlier import OutOfDistribution, transform_distances_to_scores

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from sklearn.neighbors import NearestNeighbors
    from cleanlab.datalab.datalab import Datalab


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

        if params:
            ood_kwargs["params"] = params

        self.ood: OutOfDistribution = OutOfDistribution(**ood_kwargs)

        self.threshold = threshold
        self._embeddings: Optional[np.ndarray] = None
        self._metric: str = None  # type: ignore

    def find_issues(
        self,
        features: Optional[npt.NDArray] = None,
        pred_probs: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        knn_graph = self._process_knn_graph_from_inputs(kwargs)
        distances: Optional[np.ndarray] = None

        if knn_graph is not None:
            N = knn_graph.shape[0]
            k = knn_graph.nnz // N
            t = cast(int, self.ood.params["t"])
            distances = knn_graph.data.reshape(-1, k)
            assert isinstance(distances, np.ndarray)
            avg_distances = distances.mean(axis=1)
            median_avg_distance = np.median(avg_distances)
            scores = transform_distances_to_scores(
                avg_distances, t=t, scaling_factor=median_avg_distance
            )
        elif features is not None:
            scores = self._score_with_features(features, **kwargs)
        elif pred_probs is not None:
            scores = self._score_with_pred_probs(pred_probs, **kwargs)
        else:
            if kwargs.get("knn_graph", None) is not None:
                raise ValueError(
                    "knn_graph is provided, but not sufficiently large to compute the scores based on the provided hyperparameters."
                )
            raise ValueError(f"Either features pred_probs must be provided.")

        if features is not None or knn_graph is not None:
            if knn_graph is None:
                assert (
                    features is not None
                ), "features must be provided so that we can compute the knn graph."
                knn_graph = self._process_knn_graph_from_features(kwargs)
            distances = knn_graph.data.reshape(knn_graph.shape[0], -1)

            assert isinstance(distances, np.ndarray)
            (
                self.threshold,
                is_issue_column,
            ) = self._compute_threshold_and_issue_column_from_distances(distances, self.threshold)

        else:
            assert pred_probs is not None
            # Threshold based on pred_probs, very small scores are outliers
            if self.threshold is None:
                self.threshold = self.DEFAULT_THRESHOLDS["pred_probs"]
            if not 0 <= self.threshold:
                raise ValueError(f"threshold must be non-negative, but got {self.threshold}.")
            is_issue_column = scores < self.threshold * np.median(scores)

        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": is_issue_column,
                self.issue_score_key: scores,
            },
        )

        self.summary = self.make_summary(score=scores.mean())

        self.info = self.collect_info(knn_graph=knn_graph)

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

    def _compute_threshold_and_issue_column_from_distances(
        self, distances: np.ndarray, threshold: Optional[float] = None
    ) -> Tuple[float, np.ndarray]:
        avg_distances = distances.mean(axis=1)
        if threshold:
            if not (isinstance(threshold, (int, float)) and 0 <= threshold <= 1):
                raise ValueError(
                    f"threshold must be a number between 0 and 1, got {threshold} of type {type(threshold)}."
                )
        if threshold is None:
            threshold = OutlierIssueManager.DEFAULT_THRESHOLDS["features"]
        q3_distance = np.percentile(avg_distances, 75)
        iqr_scale = 1 / threshold - 1 if threshold != 0 else np.inf
        return threshold, avg_distances > q3_distance + iqr_scale * iqr(avg_distances)

    def _process_knn_graph_from_features(self, kwargs: Dict) -> csr_matrix:
        # Check if the weighted knn graph exists in info
        knn_graph = self.datalab.get_info("statistics").get("weighted_knn_graph", None)

        # Used to check if the knn graph needs to be recomputed, already set in the knn object
        k: int = 0
        if knn_graph is not None:
            k = knn_graph.nnz // knn_graph.shape[0]

        knn: NearestNeighbors = self.ood.params["knn"]  # type: ignore
        if kwargs.get("knn", None) is not None or knn.n_neighbors > k:  # type: ignore[union-attr]
            # If the pre-existing knn graph has fewer neighbors than the knn object,
            # then we need to recompute the knn graph
            assert knn == self.ood.params["knn"]  # type: ignore[union-attr]
            knn_graph = knn.kneighbors_graph(mode="distance")  # type: ignore[union-attr]
            self._metric = knn.metric  # type: ignore[union-attr]

        return knn_graph

    def collect_info(self, *, knn_graph: Optional[csr_matrix] = None) -> dict:
        issues_dict = {
            "average_ood_score": self.issues[self.issue_score_key].mean(),
            "threshold": self.threshold,
        }
        pred_probs_issues_dict: Dict[str, Any] = {}
        feature_issues_dict = {}

        if knn_graph is not None:
            knn = self.ood.params["knn"]  # type: ignore
            N = knn_graph.shape[0]
            k = knn_graph.nnz // N
            dists = knn_graph.data.reshape(N, -1)[:, 0]
            nn_ids = knn_graph.indices.reshape(N, -1)[:, 0]

            feature_issues_dict.update(
                {
                    "k": k,  # type: ignore[union-attr]
                    "nearest_neighbor": nn_ids.tolist(),
                    "distance_to_nearest_neighbor": dists.tolist(),
                }
            )
            if self.ood.params["knn"] is not None:
                knn = self.ood.params["knn"]
                feature_issues_dict.update({"metric": knn.metric})  # type: ignore[union-attr]

        if self.ood.params["confident_thresholds"] is not None:
            pass  #
        statistics_dict = self._build_statistics_dictionary(knn_graph=knn_graph)
        ood_params_dict = self.ood.params
        knn_dict = {
            **pred_probs_issues_dict,
            **feature_issues_dict,
        }
        info_dict: Dict[str, Any] = {
            **issues_dict,
            **ood_params_dict,  # type: ignore[arg-type]
            **knn_dict,
            **statistics_dict,
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
            or (isinstance(knn_graph, csr_matrix) and knn_graph.nnz > old_knn_graph.nnz)
            or self._metric != self.datalab.get_info("statistics").get("knn_metric", None)
        )
        if prefer_new_graph:
            if knn_graph is not None:
                statistics_dict["statistics"][graph_key] = knn_graph
        if self._metric is not None:
            statistics_dict["statistics"]["knn_metric"] = self._metric

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

    def _score_with_features(self, features: npt.NDArray, **kwargs) -> npt.NDArray:
        scores = self.ood.fit_score(features=features)
        return scores

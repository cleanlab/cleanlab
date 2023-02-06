from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

from cleanlab.experimental.datalab import data as datalab_data
from cleanlab.experimental.datalab.issue_manager import IssueManager
from cleanlab.outlier import OutOfDistribution

if TYPE_CHECKING:  # pragma: no cover
    from sklearn.neighbors import NearestNeighbors

    from cleanlab import Datalab


class OutOfDistributionIssueManager(IssueManager):
    """Manages issues related to out-of-distribution examples."""

    issue_name: str = "outlier"

    def __init__(
        self,
        datalab: Datalab,
        ood_kwargs: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        **_,
    ):
        super().__init__(datalab)
        self.ood: OutOfDistribution = OutOfDistribution(**(ood_kwargs or {}))
        self.threshold = threshold
        self._embeddings: Optional[np.ndarray] = None

    def find_issues(
        self,
        features: Optional[List[str]] = None,
        pred_probs: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:

        if features is not None:
            scores = self._score_with_features(features, **kwargs)
        elif pred_probs is not None:
            scores = self._score_with_pred_probs(pred_probs, **kwargs)
        else:
            raise ValueError(f"Either features or pred_probs must be provided.")

        if self.threshold is None:
            if self._embeddings is not None:
                knn: NearestNeighbors = self.ood.params["knn"]  # type: ignore
                distances, _ = knn.kneighbors(self._embeddings, n_neighbors=2)
                nn_distances = distances[:, 1]
                count_scale = min(len(nn_distances) - 1, 10)
                self.threshold = np.exp(-count_scale * np.mean(nn_distances) * self.ood.params["t"])
            else:
                self.threshold = np.percentile(scores, 10)

        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": scores < self.threshold,
                self.issue_score_key: scores,
            },
        )

        self.summary = self.get_summary(score=scores.mean())

        self.info = self.collect_info()

    def collect_info(self) -> dict:

        issues_dict = {
            "num_outlier_issues": sum(self.issues[f"is_{self.issue_name}_issue"]),
            "average_ood_score": self.issues[self.issue_score_key].mean(),
        }
        pred_probs_issues_dict: Dict[
            str, Any
        ] = {}  # TODO: Implement collect_info for pred_probs related issues
        feature_issues_dict = {}

        # Compute
        if self.ood.params["knn"] is not None:
            knn = self.ood.params["knn"]
            dists, nn_ids = [array[:, 0] for array in knn.kneighbors()]  # type: ignore[union-attr]
            weighted_knn_graph = knn.kneighbors_graph(mode="distance").toarray()  # type: ignore[union-attr]

            # TODO: Reverse the order of the calls to knn.kneighbors() and knn.kneighbors_graph()
            #   to avoid computing the (distance, id) pairs twice.
            feature_issues_dict.update(
                {
                    "nearest_neighbor": nn_ids.tolist(),
                    "distance_to_nearest_neighbor": dists.tolist(),
                    # TODO Check scipy-dependency
                    "weighted_knn_graph": weighted_knn_graph.tolist(),
                }
            )

        if self.ood.params["confident_thresholds"] is not None:
            pass  #
        ood_params_dict = self.ood.params
        knn_dict = {
            **pred_probs_issues_dict,
            **feature_issues_dict,
        }
        info_dict = {
            **issues_dict,
            **ood_params_dict,
            **knn_dict,
        }
        return info_dict

    def _score_with_pred_probs(self, pred_probs: np.ndarray, **kwargs) -> np.ndarray:
        scores = self.ood.fit_score(pred_probs=pred_probs, labels=self.datalab._labels, **kwargs)
        return scores

    def _score_with_features(self, features: List[str], **kwargs) -> np.ndarray:
        self._embeddings = datalab_data._extract_embeddings(
            dataset=self.datalab.data, columns=features, **kwargs
        )

        scores = self.ood.fit_score(features=self._embeddings)
        return scores

    @property
    def verbosity_levels(self) -> Dict[int, Any]:
        return {
            0: {},
            1: {"info": ["num_outlier_issues"], "issue": ["nearest_neighbor"]},
            2: {"issue": ["distance_to_nearest_neighbor"]},
        }

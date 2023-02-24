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

from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from cleanlab.experimental.datalab.issue_manager import IssueManager
from cleanlab.outlier import OutOfDistribution

if TYPE_CHECKING:  # pragma: no cover
    from sklearn.neighbors import NearestNeighbors

    from cleanlab.experimental.datalab.datalab import Datalab


class OutOfDistributionIssueManager(IssueManager):
    """Manages issues related to out-of-distribution examples."""

    description: ClassVar[
        str
    ] = """An outlier issue refers to examples that are very different
        from the rest of the dataset (i.e. potentially out-of-distribution).

        Training/evaluating ML models with such examples may have unexpected consequences.

        Examples may be considered as outliers if they:
            - Are drawn from different distributions than the rest of the dataset.
            - Are rare or anomalous events with extreme values.
            - Have measurement- or data-collection errors.
            - etc.
        """
    issue_name: ClassVar[str] = "outlier"
    verbosity_levels = {
        0: {},
        1: {},
        2: {"info": ["average_ood_score"], "issue": ["nearest_neighbor"]},
        3: {"issue": ["distance_to_nearest_neighbor"]},
    }

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
        features: Optional[npt.NDArray] = None,
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

        self.summary = self.make_summary(score=scores.mean())

        self.info = self.collect_info()

    def collect_info(self) -> dict:
        issues_dict = {
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
                    "metric": knn.metric,  # type: ignore[union-attr]
                    "k": knn.n_neighbors,  # type: ignore[union-attr]
                    "nearest_neighbor": nn_ids.tolist(),
                    "distance_to_nearest_neighbor": dists.tolist(),
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
            **ood_params_dict,  # type: ignore[arg-type]
            **knn_dict,
        }
        return info_dict

    def _score_with_pred_probs(self, pred_probs: np.ndarray, **kwargs) -> np.ndarray:
        scores = self.ood.fit_score(pred_probs=pred_probs, labels=self.datalab._labels, **kwargs)
        return scores

    def _score_with_features(self, features: npt.NDArray, **kwargs) -> npt.NDArray:
        self._embeddings = features
        scores = self.ood.fit_score(features=self._embeddings)
        return scores

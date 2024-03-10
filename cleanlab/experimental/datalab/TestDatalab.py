# Copyright (C) 2017-2024  Cleanlab Inc.
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
"""
The experimental feature that construct a datalab instance with the statistic information from a trained datalab.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

from cleanlab.datalab.datalab import Datalab
from cleanlab.datalab.internal.issue_finder import IssueFinder

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from datasets.arrow_dataset import Dataset
    from scipy.sparse import csr_matrix

    DatasetLike = Union[Dataset, pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], str]


class TestDatalab(Datalab):
    def __init__(
        self,
        trained_datalab: Datalab,
        data: "DatasetLike",
        task: str = "classification",
        label_name: Optional[str] = None,
        image_key: Optional[str] = None,
        verbosity: int = 1,
    ):
        super().__init__(data, task, label_name, image_key, verbosity)
        self.trained_datalab = trained_datalab
        self.check_label_map()
        self._label_map = trained_datalab._label_map
        self.data.label_map = self._label_map
        self.data_issues._data.labels.label_map = self._label_map
        for k in trained_datalab.get_info().keys():
            self.data_issues._update_issue_info(k, trained_datalab.get_info(k))

    def check_label_map(self):
        for k in self._label_map.keys():
            if k not in self.trained_datalab._label_map:
                raise ValueError(f"The label {k} is not in the trained datalab's label_map.")

    def find_issues(
        self,
        *,
        pred_probs: Optional[np.ndarray] = None,
        features: Optional[npt.NDArray] = None,
        knn_graph: Optional[csr_matrix] = None,
        issue_types: Optional[Dict[str, Any]] = None,
    ) -> None:
        issue_finder = TestIssueFinder(datalab=self, task=self.task, verbosity=self.verbosity)
        issue_finder.find_issues(
            pred_probs=pred_probs,
            features=features,
            knn_graph=knn_graph,
            issue_types=issue_types,
        )

        if self.verbosity:
            print(
                f"\nAudit complete. {self.data_issues.issue_summary['num_issues'].sum()} issues found in the dataset."
            )


class TestIssueFinder(IssueFinder):
    def _resolve_trained_statistics_args(self, issue_types: Dict[str, Any]):
        """For label error only now"""
        # get 'confident_joint' if it not passed by user. Note: the 'noise_matrix' and 'inversed_noise_matrix' will always be recomputed by the `CleanLearning` instance.
        supported_issue_types = ["label"]
        issue_keys = {"label": ["confident_joint"]}
        for issue in supported_issue_types:
            issue_types[issue].update(
                {
                    "clean_learning_kwargs": {
                        "find_label_issues_kwargs": {
                            k: v
                            for k, v in self.datalab.get_info(issue).items()
                            if (k in issue_keys[issue] and issue_types[issue].get(k) is None)
                        }
                    }
                }
            )
        return issue_types

    def get_available_issue_types(self, **kwargs):
        issue_types_copy = super().get_available_issue_types(**kwargs)
        label_use_trained_statistics = not (
            "label" in issue_types_copy and not self.datalab.has_labels
        )
        if label_use_trained_statistics:
            issue_types_copy = self._resolve_trained_statistics_args(issue_types_copy)
        return issue_types_copy

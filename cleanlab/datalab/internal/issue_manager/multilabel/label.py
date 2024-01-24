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

from typing import TYPE_CHECKING, Any, ClassVar, Dict, List

import pandas as pd

from cleanlab.multilabel_classification.filter import find_label_issues
from cleanlab.multilabel_classification.rank import get_label_quality_scores
from cleanlab.datalab.internal.issue_manager import IssueManager
from cleanlab.internal.multilabel_utils import onehot2int

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    import numpy.typing as npt
    from cleanlab.datalab.datalab import Datalab


class MultilabelIssueManager(IssueManager):
    """Manages label issues in Datalab for multilabel tasks.

    Parameters
    ----------
    datalab :
        A Datalab instance.
    """

    description: ClassVar[
        str
    ] = """Examples whose given label(s) are estimated to be potentially incorrect
    (e.g. due to annotation error) are flagged as having label issues.
    """

    _PREDICTED_LABEL_THRESH = 0.5
    """Internal variable specifying threshold for predicted label."""

    issue_name: ClassVar[str] = "label"
    verbosity_levels = {
        0: [],
        1: [],
        2: [],
        3: [],
    }

    def __init__(
        self,
        datalab: Datalab,
        **_,
    ):
        super().__init__(datalab)

    @staticmethod
    def _process_find_label_issues_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Searches for keyword arguments that are meant for the
        multilabel_classification.filter.find_label_issues method call.

        Examples
        --------
        >>> from cleanlab.datalab.internal.issue_manager.multilabel.label import MultilabelIssueManager
        >>> MultilabelIssueManager._process_find_label_issues_kwargs(frac_noise=0.9)
        {'frac_noise': 0.1}
        """
        accepted_kwargs = (
            [
                "filter_by",
                "frac_noise",
                "num_to_remove_per_class",
                "min_examples_per_class",
                "confident_joint",
                "n_jobs",
                "verbose",
                "low_memory",
            ],
        )
        return {k: v for k, v in kwargs.items() if k in accepted_kwargs and v is not None}

    @staticmethod
    def _process_get_label_quality_scores_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Searches for keyword arguments that are meant for the
        multilabel_classification.rank.get_label_quality_scores method call.

        Examples
        --------
        >>> from cleanlab.datalab.internal.issue_manager.multilabel.label import MultilabelIssueManager
        >>> MultilabelIssueManager._process_get_label_quality_scores_kwargs(method="self_confidence")
        {'method': "self_confidence"}
        """
        accepted_kwargs = (["method", "adjust_pred_probs", "aggregator_kwargs"],)
        return {k: v for k, v in kwargs.items() if k in accepted_kwargs and v is not None}

    def find_issues(
        self,
        pred_probs: npt.NDArray,
        **kwargs,
    ) -> None:
        """Find label issues in a multilabel dataset.

        Parameters
        ----------
        pred_probs :
            The predicted probabilities for each example.
        """
        predicted_labels = onehot2int(pred_probs > self._PREDICTED_LABEL_THRESH)

        # Find examples with label issues
        assert isinstance(self.datalab.labels, List)  # Type Narrowing
        is_issue_column = find_label_issues(
            labels=self.datalab.labels,
            pred_probs=pred_probs,
            **self._process_find_label_issues_kwargs(kwargs),
        )
        scores = get_label_quality_scores(
            labels=self.datalab.labels,
            pred_probs=pred_probs,
            **self._process_get_label_quality_scores_kwargs(kwargs),
        )

        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": is_issue_column,
                self.issue_score_key: scores,
            },
        )
        # Get a summarized dataframe of the label issues
        self.summary = self.make_summary(score=scores.mean())

        # Collect info about the label issues
        self.info = self.collect_info(self.datalab.labels, predicted_labels)

    def collect_info(
        self, given_labels: List[List[int]], predicted_labels: List[List[int]]
    ) -> dict:
        issues_info = {
            "given_label": given_labels,
            "predicted_label": predicted_labels,
        }
        info_dict = {**issues_info}

        return info_dict

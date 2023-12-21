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
import pandas as pd

from cleanlab.multilabel_classification.filter import find_label_issues
from cleanlab.multilabel_classification.rank import get_label_quality_scores
from cleanlab.datalab.internal.issue_manager import IssueManager
from cleanlab.internal.validation import assert_valid_inputs

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    import numpy.typing as npt
    from cleanlab.datalab.datalab import Datalab


class MultilabelIssueManager(IssueManager):
    """Manages label issues in a Datalab for multilabel tasks.

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
        TODO: Fix docstring

        Examples
        --------
        >>> from cleanlab.datalab.internal.issue_manager.multilabel.label import MultilabelIssueManager
        >>> MultilabelIssueManager._process_find_label_issues_kwargs(thresholds=[0.1, 0.9])
        {'thresholds': [0.1, 0.9]}
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
        TODO: Fix docstring

        Examples
        --------
        >>> from cleanlab.datalab.internal.issue_manager.multilabel.label import MultilabelIssueManager
        >>> MultilabelIssueManager._process_get_label_quality_scores_kwargs(thresholds=[0.1, 0.9])
        {'thresholds': [0.1, 0.9]}
        """
        accepted_kwargs = (["method", "adjust_pred_probs", "aggregator_kwargs"],)
        return {k: v for k, v in kwargs.items() if k in accepted_kwargs and v is not None}

    def find_issues(
        self,
        pred_probs: Optional[npt.NDArray] = None,
        **kwargs,
    ) -> None:
        """Find label issues in the datalab.

        Parameters
        ----------
        pred_probs :
            The predicted probabilities for each example.

        features :
            The features for each example.
        """
        if pred_probs is None:
            raise ValueError(
                "Both pred_probs and features must be provided to find label issues in multilabel data."
            )

        # Find examples with label issues
        is_issue_column = find_label_issues(
            labels=self.datalab.labels,
            pred_probs=pred_probs,
            **self._process_find_label_issues_kwargs(kwargs),
        )
        scores = get_label_quality_scores(
            labels=self.datalab.labels,
            pred_probs=pred_probs,
            **self._process_find_label_issues_kwargs(kwargs),
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
        self.info = self.collect_info()

    def collect_info(self) -> dict:
        return {}


# Todo: Validate Input

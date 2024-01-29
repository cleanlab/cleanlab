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

from typing import Dict, Optional, Type

from cleanlab.datalab.internal.adapter.imagelab import (
    ImagelabDataIssuesAdapter,
    ImagelabIssueFinderAdapter,
    ImagelabReporterAdapter,
)
from cleanlab.datalab.internal.data import Data
from cleanlab.datalab.internal.data_issues import (
    _InfoStrategy,
    DataIssues,
    _ClassificationInfoStrategy,
    _RegressionInfoStrategy,
    _MultilabelInfoStrategy,
)
from cleanlab.datalab.internal.issue_finder import IssueFinder
from cleanlab.datalab.internal.report import Reporter
from cleanlab.datalab.internal.task import Task


def issue_finder_factory(imagelab):
    if imagelab:
        return ImagelabIssueFinderAdapter
    else:
        return IssueFinder


def report_factory(imagelab):
    if imagelab:
        return ImagelabReporterAdapter
    else:
        return Reporter


class _DataIssuesBuilder:
    """A helper class for constructing DataIssues instances.
    It uses the builder pattern to allow users to specify the desired
    configuration of the DataIssues instance.
    It uses the `set_X` naming convention for methods that set the
    desired configuration, before calling the `build` method to
    construct the DataIssues instance.
    """

    def __init__(self, data: Data):
        self.data = data
        self.imagelab = None
        self.task: Optional[Task] = None

    def set_imagelab(self, imagelab):
        self.imagelab = imagelab
        return self

    def set_task(self, task: Task):
        """Set the task that the data is intended for.

        Parameters
        ----------
        task : Task
            Specific machine learning task that the datset is intended for.
            See details about supported tasks in :py:class:`Task <cleanlab.datalab.internal.task.Task>`.
        """
        self.task = task
        return self

    def build(self) -> DataIssues:
        data_issues_class = self._data_issues_factory()
        strategy = self._select_info_strategy()
        return data_issues_class(self.data, strategy)

    def _data_issues_factory(self) -> Type[DataIssues]:
        """Factory method that selects the appropriate class for
        constructing the DataIssues instance.
        """
        if self.imagelab:
            return ImagelabDataIssuesAdapter
        else:
            return DataIssues

    def _select_info_strategy(self) -> Type[_InfoStrategy]:
        """The DataIssues class takes in a strategy class
        for processing info dictionaries. This method selects
        the appropriate strategy class based on the task during
        the `build` method-call.
        """
        _default_return = _ClassificationInfoStrategy
        strategy_lookup: Dict[Task, Type[_InfoStrategy]] = {
            Task.REGRESSION: _RegressionInfoStrategy,
            Task.MULTILABEL: _MultilabelInfoStrategy,
        }
        if self.task is None:
            return _default_return
        return strategy_lookup.get(self.task, _default_return)

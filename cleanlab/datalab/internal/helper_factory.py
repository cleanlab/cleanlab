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

from typing import Optional, Type

from cleanlab.datalab.internal.adapter.imagelab import (
    ImagelabDataIssuesAdapter,
    ImagelabIssueFinderAdapter,
    ImagelabReporterAdapter,
)
from cleanlab.datalab.internal.data import Data
from cleanlab.datalab.internal.data_issues import (
    DataIssues,
    _ClassificationInfoStrategy,
    _RegressionInfoStrategy,
)
from cleanlab.datalab.internal.issue_finder import IssueFinder
from cleanlab.datalab.internal.report import Reporter


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
    def __init__(self, data: Data):
        self.data = data
        self.imagelab = None
        self.task: Optional[str] = None

    def set_imagelab(self, imagelab):
        self.imagelab = imagelab
        return self

    def set_task(self, task):
        self.task = task
        return self

    def build(self) -> DataIssues:
        data_issues_class = self._data_issues_factory()
        strategy = self._select_info_strategy()
        return data_issues_class(self.data, strategy)

    def _data_issues_factory(self) -> Type[DataIssues]:
        if self.imagelab:
            return ImagelabDataIssuesAdapter
        else:
            return DataIssues

    def _select_info_strategy(self):
        if self.task == "regression":
            return _RegressionInfoStrategy
        else:
            return _ClassificationInfoStrategy

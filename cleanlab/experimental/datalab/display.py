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
"""
Module for class that handles the string representation of DataLab objects.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cleanlab.experimental.datalab.datalab import Datalab  # pragma: no cover


class _Displayer:
    def __init__(self, datalab: "Datalab") -> None:
        self.datalab = datalab

    def __repr__(self) -> str:
        """What is displayed in console if user executes: >>> datalab"""
        checks_run = self.datalab.issues is None
        display_str = f"checks_run={checks_run},"
        num_examples = self.datalab.get_info("data", "num_examples")
        if num_examples is not None:
            display_str += f"num_examples={num_examples},"
        num_classes = self.datalab.get_info("data", "num_classes")
        if num_classes is not None:
            display_str += f"num_classes={num_classes},"
        if display_str[-1] == ",":  # delete trailing comma
            display_str = display_str[:-1]

        # Useful info could be: num_examples, task, issues_identified
        # (numeric or None if issue-finding not run yet).
        return f"Datalab({display_str})"

    def __str__(self) -> str:
        """What is displayed if user executes: print(datalab)"""
        return "Datalab"  # TODO

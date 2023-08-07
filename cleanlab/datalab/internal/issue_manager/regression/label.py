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

# from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import OneHotEncoder

# import numpy as np

# from cleanlab.regression.learn import CleanLearning
from cleanlab.datalab.internal.issue_manager import IssueManager
# from cleanlab.internal.validation import assert_valid_inputs

# if TYPE_CHECKING:  # pragma: no cover
#     import pandas as pd
#     import numpy.typing as npt
#     from cleanlab.datalab.datalab import Datalab


class RegressionLabelIssueManager(IssueManager):
    """Manages label issues in a Datalab.

    Parameters
    ----------
    datalab :
        A Datalab instance.

    ... :
        ...

    clean_learning_kwargs :
        Keyword arguments to pass to the :py:meth:`CleanLearning <cleanlab.classification.CleanLearning>` constructor.

    health_summary_parameters :
        Keyword arguments to pass to the :py:meth:`health_summary <cleanlab.dataset.health_summary>` function.
    """
    pass


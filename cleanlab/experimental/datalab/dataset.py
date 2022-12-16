# Copyright (C) 2017-2022  Cleanlab Inc.
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
Implements cleanlab's Dataset class for storing ML datasets and
working with them in a
:py:func:`Datalab <cleanlab.experimental.datalab.datalab.Datalab>` object.
"""


class Dataset:
    """
    Stores a dataset and iassociated metadata.
    """

    def __init__(
        self,
    ) -> None:
        ...

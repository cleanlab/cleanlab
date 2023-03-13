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


FLOATING_POINT_COMPARISON = 1e-6  # floating point comparison for fuzzy equals
CLIPPING_LOWER_BOUND = 1e-6  # lower-bound clipping threshold for expected behavior
CONFIDENT_THRESHOLDS_LOWER_BOUND = (
    2 * FLOATING_POINT_COMPARISON
)  # lower bound imposed to clip confident thresholds from below, has to be larger than floating point comparison
TINY_VALUE = 1e-100  # very tiny value for clipping

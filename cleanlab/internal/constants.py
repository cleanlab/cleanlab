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


# Object Detection Constants
EUC_FACTOR = 0.1  # factor to control magnitude of euclidian distance
MAX_ALLOWED_BOX_PRUNE = 0.97  # This is max allowed percent of prune for boxes below threshold before a warning is thrown.

ALPHA = 0.91  # param for objectlab, weight between IoU and distance when considering similarity matrix. High alpha means considering IoU more strongly over distance
LOW_PROBABILITY_THRESHOLD = 0.001  # param for objectlab, lowest prediction threshold allowed when considering predicted boxes to identify badly located label boxes
HIGH_PROBABILITY_THRESHOLD = 0.5  # param for objectlab, high probability threshold for considering predicted boxes to identify overlooked and swapped label boxes
TEMPERATURE = 0.1  # param for objectlab, temperature of the softmin function where a lower score suggests softmin acts closer to min

OVERLOOKED_THRESHOLD = (
    0.3  # threshold to determine max score for a box to be considered an overlooked issue
)
BADLOC_THRESHOLD = (
    0.3  # threshold to determine max score for a box to be considered a bad location issue
)
SWAP_THRESHOLD = 0.3  # threshold to determine max score for a box to be considered a swap issue

CUSTOM_SCORE_WEIGHT_OVERLOOKED = 0.6  # weight to determine how much to value overlooked scores over other subtypes when deciding issues
CUSTOM_SCORE_WEIGHT_BADLOC = 0.2  # weight to determine how much to value badloc scores over other subtypes when deciding issues
CUSTOM_SCORE_WEIGHT_SWAP = 0.2  # weight to determine how much to value swap scores over other subtypes when deciding issues

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
EUC_FACTOR = 0.1  # Factor to control magnitude of euclidian distance. Increasing the factor makes the distances between two objects go to zero more rapidly.
MAX_ALLOWED_BOX_PRUNE = 0.97  # This is max allowed percent of boxes that are pruned before a warning is thrown given a specific threshold. Pruning too many boxes negatively affects performance.

ALPHA = 0.9  # Param for objectlab, weight between IoU and distance when considering similarity matrix. High alpha means considering IoU more strongly over distance
LOW_PROBABILITY_THRESHOLD = 0.5  # Param for get_label_quality_score, lowest predicted class probability threshold allowed when considering predicted boxes to identify badly located label boxes.
HIGH_PROBABILITY_THRESHOLD = 0.95  # Param for objectlab, high probability threshold for considering predicted boxes to identify overlooked and swapped label boxes
TEMPERATURE = 0.1  # Param for objectlab, temperature of the softmin function used to pool the per-box quality scores for an error subtype across all boxes into a single subtype score for the image. With a lower temperature, softmin pooling acts more like minimum pooling, alternatively it acts more like mean pooling with high temperature.

OVERLOOKED_THRESHOLD = 0.3  # Param for find_label_issues. Per-box label quality score threshold to determine max score for a box to be considered an overlooked issue
BADLOC_THRESHOLD = 0.3  # Param for find_label_issues. Per-box label quality score threshold to determine max score for a box to be considered a bad location issue
SWAP_THRESHOLD = 0.3  # Param for find_label_issues. Per-box label quality score threshold to determine max score for a box to be considered a swap issue

CUSTOM_SCORE_WEIGHT_OVERLOOKED = (
    1 / 3
)  # Param for get_label_quality_score, weight to determine how much to value overlooked scores over other subtypes when deciding the overall label quality score for an image.
CUSTOM_SCORE_WEIGHT_BADLOC = (
    1 / 3
)  # Param for get_label_quality_score, weight to determine how much to value badloc scores over other subtypes when deciding issues
CUSTOM_SCORE_WEIGHT_SWAP = (
    1 / 3
)  # Param for get_label_quality_score, weight to determine how much to value swap scores over other subtypes when deciding issues

MAX_CLASS_TO_SHOW = 10  # Nmber of classes to show in legend during the visualize method. Classes over max_class_to_show are cut off.

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

"""Methods to find label issues in an object detection dataset (object detection data), where each annotated bounding box in an image receives its own class label."""

import warnings
from typing import Optional, Union, Tuple, List, Any, Dict
import numpy as np

from cleanlab.object_detection.rank import (
    _get_valid_inputs_for_compute_scores,
    _compute_overlooked_box_scores,
    _compute_badloc_box_scores,
    _compute_swap_box_scores,
)


def find_label_issues(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
) -> np.ndarray:
    """
    Identifies potentially mislabeled examples in an object detection dataset.
    An example is flagged as with a label issue if *any* of the boxes appear to be incorrectly annotated for this example.
    Incorrectlt labeled ex

    Parameters
    ----------
    labels:
        A list of ``N`` dictionaries such that ``labels[i]`` contains the given labels for the `i`-th image in the format
       ``{'bboxes': np.ndarray((M,4)), 'labels': np.ndarray((M,)), 'image_name': str}`` where ``L`` is the number of annotated bounding boxes
       for the `i`-th image and ``bboxes[j]`` is in the format ``[x,y,x,y]`` with given label ``labels[j]``. (``image_name`` is optional here)

    predictions:
        A list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model predictions for the `i`-th image
        in the format ``np.ndarray((K,))`` and ``predictions[i][k]`` is of shape ``np.ndarray(M,5)``
        where ``M`` is the number of predicted bounding boxes for class ``k`` and the five columns correspond to ``[x,y,x,y,pred_prob]`` returned
        by the model.

    Returns
    -------
    label_issues : np.ndarray
      Returns a list of **indices** of examples identified with label issues (i.e. those indices where the mask would be ``True``).
    """
    alpha = 0.91  # hyperparameter
    high_probability_threshold = 0.7  # hyperparameter
    low_probability_threshold = 0.1  # hyperparameter

    thr_overlooked = 0.4  # hyperparameter
    thr_badloc = 0.4  # hyperparameter
    thr_swap = 0.4  # hyperparameter

    auxiliary_input_dict = _get_valid_inputs_for_compute_scores(alpha, labels, predictions)

    overlooked_scores_per_box = _compute_overlooked_box_scores(
        alpha=alpha, high_probability_threshold=high_probability_threshold, **auxiliary_input_dict
    )
    overlooked_issues_per_box = _find_label_issues_per_box(
        overlooked_scores_per_box, thr_overlooked
    )
    overlooked_issues_per_image = _pool_box_scores_per_image(overlooked_issues_per_box)

    badloc_scores_per_box = _compute_badloc_box_scores(
        alpha=alpha, low_probability_threshold=low_probability_threshold, **auxiliary_input_dict
    )
    badloc_issues_per_box = _find_label_issues_per_box(badloc_scores_per_box, thr_badloc)
    badloc_issues_per_image = _pool_box_scores_per_image(badloc_issues_per_box)

    swap_scores_per_box = _compute_swap_box_scores(
        alpha=alpha, high_probability_threshold=high_probability_threshold, **auxiliary_input_dict
    )
    swap_issues_per_box = _find_label_issues_per_box(swap_scores_per_box, thr_swap)
    swap_issues_per_image = _pool_box_scores_per_image(swap_issues_per_box)

    issues_per_image = overlooked_issues_per_image + badloc_issues_per_image + swap_issues_per_image
    return issues_per_image


def _find_label_issues_per_box(scores_per_box, threshold):
    is_issue_per_box = []
    for idx, score_per_box in enumerate(scores_per_box):
        score_per_box[np.isnan(score_per_box)] = 1.0
        issue_per_box = score_per_box <= threshold
        is_issue_per_box.append(issue_per_box)
    return is_issue_per_box


def _pool_box_scores_per_image(is_issue_per_box):
    is_issue = np.zeros(
        shape=[
            len(
                is_issue_per_box,
            )
        ]
    )
    for idx, issue_per_box in enumerate(is_issue_per_box):
        if np.sum(issue_per_box) > 0:
            is_issue[idx] = 1
    return is_issue

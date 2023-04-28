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

from typing import List, Any, Dict
import numpy as np

from cleanlab.internal.constants import (
    ALPHA,
    LOW_PROBABILITY_THRESHOLD,
    HIGH_PROBABILITY_THRESHOLD,
    OVERLOOKED_THRESHOLD,
    BADLOC_THRESHOLD,
    SWAP_THRESHOLD,
)

from cleanlab.object_detection.rank import (
    _get_valid_inputs_for_compute_scores,
    _compute_overlooked_box_scores,
    _compute_badloc_box_scores,
    _compute_swap_box_scores,
    _assert_valid_inputs,
)


def find_label_issues(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
) -> np.ndarray:
    """
    Identifies potentially mislabeled examples in an object detection dataset.
    An image is flagged with a label issue if *any* of its bounding boxes appear incorrectly annotated -- this includes images for which a bounding box: should have been annotated but is missing, has been annotated with the wrong class, or has been annotated in a suboptimal location.

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
        by the model and ``pred_prob`` is the model's confidence in ``predictions[i]``.

    Returns
    -------
    label_issues : np.ndarray
      Returns a list of **indices** of examples identified with label issues (i.e. those indices where the mask would be ``True``).
    """
    scoring_method = "objectlab"

    _assert_valid_inputs(
        labels=labels,
        predictions=predictions,
        method=scoring_method,
    )

    _find_label_issues(labels, predictions, scoring_method=scoring_method)

    auxiliary_inputs = _get_valid_inputs_for_compute_scores(ALPHA, labels, predictions)

    overlooked_scores_per_box = _compute_overlooked_box_scores(
        alpha=ALPHA,
        high_probability_threshold=HIGH_PROBABILITY_THRESHOLD,
        auxiliary_inputs=auxiliary_inputs,
    )
    overlooked_issues_per_box = _find_label_issues_per_box(
        overlooked_scores_per_box, OVERLOOKED_THRESHOLD
    )
    overlooked_issues_per_image = _pool_box_scores_per_image(overlooked_issues_per_box)

    badloc_scores_per_box = _compute_badloc_box_scores(
        alpha=ALPHA,
        low_probability_threshold=LOW_PROBABILITY_THRESHOLD,
        auxiliary_inputs=auxiliary_inputs,
    )
    badloc_issues_per_box = _find_label_issues_per_box(badloc_scores_per_box, BADLOC_THRESHOLD)
    badloc_issues_per_image = _pool_box_scores_per_image(badloc_issues_per_box)

    swap_scores_per_box = _compute_swap_box_scores(
        alpha=ALPHA,
        high_probability_threshold=HIGH_PROBABILITY_THRESHOLD,
        auxiliary_inputs=auxiliary_inputs,
    )
    swap_issues_per_box = _find_label_issues_per_box(swap_scores_per_box, SWAP_THRESHOLD)
    swap_issues_per_image = _pool_box_scores_per_image(swap_issues_per_box)

    issues_per_image = overlooked_issues_per_image + badloc_issues_per_image + swap_issues_per_image
    is_issue = issues_per_image > 0
    return is_issue


def _find_label_issues(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    *,
    scoring_method: str = "objectlab",
):
    """Internal function to find label issues based on passed in method."""

    if scoring_method == "objectlab":
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(ALPHA, labels, predictions)

        overlooked_scores_per_box = _compute_overlooked_box_scores(
            alpha=ALPHA,
            high_probability_threshold=HIGH_PROBABILITY_THRESHOLD,
            auxiliary_inputs=auxiliary_inputs,
        )
        overlooked_issues_per_box = _find_label_issues_per_box(
            overlooked_scores_per_box, OVERLOOKED_THRESHOLD
        )
        overlooked_issues_per_image = _pool_box_scores_per_image(overlooked_issues_per_box)

        badloc_scores_per_box = _compute_badloc_box_scores(
            alpha=ALPHA,
            low_probability_threshold=LOW_PROBABILITY_THRESHOLD,
            auxiliary_inputs=auxiliary_inputs,
        )
        badloc_issues_per_box = _find_label_issues_per_box(badloc_scores_per_box, BADLOC_THRESHOLD)
        badloc_issues_per_image = _pool_box_scores_per_image(badloc_issues_per_box)

        swap_scores_per_box = _compute_swap_box_scores(
            alpha=ALPHA,
            high_probability_threshold=HIGH_PROBABILITY_THRESHOLD,
            auxiliary_inputs=auxiliary_inputs,
        )
        swap_issues_per_box = _find_label_issues_per_box(swap_scores_per_box, SWAP_THRESHOLD)
        swap_issues_per_image = _pool_box_scores_per_image(swap_issues_per_box)

        issues_per_image = (
            overlooked_issues_per_image + badloc_issues_per_image + swap_issues_per_image
        )
        is_issue = issues_per_image > 0
    else:
        is_issue = np.full(
            shape=[
                len(labels),
            ],
            fill_value=-1,
        )
    return is_issue


def _find_label_issues_per_box(
    scores_per_box: List[np.ndarray], threshold: float
) -> List[np.ndarray]:
    """Takes in a list of size ``N`` where each index is an array of scores for each bounding box in the `n-th` image
    and a threshold. Each box below or equal to the threshold will be marked as an issue.

    Returns a list of size ``N`` where each index is a boolean array of length number of boxes per image `n `
    marking if a specific box is an issue - 1 or not - 0."""
    is_issue_per_box = []
    for idx, score_per_box in enumerate(scores_per_box):
        score_per_box[np.isnan(score_per_box)] = 1.0
        issue_per_box = score_per_box <= threshold
        is_issue_per_box.append(issue_per_box)
    return is_issue_per_box


def _pool_box_scores_per_image(is_issue_per_box: List[np.ndarray]) -> np.ndarray:
    """Takes in a list of size ``N`` where each index is a boolean array of length number of boxes per image `n `
    marking if a specific box is an issue - 1 or not - 0.

    Returns a list of size ``N`` where each index marks if the image contains an issue - 1 or not - 0.
    Images are marked as issues if 1 or more bounding boxes in the image is an issue."""
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

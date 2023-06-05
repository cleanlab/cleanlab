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

"""Methods to find label issues in an object detection dataset, where each annotated bounding box in an image receives its own class label."""

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
from cleanlab.internal.object_detection_utils import assert_valid_inputs

from cleanlab.object_detection.rank import (
    _get_valid_inputs_for_compute_scores,
    compute_overlooked_box_scores,
    compute_badloc_box_scores,
    compute_swap_box_scores,
    get_label_quality_scores,
    issues_from_scores,
)


def find_label_issues(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    *,
    return_indices_ranked_by_score: bool = False,
) -> np.ndarray:
    """
    Identifies potentially mislabeled images in an object detection dataset.
    An image is flagged with a label issue if *any* of its bounding boxes appear incorrectly annotated.
    This includes images for which a bounding box: should have been annotated but is missing,
    has been annotated with the wrong class, or has been annotated in a suboptimal location.

    Suppose the dataset has ``N`` images, ``K`` possible class labels.
    If ``return_indices_ranked_by_score`` is ``False``, a boolean mask of length ``N`` is returned,
    indicating whether each image has a label issue (``True``) or not (``False``).
    If ``return_indices_ranked_by_score`` is ``True``, the indices of images flagged with label issues are returned,
    sorted with the most likely-mislabeled images ordered first.

    Parameters
    ----------
    labels:
        Annotated boxes and class labels in the original dataset, which may contain some errors.
        This is a list of ``N`` dictionaries such that ``labels[i]`` contains the given labels for the `i`-th image in the following format:
        ``{'bboxes': np.ndarray((L,4)), 'labels': np.ndarray((L,)), 'image_name': str}`` where ``L`` is the number of annotated bounding boxes
        for the `i`-th image and ``bboxes[l]`` is a bounding box of coordinates in ``[x1,y1,x2,y2]`` format with given class label ``labels[j]``.
        ``image_name`` is an optional part of the labels that can be used to later refer to specific images.

       For more information on proper labels formatting, check out the `MMDetection library <https://mmdetection.readthedocs.io/en/dev-3.x/advanced_guides/customize_dataset.html>`_.

    predictions:
        Predictions output by a trained object detection model.
        For the most accurate results, predictions should be out-of-sample to avoid overfitting, eg. obtained via :ref:`cross-validation <pred_probs_cross_val>`.
        This is a list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model prediction for the `i`-th image.
        For each possible class ``k`` in 0, 1, ..., K-1: ``predictions[i][k]`` is a ``np.ndarray`` of shape ``(M,5)``,
        where ``M`` is the number of predicted bounding boxes for class ``k``. Here the five columns correspond to ``[x1,y1,x2,y2,pred_prob]``,
        where ``[x1,y1,x2,y2]`` are coordinates of the bounding box predicted by the model
        and ``pred_prob`` is the model's confidence in the predicted class label for this bounding box.

        Note: Here, ``[x1,y1]`` corresponds to the coordinates of the bottom-left corner of the bounding box, while ``[x2,y2]`` corresponds to the coordinates of the top-right corner of the bounding box. The last column, pred_prob, represents the predicted probability that the bounding box contains an object of the class k.

        For more information see the `MMDetection package <https://github.com/open-mmlab/mmdetection>`_ for an example object detection library that outputs predictions in the correct format.

    return_indices_ranked_by_score:
        Determines what is returned by this method (see description of return value for details).

    Returns
    -------
    label_issues : np.ndarray
        Specifies which images are identified to have a label issue.
        If ``return_indices_ranked_by_score = False``, this function returns a boolean mask of length ``N`` (``True`` entries indicate which images have label issue).
        If ``return_indices_ranked_by_score = True``, this function returns a (shorter) array of indices of images with label issues, sorted by how likely the image is mislabeled.

        More precisely, indices are sorted by image label quality score calculated via :py:func:`object_detection.rank.get_label_quality_scores <cleanlab.object_detection.rank.get_label_quality_scores>`.
    """
    scoring_method = "objectlab"

    assert_valid_inputs(
        labels=labels,
        predictions=predictions,
        method=scoring_method,
    )

    is_issue = _find_label_issues(
        labels,
        predictions,
        scoring_method=scoring_method,
        return_indices_ranked_by_score=return_indices_ranked_by_score,
    )

    return is_issue


def _find_label_issues(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    *,
    return_indices_ranked_by_score: bool = True,
    scoring_method: str = "objectlab",
):
    """Internal function to find label issues based on passed in method."""

    if scoring_method == "objectlab":
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(ALPHA, labels, predictions)

        overlooked_scores_per_box = compute_overlooked_box_scores(
            alpha=ALPHA,
            high_probability_threshold=HIGH_PROBABILITY_THRESHOLD,
            auxiliary_inputs=auxiliary_inputs,
        )
        overlooked_issues_per_box = _find_label_issues_per_box(
            overlooked_scores_per_box, OVERLOOKED_THRESHOLD
        )
        overlooked_issues_per_image = _pool_box_scores_per_image(overlooked_issues_per_box)

        badloc_scores_per_box = compute_badloc_box_scores(
            alpha=ALPHA,
            low_probability_threshold=LOW_PROBABILITY_THRESHOLD,
            auxiliary_inputs=auxiliary_inputs,
        )
        badloc_issues_per_box = _find_label_issues_per_box(badloc_scores_per_box, BADLOC_THRESHOLD)
        badloc_issues_per_image = _pool_box_scores_per_image(badloc_issues_per_box)

        swap_scores_per_box = compute_swap_box_scores(
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

    if return_indices_ranked_by_score:
        scores = get_label_quality_scores(labels, predictions)
        sorted_scores_idx = issues_from_scores(scores, threshold=1.0)
        is_issue_idx = np.where(is_issue == True)[0]
        sorted_issue_mask = np.in1d(sorted_scores_idx, is_issue_idx, assume_unique=True)
        issue_idx = sorted_scores_idx[sorted_issue_mask]
        return issue_idx
    else:
        return is_issue


def _find_label_issues_per_box(
    scores_per_box: List[np.ndarray], threshold: float
) -> List[np.ndarray]:
    """Takes in a list of size ``N`` where each index is an array of scores for each bounding box in the `n-th` example
    and a threshold. Each box below or equal to the threshold will be marked as an issue.

    Returns a list of size ``N`` where each index is a boolean array of length number of boxes per example `n`
    marking if a specific box is an issue - 1 or not - 0."""
    is_issue_per_box = []
    for idx, score_per_box in enumerate(scores_per_box):
        if len(score_per_box) == 0:  # if no for specific image, then image not an issue
            is_issue_per_box.append(np.array([False]))
        else:
            score_per_box[np.isnan(score_per_box)] = 1.0
            score_per_box = score_per_box
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

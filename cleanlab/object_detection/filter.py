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

from collections import defaultdict
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from cleanlab.internal.constants import (
    ALPHA,
    HIGH_PROBABILITY_THRESHOLD,
    LOW_PROBABILITY_THRESHOLD,
    OVERLOOKED_THRESHOLD_FACTOR,
    BADLOC_THRESHOLD_FACTOR,
    SWAP_THRESHOLD_FACTOR,
    AP_SCALE_FACTOR,
)
from cleanlab.internal.object_detection_utils import assert_valid_inputs
from cleanlab.object_detection.rank import (
    _get_valid_inputs_for_compute_scores,
    _separate_label,
    _separate_prediction,
    compute_badloc_box_scores,
    compute_overlooked_box_scores,
    compute_swap_box_scores,
    get_label_quality_scores,
    issues_from_scores,
    _get_overlap_matrix,
)


def find_label_issues(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    *,
    return_indices_ranked_by_score: Optional[bool] = False,
    overlapping_label_check: Optional[bool] = True,
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
        for the `i`-th image and ``bboxes[l]`` is a bounding box of coordinates in ``[x1,y1,x2,y2]`` format and with given class label ``labels[j]``.
        ``image_name`` is an optional part of the labels that can be used to later refer to specific images.

        Note: Here, ``(x1,y1)`` corresponds to the top-left and ``(x2,y2)`` corresponds to the bottom-right corner of the bounding box with respect to the image matrix [e.g. `XYXY in Keras <https://keras.io/api/keras_cv/bounding_box/formats/>`, `Detectron 2 <https://detectron2.readthedocs.io/en/latest/modules/utils.html#detectron2.utils.visualizer.Visualizer.draw_box>`].

        For more information on proper labels formatting, check out the `MMDetection library <https://mmdetection.readthedocs.io/en/dev-3.x/advanced_guides/customize_dataset.html>`_.

    predictions:
        Predictions output by a trained object detection model.
        For the most accurate results, predictions should be out-of-sample to avoid overfitting, eg. obtained via :ref:`cross-validation <pred_probs_cross_val>`.
        This is a list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model prediction for the `i`-th image.
        For each possible class ``k`` in 0, 1, ..., K-1: ``predictions[i][k]`` is a ``np.ndarray`` of shape ``(M,5)``,
        where ``M`` is the number of predicted bounding boxes for class ``k``. Here the five columns correspond to ``[x1,y1,x2,y2,pred_prob]``,
        where ``[x1,y1,x2,y2]`` are coordinates of the bounding box predicted by the model and ``pred_prob`` is the model's confidence in the predicted class label for this bounding box.

        Note: Here, ``(x1,y1)`` corresponds to the top-left and ``(x2,y2)`` corresponds to the bottom-right corner of the bounding box with respect to the image matrix [e.g. `XYXY in Keras <https://keras.io/api/keras_cv/bounding_box/formats/>`, `Detectron 2 <https://detectron2.readthedocs.io/en/latest/modules/utils.html#detectron2.utils.visualizer.Visualizer.draw_box>`]. The last column, pred_prob, represents the predicted probability that the bounding box contains an object of the class k.

        For more information see the `MMDetection package <https://github.com/open-mmlab/mmdetection>`_ for an example object detection library that outputs predictions in the correct format.

    return_indices_ranked_by_score:
        Determines what is returned by this method (see description of return value for details).

    overlapping_label_check : bool, default = True
       If True, boxes annotated with more than one class label have their swap score penalized.  Set this to False if you are not concerned when two very similar boxes exist with different class labels in the given annotations.


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
        overlapping_label_check=overlapping_label_check,
    )

    return is_issue


def _find_label_issues(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    *,
    scoring_method: Optional[str] = "objectlab",
    return_indices_ranked_by_score: Optional[bool] = True,
    overlapping_label_check: Optional[bool] = True,
):
    """Internal function to find label issues based on passed in method."""

    if scoring_method == "objectlab":
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(ALPHA, labels, predictions)

        per_class_scores = _get_per_class_ap(labels, predictions)
        lab_list = [_separate_label(label)[1] for label in labels]
        pred_list = [_separate_prediction(pred)[1] for pred in predictions]
        pred_thresholds_list = _process_class_list(pred_list, per_class_scores)
        lab_thresholds_list = _process_class_list(lab_list, per_class_scores)
        overlooked_scores_per_box = compute_overlooked_box_scores(
            alpha=ALPHA,
            high_probability_threshold=HIGH_PROBABILITY_THRESHOLD,
            auxiliary_inputs=auxiliary_inputs,
        )
        overlooked_issues_per_box = _find_label_issues_per_box(
            overlooked_scores_per_box, pred_thresholds_list, OVERLOOKED_THRESHOLD_FACTOR
        )
        overlooked_issues_per_image = _pool_box_scores_per_image(overlooked_issues_per_box)

        badloc_scores_per_box = compute_badloc_box_scores(
            alpha=ALPHA,
            low_probability_threshold=LOW_PROBABILITY_THRESHOLD,
            auxiliary_inputs=auxiliary_inputs,
        )
        badloc_issues_per_box = _find_label_issues_per_box(
            badloc_scores_per_box, lab_thresholds_list, BADLOC_THRESHOLD_FACTOR
        )
        badloc_issues_per_image = _pool_box_scores_per_image(badloc_issues_per_box)

        swap_scores_per_box = compute_swap_box_scores(
            alpha=ALPHA,
            high_probability_threshold=HIGH_PROBABILITY_THRESHOLD,
            overlapping_label_check=overlapping_label_check,
            auxiliary_inputs=auxiliary_inputs,
        )
        swap_issues_per_box = _find_label_issues_per_box(
            swap_scores_per_box, lab_thresholds_list, SWAP_THRESHOLD_FACTOR
        )
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
    scores_per_box: List[np.ndarray], threshold_classes, threshold_factor=1.0
) -> List[np.ndarray]:
    """Takes in a list of size ``N`` where each index is an array of scores for each bounding box in the `n-th` example
    and a threshold. Each box below or equal to the corresponding threshold in threshold_classes will be marked as an issue.

    Returns a list of size ``N`` where each index is a boolean array of length number of boxes per example `n`
    marking if a specific box is an issue - 1 or not - 0."""
    is_issue_per_box = []
    for idx, score_per_box in enumerate(scores_per_box):
        if len(score_per_box) == 0:  # if no for specific image, then image not an issue
            is_issue_per_box.append(np.array([False]))
        else:
            score_per_box[np.isnan(score_per_box)] = 1.0
            score_per_box = score_per_box
            issue_per_box = []
            for i in range(len(score_per_box)):
                issue_per_box.append(
                    score_per_box[i] <= threshold_classes[idx][i] * threshold_factor
                )
            is_issue_per_box.append(np.array(issue_per_box, bool))
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


def _process_class_list(class_list: List[np.ndarray], class_dict: Dict[int, float]) -> List:
    """
    Converts a list of classes represented as numpy arrays using a class-to-float dictionary,
    and returns a list where each class is replaced by its corresponding float value from the dictionary.

    Args:
        class_list (List[np.ndarray]): A list of classes represented as numpy arrays.
        class_dict (Dict[int, float]): A dictionary mapping class indices to their corresponding float values.

    Returns:
        List[float]: A list of float values corresponding to the classes in the input list.
    """
    class_l2 = []
    for i in class_list:
        l3 = [class_dict[j] for j in i]
        class_l2.append(l3)
    return class_l2


def _calculate_ap_per_class(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    *,
    iou_threshold: Optional[float] = 0.5,
    num_procs: int = 1,
) -> List:
    """
    Computes the average precision for each class based on provided labels and predictions.
    It uses an Intersection over Union (IoU) threshold and supports parallel processing with a specified number of processes.

    """
    num_images = len(predictions)
    num_scale = 1
    num_classes = len(predictions[0])
    if num_images > 1:
        num_procs = min(num_procs, num_images)
        pool = Pool(num_procs)
    ap_per_class_list = []
    for class_num in range(num_classes):
        pred_bboxes, lab_bboxes = _filter_by_class(labels, predictions, class_num)
        if num_images > 1:
            tpfp = pool.starmap(
                _calculate_true_positives_false_positives,
                zip(pred_bboxes, lab_bboxes, [iou_threshold for _ in range(num_images)]),
            )
        else:
            tpfp = [
                _calculate_true_positives_false_positives(
                    pred_bboxes[0],
                    lab_bboxes[0],
                    iou_threshold,
                )
            ]
        true_positives, false_positives = tuple(zip(*tpfp))
        num_gts = np.zeros(num_scale, dtype=int)
        for j, bbox in enumerate(lab_bboxes):
            num_gts[0] += bbox.shape[0]
        pred_bboxes = np.vstack(pred_bboxes)
        sort_inds = np.argsort(-pred_bboxes[:, -1])
        true_positives = np.hstack(true_positives)[:, sort_inds]
        false_positives = np.hstack(false_positives)[:, sort_inds]
        true_positives = np.cumsum(true_positives, axis=1)
        false_positives = np.cumsum(false_positives, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = true_positives / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = true_positives / np.maximum((true_positives + false_positives), eps)
        recalls = recalls[0, :]
        precisions = precisions[0, :]
        ap = _calculate_average_precision(recalls, precisions)
        ap_per_class_list.append(ap)
    if num_images > 1:
        pool.close()
    return ap_per_class_list


def _filter_by_class(
    labels: List[Dict[str, Any]], predictions: List[np.ndarray], class_num: int
) -> Tuple[List, List]:
    """
    Filters predictions and labels based on a specific class number.
    """
    pred_bboxes = [prediction[class_num] for prediction in predictions]
    lab_bboxes = []
    for label in labels:
        gt_inds = label["labels"] == class_num
        lab_bboxes.append(label["bboxes"][gt_inds, :])
    return pred_bboxes, lab_bboxes


def _calculate_true_positives_false_positives(
    pred_bboxes: np.ndarray,
    lab_bboxes: np.ndarray,
    iou_threshold: Optional[float] = 0.5,
    return_false_negative: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calculates true positives (TP) and false positives (FP) for object detection tasks.
    It takes predicted bounding boxes, ground truth bounding boxes, and an optional Intersection over Union (IoU) threshold as inputs.
    If return_false_negative is True, it returns an array of False negatives as well.
    """
    num_preds = pred_bboxes.shape[0]
    num_labels = lab_bboxes.shape[0]
    num_scales = 1
    true_positives = np.zeros((num_scales, num_preds), dtype=np.float32)
    false_positives = np.zeros((num_scales, num_preds), dtype=np.float32)

    if lab_bboxes.shape[0] == 0:
        false_positives[...] = 1
        if return_false_negative:
            return true_positives, false_positives, np.array([], dtype=np.float32)
        else:
            return true_positives, false_positives
    ious = _get_overlap_matrix(pred_bboxes, lab_bboxes)
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sorted_indices = np.argsort(-pred_bboxes[:, -1])
    is_covered = np.zeros(num_labels, dtype=bool)
    for index in sorted_indices:
        if ious_max[index] >= iou_threshold:
            matching_label = ious_argmax[index]
            if not is_covered[matching_label]:
                is_covered[matching_label] = True
                true_positives[0, index] = 1
            else:
                false_positives[0, index] = 1
        else:
            false_positives[0, index] = 1
    if return_false_negative:
        false_negatives = np.zeros((num_scales, num_labels), dtype=np.float32)
        for label_index in range(num_labels):
            if not is_covered[label_index]:
                false_negatives[0, label_index] = 1
        return true_positives, false_positives, false_negatives
    return true_positives, false_positives


def _calculate_average_precision(
    recall_values: np.ndarray, precision_values: np.ndarray
) -> np.ndarray:
    """Computes the average precision (AP) for a set of recall and precision values. It takes arrays of recall and precision values as inputs."""
    recall_values = recall_values[np.newaxis, :]
    precision_values = precision_values[np.newaxis, :]
    num_scales = recall_values.shape[0]
    average_precision = np.zeros(num_scales, dtype=np.float32)
    zeros_matrix = np.zeros((num_scales, 1), dtype=recall_values.dtype)
    ones_matrix = np.ones((num_scales, 1), dtype=recall_values.dtype)
    modified_recall = np.hstack((zeros_matrix, recall_values, ones_matrix))
    modified_precision = np.hstack((zeros_matrix, precision_values, zeros_matrix))

    for i in range(modified_precision.shape[1] - 1, 0, -1):
        modified_precision[:, i - 1] = np.maximum(
            modified_precision[:, i - 1], modified_precision[:, i]
        )

    for i in range(num_scales):
        index = np.where(modified_recall[i, 1:] != modified_recall[i, :-1])[0]
        average_precision[i] = np.sum(
            (modified_recall[i, index + 1] - modified_recall[i, index])
            * modified_precision[i, index + 1]
        )

    return average_precision


def _get_per_class_ap(
    labels: List[Dict[str, Any]], predictions: List[np.ndarray]
) -> Dict[int, float]:
    """Computes the Average Precision (AP) for each class in an object detection task.
    It takes a list of label dictionaries and a list of prediction arrays as inputs.
    It calculates AP values for different Intersection over Union (IoU) thresholds, averages them per class, and then scales the AP values.
    """
    iou_thrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
    class_num_to_iou_list = defaultdict(list)
    for threshold in iou_thrs:
        ap_per_class = _calculate_ap_per_class(labels, predictions, iou_threshold=threshold)
        for class_num in range(0, len(ap_per_class)):
            class_num_to_iou_list[class_num].append(ap_per_class[class_num])
    class_num_to_AP = {}
    for class_num in class_num_to_iou_list:
        class_num_to_AP[class_num] = np.mean(class_num_to_iou_list[class_num]) * AP_SCALE_FACTOR
    return class_num_to_AP

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

"""Methods to rank and score images in an object detection dataset (object detection data), based on how likely they
are to contain label errors. """

import warnings

from cleanlab.internal.constants import (
    ALPHA,
    CUSTOM_SCORE_WEIGHT_BADLOC,
    CUSTOM_SCORE_WEIGHT_OVERLOOKED,
    CUSTOM_SCORE_WEIGHT_SWAP,
    EUC_FACTOR,
    HIGH_PROBABILITY_THRESHOLD,
    LOW_PROBABILITY_THRESHOLD,
    MAX_ALLOWED_BOX_PRUNE,
    TINY_VALUE,
    TEMPERATURE,
    LABEL_OVERLAP_THRESHOLD,
)


import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypeVar

import numpy as np
from cleanlab.internal.object_detection_utils import (
    softmin1d,
    assert_valid_aggregation_weights,
    assert_valid_inputs,
)

if TYPE_CHECKING:  # pragma: no cover
    from typing import TypedDict

    AuxiliaryTypesDict = TypedDict(
        "AuxiliaryTypesDict",
        {
            "pred_labels": np.ndarray,
            "pred_label_probs": np.ndarray,
            "pred_bboxes": np.ndarray,
            "lab_labels": np.ndarray,
            "lab_bboxes": np.ndarray,
            "similarity_matrix": np.ndarray,
            "iou_matrix": np.ndarray,
            "min_possible_similarity": float,
        },
    )
else:
    AuxiliaryTypesDict = TypeVar("AuxiliaryTypesDict")


def get_label_quality_scores(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    *,
    aggregation_weights: Optional[Dict[str, float]] = None,
    overlapping_label_check: Optional[bool] = True,
    verbose: bool = True,
) -> np.ndarray:
    """Computes a label quality score for each image of the ``N`` images in the dataset.

    For object detection datasets, the label quality score for an image estimates how likely it has been correctly labeled.
    Lower scores indicate images whose annotation is more likely imperfect.
    Annotators may have mislabeled an image because they:

    - overlooked an object (missing annotated bounding box),
    - chose the wrong class label for an annotated box in the correct location,
    - imperfectly annotated the location/edges of a bounding box.

    Any of these annotation errors should lead to an image with a lower label quality score. This quality score is between 0 and 1.

    - 1 - clean label (given label is likely correct).
    - 0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels:
        A list of ``N`` dictionaries such that ``labels[i]`` contains the given labels for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    predictions:
        A list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model predictions for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    verbose : bool, default = True
      Set to ``False`` to suppress all print statements.

    aggregation_weights:
       Optional dictionary to specify weights for aggregating quality scores for subtype of label issue into an overall label quality score for the image.
       Its keys are: "overlooked", "swap", "badloc", and values should be nonnegative weights that sum to 1.
       Increase one of these weights to prioritize images with bounding boxes that were either:
       missing in the annotations (overlooked object), annotated with the wrong class label (class for the object should be swapped to another class), or annotated in a suboptimal location (badly located).

       swapped examples, bad location examples, and overlooked examples.
       It is important to ensure that the weights are non-negative values and that their sum equals 1.0.

    overlapping_label_check : bool, default = True
        If True, boxes annotated with more than one class label have their swap score penalized. Set this to False if you are not concerned when two very similar boxes exist with different class labels in the given annotations.

    Returns
    ---------
    label_quality_scores:
        Array of shape ``(N, )`` of scores between 0 and 1, one per image in the object detection dataset.
        Lower scores indicate images that are more likely mislabeled.
    """
    method = "objectlab"
    probability_threshold = 0.0

    assert_valid_inputs(
        labels=labels,
        predictions=predictions,
        method=method,
        threshold=probability_threshold,
    )
    aggregation_weights = _get_aggregation_weights(aggregation_weights)

    return _compute_label_quality_scores(
        labels=labels,
        predictions=predictions,
        method=method,
        threshold=probability_threshold,
        aggregation_weights=aggregation_weights,
        overlapping_label_check=overlapping_label_check,
        verbose=verbose,
    )


def issues_from_scores(label_quality_scores: np.ndarray, *, threshold: float = 0.1) -> np.ndarray:
    """Convert label quality scores to a list of indices of images with issues sorted from most to least severe cut off at threshold.

    Returns the list of indices of images with issues sorted from most to least severe cut off at threshold.

    Parameters
    ----------
    label_quality_scores:
        Array of shape ``(N, )`` of scores between 0 and 1, one per image in the object detection dataset.
        Lower scores indicate images are more likely to contain a label issue.

    threshold:
        Label quality scores above the threshold are not considered to be label issues. The corresponding examples' indices are omitted from the returned array.

    Returns
    ---------
    issue_indices:
        Array of issue indices sorted from most to least severe who's label quality scores fall below the threshold if one is provided.
    """

    if threshold > 1.0:
        raise ValueError(
            f"""
            Threshold is a cutoff of label_quality_scores and therefore should be <= 1.
            """
        )

    issue_indices = np.argwhere(label_quality_scores <= threshold).flatten()
    issue_vals = label_quality_scores[issue_indices]
    sorted_idx = issue_vals.argsort()
    return issue_indices[sorted_idx]


def _compute_label_quality_scores(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    *,
    method: Optional[str] = "objectlab",
    aggregation_weights: Optional[Dict[str, float]] = None,
    threshold: Optional[float] = None,
    overlapping_label_check: Optional[bool] = True,
    verbose: bool = True,
) -> np.ndarray:
    """Internal function to prune extra bounding boxes and compute label quality scores based on passed in method."""

    pred_probs_prepruned = False
    min_pred_prob = _get_min_pred_prob(predictions)
    aggregation_weights = _get_aggregation_weights(aggregation_weights)

    if threshold is not None:
        predictions = _prune_by_threshold(
            predictions=predictions, threshold=threshold, verbose=verbose
        )
        if np.abs(min_pred_prob - threshold) < 0.001 and threshold > 0:
            pred_probs_prepruned = True  # the provided threshold is the threshold used for pre_pruning the pred_probs during model prediction.
    else:
        threshold = min_pred_prob  # assume model was not pre_pruned if no threshold was provided

    if method == "objectlab":
        scores = _get_subtype_label_quality_scores(
            labels=labels,
            predictions=predictions,
            alpha=ALPHA,
            low_probability_threshold=LOW_PROBABILITY_THRESHOLD,
            high_probability_threshold=HIGH_PROBABILITY_THRESHOLD,
            temperature=TEMPERATURE,
            aggregation_weights=aggregation_weights,
            overlapping_label_check=overlapping_label_check,
        )
    else:
        raise ValueError(
            "Invalid method: '{}' is not a valid method for computing label quality scores. Please use the 'objectlab' method.".format(
                method
            )
        )
    return scores


def _get_min_pred_prob(
    predictions: List[np.ndarray],
) -> float:
    """Returns min pred_prob out of all predictions."""
    pred_probs = [1.0]  # avoid calling np.min on empty array.
    for prediction in predictions:
        for class_prediction in prediction:
            pred_probs.extend(list(class_prediction[:, -1]))

    min_pred_prob = np.min(pred_probs)
    return min_pred_prob


def _prune_by_threshold(
    predictions: List[np.ndarray], threshold: float, verbose: bool = True
) -> List[np.ndarray]:
    """Removes predicted bounding boxes from predictions who's pred_prob is below the cuttoff threshold."""

    predictions_copy = copy.deepcopy(predictions)
    num_ann_to_zero = 0
    total_ann = 0
    for idx_predictions, prediction in enumerate(predictions_copy):
        for idx_class, class_prediction in enumerate(prediction):
            filtered_class_prediction = class_prediction[class_prediction[:, -1] >= threshold]
            if len(class_prediction) > 0:
                total_ann += 1
                if len(filtered_class_prediction) == 0:
                    num_ann_to_zero += 1

            predictions_copy[idx_predictions][idx_class] = filtered_class_prediction

    p_ann_pruned = total_ann and num_ann_to_zero / total_ann or 0  # avoid division by zero
    if p_ann_pruned > MAX_ALLOWED_BOX_PRUNE:
        warnings.warn(
            f"Pruning with threshold=={threshold} prunes {p_ann_pruned}% labels. Consider lowering the threshold.",
            UserWarning,
        )
    if verbose:
        print(
            f"Pruning {num_ann_to_zero} predictions out of {total_ann} using threshold=={threshold}. These predictions are no longer considered as potential candidates for identifying label issues as their similarity with the given labels is no longer considered."
        )
    return predictions_copy


def _separate_label(label: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Separates labels into bounding box and class label lists."""
    bboxes = label["bboxes"]
    labels = label["labels"]
    return bboxes, labels


# TODO: make object detection work for all predicted probabilities
def _separate_prediction_all_preds(
    prediction: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_bboxes, pred_labels, det_probs = prediction
    return pred_bboxes, pred_labels, det_probs


def _separate_prediction_single_box(
    prediction: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separates predictions into class labels, bounding boxes and pred_prob lists"""
    labels = []
    boxes = []
    for idx, prediction_class in enumerate(prediction):
        labels.extend([idx] * len(prediction_class))
        boxes.extend(prediction_class.tolist())
    bboxes = [box[:4] for box in boxes]
    pred_probs = [box[-1] for box in boxes]
    return np.array(bboxes), np.array(labels), np.array(pred_probs)


def _get_prediction_type(prediction: np.ndarray) -> str:
    if (
        len(prediction) == 3
        and prediction[0].shape[0] == prediction[2].shape[1]
        and prediction[1].shape[0] == prediction[2].shape[0]
    ):
        return "all_pred"
    else:
        return "single_pred"


def _separate_prediction(
    prediction, prediction_type="single_pred"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns bbox, label and pred_prob values for prediction."""

    if prediction_type == "all_pred":
        boxes, labels, pred_probs = _separate_prediction_all_preds(prediction)
    else:
        boxes, labels, pred_probs = _separate_prediction_single_box(prediction)
    return boxes, labels, pred_probs


def _mod_coordinates(x: List[float]) -> Dict[str, Any]:
    """Takes is a list of xyxy coordinates and returns them in dictionary format."""

    wd = {"x1": x[0], "y1": x[1], "x2": x[2], "y2": x[3]}
    return wd


def _get_overlap(bb1: List[float], bb2: List[float]) -> float:
    """Takes in two bounding boxes `bb1` and `bb2` and returns their IoU overlap."""

    return _get_iou(_mod_coordinates(bb1), _mod_coordinates(bb2))


def _get_overlap_matrix(bb1_list: np.ndarray, bb2_list: np.ndarray) -> np.ndarray:
    """Takes in two lists of bounding boxes and returns an IoU matrix where IoU[i][j] is the overlap between
    the i-th box in `bb1_list` and the j-th box in `bb2_list`."""
    wd = np.zeros(shape=(len(bb1_list), len(bb2_list)))
    for i in range(len(bb1_list)):
        for j in range(len(bb2_list)):
            wd[i][j] = _get_overlap(bb1_list[i], bb2_list[j])
    return wd


def _get_iou(bb1: Dict[str, Any], bb2: Dict[str, Any]) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    I've modified this to calculate overlap ratio in the line:
    iou = np.clip(intersection_area / float(min(bb1_area,bb2_area)),0.0,1.0)

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
    float
        in [0, 1]
    """
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    # There are some hyper-parameters here like consider tile area/object area
    return iou


def _has_overlap(bbox_list, labels):
    """This function determines whether each labeled box overlaps with another box of a different class (i.e. virtually the same box having multiple conflicting annotations). It returns a boolean array."""
    iou_matrix = _get_overlap_matrix(bbox_list, bbox_list)
    results_overlap = []
    for i in range(0, len(iou_matrix)):
        is_overlap = False
        for j in range(0, len(iou_matrix)):
            if i != j:
                if iou_matrix[i][j] >= LABEL_OVERLAP_THRESHOLD:
                    lab_1 = labels[i]
                    lab_2 = labels[j]
                    if lab_1 != lab_2:
                        is_overlap = True
        results_overlap.append(is_overlap)
    return np.array(results_overlap)


def _euc_dis(box1: List[float], box2: List[float]) -> float:
    """Calculates the Euclidean distance between `box1` and `box2`."""
    x1, y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    x2, y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    val2 = np.exp(-np.linalg.norm(p1 - p2) * EUC_FACTOR)
    return val2


def _get_dist_matrix(bb1_list: np.ndarray, bb2_list: np.ndarray) -> np.ndarray:
    """Returns a distance matrix of distances from all of boxes in bb1_list to all of boxes in bb2_list."""
    wd = np.zeros(shape=(len(bb1_list), len(bb2_list)))
    for i in range(len(bb1_list)):
        for j in range(len(bb2_list)):
            wd[i][j] = _euc_dis(bb1_list[i], bb2_list[j])
    return wd


def _get_min_possible_similarity(
    alpha: float,
    predictions,
    labels: List[Dict[str, Any]],
) -> float:
    """Gets the min possible similarity score between two bounding boxes out of all images."""
    min_possible_similarity = 1.0
    for prediction, label in zip(predictions, labels):
        lab_bboxes, lab_labels = _separate_label(label)
        pred_bboxes, pred_labels, _ = _separate_prediction(prediction)
        iou_matrix = _get_overlap_matrix(lab_bboxes, pred_bboxes)
        dist_matrix = 1 - _get_dist_matrix(lab_bboxes, pred_bboxes)
        similarity_matrix = iou_matrix * alpha + (1 - alpha) * (1 - dist_matrix)
        non_zero_similarity_matrix = similarity_matrix[np.nonzero(similarity_matrix)]
        min_image_similarity = (
            1.0 if 0 in non_zero_similarity_matrix.shape else np.min(non_zero_similarity_matrix)
        )
        min_possible_similarity = np.min([min_possible_similarity, min_image_similarity])
    return min_possible_similarity


def _get_valid_inputs_for_compute_scores_per_image(
    alpha: float,
    *,
    label: Optional[Dict[str, Any]] = None,
    prediction: Optional[np.ndarray] = None,
    pred_labels: Optional[np.ndarray] = None,
    pred_label_probs: Optional[np.ndarray] = None,
    pred_bboxes: Optional[np.ndarray] = None,
    lab_labels: Optional[np.ndarray] = None,
    lab_bboxes: Optional[np.ndarray] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    iou_matrix: Optional[np.ndarray] = None,
    min_possible_similarity: Optional[float] = None,
) -> AuxiliaryTypesDict:
    """Returns valid inputs for compute scores by either passing through values or calculating the inputs internally."""
    if lab_labels is None or lab_bboxes is None:
        if label is None:
            raise ValueError(
                f"Pass in either one of label or label labels into auxiliary inputs. Both can not be None."
            )
        lab_bboxes, lab_labels = _separate_label(label)

    if pred_labels is None or pred_label_probs is None or pred_bboxes is None:
        if prediction is None:
            raise ValueError(
                f"Pass in either one of prediction or prediction labels and prediction probabilities into auxiliary inputs. Both can not be None."
            )
        pred_bboxes, pred_labels, pred_label_probs = _separate_prediction(prediction)

    if similarity_matrix is None:
        iou_matrix = _get_overlap_matrix(lab_bboxes, pred_bboxes)
        dist_matrix = 1 - _get_dist_matrix(lab_bboxes, pred_bboxes)
        similarity_matrix = iou_matrix * alpha + (1 - alpha) * (1 - dist_matrix)

    if iou_matrix is None:
        iou_matrix = _get_overlap_matrix(lab_bboxes, pred_bboxes)

    if min_possible_similarity is None:
        min_possible_similarity = (
            1.0
            if 0 in similarity_matrix.shape
            else np.min(similarity_matrix[np.nonzero(similarity_matrix)])
        )

    auxiliary_input_dict: AuxiliaryTypesDict = {
        "pred_labels": pred_labels,
        "pred_label_probs": pred_label_probs,
        "pred_bboxes": pred_bboxes,
        "lab_labels": lab_labels,
        "lab_bboxes": lab_bboxes,
        "similarity_matrix": similarity_matrix,
        "iou_matrix": iou_matrix,
        "min_possible_similarity": min_possible_similarity,
    }

    return auxiliary_input_dict


def _get_valid_inputs_for_compute_scores(
    alpha: float,
    labels: Optional[List[Dict[str, Any]]] = None,
    predictions: Optional[List[np.ndarray]] = None,
) -> List[AuxiliaryTypesDict]:
    """Takes in alpha, labels and predictions and returns auxiliary input dictionary containing divided parts of labels and prediction per image."""
    if predictions is None or labels is None:
        raise ValueError(
            f"Predictions and labels can not be None. Both are needed to get valid inputs."
        )
    min_possible_similarity = _get_min_possible_similarity(alpha, predictions, labels)

    auxiliary_inputs = []

    for prediction, label in zip(predictions, labels):
        auxiliary_input_dict = _get_valid_inputs_for_compute_scores_per_image(
            alpha=alpha,
            label=label,
            prediction=prediction,
            min_possible_similarity=min_possible_similarity,
        )
        auxiliary_inputs.append(auxiliary_input_dict)

    return auxiliary_inputs


def _get_valid_score(scores_arr: np.ndarray, temperature: float) -> float:
    """Given scores array, returns valid score (softmin) or 1. Checks validity of score."""
    scores_arr = scores_arr[~np.isnan(scores_arr)]
    if len(scores_arr) > 0:
        valid_score = softmin1d(scores_arr, temperature=temperature)
    else:
        valid_score = 1.0
    return valid_score


def _get_valid_subtype_score_params(
    alpha: Optional[float] = None,
    low_probability_threshold: Optional[float] = None,
    high_probability_threshold: Optional[float] = None,
    temperature: Optional[float] = None,
):
    """This function returns valid params for subtype score. If param is None, then default constant is returned"""
    if alpha is None:
        alpha = ALPHA
    if low_probability_threshold is None:
        low_probability_threshold = LOW_PROBABILITY_THRESHOLD
    if high_probability_threshold is None:
        high_probability_threshold = HIGH_PROBABILITY_THRESHOLD
    if temperature is None:
        temperature = TEMPERATURE
    return alpha, low_probability_threshold, high_probability_threshold, temperature


def _get_aggregation_weights(
    aggregation_weights: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """This function validates aggregation weights, returning the default weights if none are provided."""
    if aggregation_weights is None:
        aggregation_weights = {
            "overlooked": CUSTOM_SCORE_WEIGHT_OVERLOOKED,
            "swap": CUSTOM_SCORE_WEIGHT_SWAP,
            "badloc": CUSTOM_SCORE_WEIGHT_BADLOC,
        }
    else:
        assert_valid_aggregation_weights(aggregation_weights)
    return aggregation_weights


def _compute_overlooked_box_scores_for_image(
    alpha: float,
    high_probability_threshold: float,
    label: Optional[Dict[str, Any]] = None,
    prediction: Optional[np.ndarray] = None,
    pred_labels: Optional[np.ndarray] = None,
    pred_label_probs: Optional[np.ndarray] = None,
    pred_bboxes: Optional[np.ndarray] = None,
    lab_labels: Optional[np.ndarray] = None,
    lab_bboxes: Optional[np.ndarray] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    iou_matrix: Optional[np.ndarray] = None,
    min_possible_similarity: Optional[float] = None,
) -> np.ndarray:
    """This method returns one score per predicted box (above threshold) in an image. Score from 0 to 1 ranking how overlooked the box is."""

    auxiliary_input_dict = _get_valid_inputs_for_compute_scores_per_image(
        alpha=alpha,
        label=label,
        prediction=prediction,
        pred_labels=pred_labels,
        pred_label_probs=pred_label_probs,
        pred_bboxes=pred_bboxes,
        lab_labels=lab_labels,
        lab_bboxes=lab_bboxes,
        similarity_matrix=similarity_matrix,
        min_possible_similarity=min_possible_similarity,
    )

    pred_labels = auxiliary_input_dict["pred_labels"]
    pred_label_probs = auxiliary_input_dict["pred_label_probs"]
    lab_labels = auxiliary_input_dict["lab_labels"]
    similarity_matrix = auxiliary_input_dict["similarity_matrix"]
    min_possible_similarity = auxiliary_input_dict["min_possible_similarity"]
    iou_matrix = auxiliary_input_dict["iou_matrix"]

    scores_overlooked = np.empty(len(pred_labels))  # same length as num of predicted boxes

    for iid, k in enumerate(pred_labels):
        if pred_label_probs[iid] < high_probability_threshold or np.any(iou_matrix[:, iid] > 0):
            scores_overlooked[iid] = np.nan
            continue

        k_similarity = similarity_matrix[lab_labels == k, iid]

        if len(k_similarity) == 0:  # if there are no annotated boxes of class k
            score = min_possible_similarity * (1 - pred_label_probs[iid])
        else:
            closest_annotated_box = np.argmax(k_similarity)
            score = k_similarity[closest_annotated_box]

        scores_overlooked[iid] = score

    return scores_overlooked


def compute_overlooked_box_scores(
    *,
    labels: Optional[List[Dict[str, Any]]] = None,
    predictions: Optional[List[np.ndarray]] = None,
    alpha: Optional[float] = None,
    high_probability_threshold: Optional[float] = None,
    auxiliary_inputs: Optional[List[AuxiliaryTypesDict]] = None,
) -> List[np.ndarray]:
    """
    Returns an array of overlooked box scores for each image.
    This is a helper method mostly for advanced users.

    An overlooked box error is when an image contains an object that is one of the given classes but there is no annotated bounding box around it.
    Score per high-confidence predicted bounding box is between 0 and 1, with lower values indicating boxes we are more confident were overlooked in the given label.

    Each image has ``L`` annotated bounding boxes and ``M`` predicted bounding boxes.
    A score is calculated for each predicted box in each of the ``N`` images in dataset.

    Note: ``M`` and ``L`` can be a different values for each image, as the number of annotated and predicted boxes varies.

    Parameters
    ----------
    labels:
        A list of ``N`` dictionaries such that ``labels[i]`` contains the given labels for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    predictions:
        A list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model predictions for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    alpha:
        Optional weighting between IoU and Euclidean distance when calculating similarity between predicted and annotated boxes. High alpha means weighting IoU more heavily over Euclidean distance. If no alpha is provided, a good default is used.

    high_probability_threshold:
        Optional probability threshold that determines which predicted boxes are considered high-confidence when computing overlooked scores. If not provided, a good default is used.

    auxiliary_inputs:
        Optional list of ``N`` dictionaries containing keys for sub-parts of label and prediction per image. Useful to minimize computation when computing multiple box scores for a single set of images. For the `i`-th image, `auxiliary_inputs[i]` should contain following keys:

       * pred_labels: np.ndarray
            Array of predicted classes for `i`-th image of shape ``(M,)``.
       * pred_label_probs: np.ndarray
            Array of predicted class probabilities for `i`-th image of shape ``(M,)``.
       * pred_bboxes: np.ndarray
            Array of predicted bounding boxes for `i`-th image of shape ``(M, 4)``.
       * lab_labels: np.ndarray
            Array of given label classed for `i`-th image of shape ``(L,)``.
       * lab_bboxes: np.ndarray
            Array of given label bounding boxes for `i`-th image of shape ``(L, 4)``.
       * similarity_matrix: np.ndarray
            Similarity matrix between labels and predictions `i`-th image.
       * min_possible_similarity: float
            Minimum possible similarity value greater than 0 between labels and predictions for the entire dataset.
    Returns
    ---------
    scores_overlooked:
        A list of ``N`` numpy arrays where scores_overlooked[i] is an array of size ``M`` of overlooked scores per predicted box for the `i`-th image.
    """
    (
        alpha,
        low_probability_threshold,
        high_probability_threshold,
        temperature,
    ) = _get_valid_subtype_score_params(alpha, None, high_probability_threshold, None)

    if auxiliary_inputs is None:
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(alpha, labels, predictions)

    scores_overlooked = []
    for auxiliary_input_dict in auxiliary_inputs:
        scores_overlooked_per_box = _compute_overlooked_box_scores_for_image(
            alpha=alpha,
            high_probability_threshold=high_probability_threshold,
            **auxiliary_input_dict,
        )
        scores_overlooked.append(scores_overlooked_per_box)
    return scores_overlooked


def _compute_badloc_box_scores_for_image(
    alpha: float,
    low_probability_threshold: float,
    label: Optional[Dict[str, Any]] = None,
    prediction: Optional[np.ndarray] = None,
    pred_labels: Optional[np.ndarray] = None,
    pred_label_probs: Optional[np.ndarray] = None,
    pred_bboxes: Optional[np.ndarray] = None,
    lab_labels: Optional[np.ndarray] = None,
    lab_bboxes: Optional[np.ndarray] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    iou_matrix: Optional[np.ndarray] = None,
    min_possible_similarity: Optional[float] = None,
) -> np.ndarray:
    """This method returns one score per labeled box in an image. Score from 0 to 1 ranking how badly located the box is."""

    auxiliary_input_dict = _get_valid_inputs_for_compute_scores_per_image(
        alpha=alpha,
        label=label,
        prediction=prediction,
        pred_labels=pred_labels,
        pred_label_probs=pred_label_probs,
        pred_bboxes=pred_bboxes,
        lab_labels=lab_labels,
        lab_bboxes=lab_bboxes,
        similarity_matrix=similarity_matrix,
        iou_matrix=iou_matrix,
        min_possible_similarity=min_possible_similarity,
    )
    pred_labels = auxiliary_input_dict["pred_labels"]
    pred_label_probs = auxiliary_input_dict["pred_label_probs"]
    lab_labels = auxiliary_input_dict["lab_labels"]
    similarity_matrix = auxiliary_input_dict["similarity_matrix"]
    iou_matrix = auxiliary_input_dict["iou_matrix"]

    scores_badloc = np.empty(len(lab_labels))

    for iid, k in enumerate(lab_labels):
        k_similarity = similarity_matrix[iid, pred_labels == k]
        k_pred = pred_label_probs[pred_labels == k]
        k_iou = iou_matrix[iid, pred_labels == k]

        if len(k_pred) == 0 or np.max(k_pred) <= low_probability_threshold:
            scores_badloc[iid] = 1.0
            continue

        idx_at_least_low_probability_threshold = np.where(k_pred > low_probability_threshold)[0]
        idx_at_least_intersection_threshold = np.where(k_iou > 0)[0]
        combined_idx = np.intersect1d(
            idx_at_least_low_probability_threshold, idx_at_least_intersection_threshold
        )

        k_similarity = k_similarity[combined_idx]
        k_pred = k_pred[combined_idx]

        scores_badloc[iid] = np.max(k_similarity) if len(k_pred) > 0 else 1.0
    return scores_badloc


def compute_badloc_box_scores(
    *,
    labels: Optional[List[Dict[str, Any]]] = None,
    predictions: Optional[List[np.ndarray]] = None,
    alpha: Optional[float] = None,
    low_probability_threshold: Optional[float] = None,
    auxiliary_inputs: Optional[List[AuxiliaryTypesDict]] = None,
) -> List[np.ndarray]:
    """
    Returns a numeric score for each annotated bounding box in each image, estimating the likelihood that the edges of this box are not badly located.
    This is a helper method mostly for advanced users.

    A badly located box error is when a box has the correct label but incorrect coordinates so it does not correctly encapsulate the entire object it is for.
    Score per high-confidence predicted bounding box is between 0 and 1, with lower values indicating boxes we are more confident were overlooked in the given label.

    Each image has ``L`` annotated bounding boxes and ``M`` predicted bounding boxes.
    A score is calculated for each predicted box in each of the ``N`` images in dataset.

    Note: ``M`` and ``L`` can be a different values for each image, as the number of annotated and predicted boxes varies.

    Parameters
    ----------
    labels:
        A list of ``N`` dictionaries such that ``labels[i]`` contains the given labels for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    predictions:
        A list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model predictions for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    alpha:
        Optional weighting between IoU and Euclidean distance when calculating similarity between predicted and annotated boxes. High alpha means weighting IoU more heavily over Euclidean distance. If no alpha is provided, a good default is used.

    low_probability_threshold:
        Optional minimum probability threshold that determines which predicted boxes are considered when computing badly located scores. If not provided, a good default is used.

    auxiliary_inputs:
        Optional list of ``N`` dictionaries containing keys for sub-parts of label and prediction per image. Useful to minimize computation when computing multiple box scores for a single set of images. For the `i`-th image, `auxiliary_inputs[i]` should contain following keys:

       * pred_labels: np.ndarray
            Array of predicted classes for `i`-th image of shape ``(M,)``.
       * pred_label_probs: np.ndarray
            Array of predicted class probabilities for `i`-th image of shape ``(M,)``.
       * pred_bboxes: np.ndarray
            Array of predicted bounding boxes for `i`-th image of shape ``(M, 4)``.
       * lab_labels: np.ndarray
            Array of given label classed for `i`-th image of shape ``(L,)``.
       * lab_bboxes: np.ndarray
            Array of given label bounding boxes for `i`-th image of shape ``(L, 4)``.
       * similarity_matrix: np.ndarray
            Similarity matrix between labels and predictions `i`-th image.
       * min_possible_similarity: float
            Minimum possible similarity value greater than 0 between labels and predictions for the entire dataset.
    Returns
    ---------
    scores_badloc:
        A list of ``N`` numpy arrays where scores_badloc[i] is an array of size ``L`` badly located scores per annotated box for the `i`-th image.
    """
    (
        alpha,
        low_probability_threshold,
        high_probability_threshold,
        temperature,
    ) = _get_valid_subtype_score_params(alpha, low_probability_threshold, None, None)
    if auxiliary_inputs is None:
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(alpha, labels, predictions)

    scores_badloc = []
    for auxiliary_input_dict in auxiliary_inputs:
        scores_badloc_per_box = _compute_badloc_box_scores_for_image(
            alpha=alpha, low_probability_threshold=low_probability_threshold, **auxiliary_input_dict
        )
        scores_badloc.append(scores_badloc_per_box)
    return scores_badloc


def _compute_swap_box_scores_for_image(
    alpha: float,
    high_probability_threshold: float,
    label: Optional[Dict[str, Any]] = None,
    prediction: Optional[np.ndarray] = None,
    pred_labels: Optional[np.ndarray] = None,
    pred_label_probs: Optional[np.ndarray] = None,
    pred_bboxes: Optional[np.ndarray] = None,
    lab_labels: Optional[np.ndarray] = None,
    lab_bboxes: Optional[np.ndarray] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    iou_matrix: Optional[np.ndarray] = None,
    min_possible_similarity: Optional[float] = None,
    overlapping_label_check: Optional[bool] = True,
) -> np.ndarray:
    """This method returns one score per labeled box in an image. Score from 0 to 1 ranking how likeley swapped the box is."""

    auxiliary_input_dict = _get_valid_inputs_for_compute_scores_per_image(
        alpha=alpha,
        label=label,
        prediction=prediction,
        pred_labels=pred_labels,
        pred_label_probs=pred_label_probs,
        pred_bboxes=pred_bboxes,
        lab_labels=lab_labels,
        lab_bboxes=lab_bboxes,
        similarity_matrix=similarity_matrix,
        min_possible_similarity=min_possible_similarity,
    )

    pred_labels = auxiliary_input_dict["pred_labels"]
    pred_label_probs = auxiliary_input_dict["pred_label_probs"]
    lab_labels = auxiliary_input_dict["lab_labels"]
    similarity_matrix = auxiliary_input_dict["similarity_matrix"]
    min_possible_similarity = auxiliary_input_dict["min_possible_similarity"]

    if overlapping_label_check:
        has_overlap_label_bboxes = _has_overlap(lab_bboxes, lab_labels)
    else:
        has_overlap_label_bboxes = np.array([False] * len(lab_labels))

    scores_swap = np.empty(len(lab_labels))

    for iid, k in enumerate(lab_labels):
        not_k_idx = np.where(pred_labels != k)[0]
        if has_overlap_label_bboxes[iid]:
            scores_swap[iid] = min_possible_similarity
            continue
        if not_k_idx.size == 0 or np.all(pred_label_probs[not_k_idx] <= high_probability_threshold):
            scores_swap[iid] = 1.0
            continue

        not_k_pred = pred_label_probs[not_k_idx]
        idx_at_least_high_probability_threshold = np.where(not_k_pred > high_probability_threshold)[
            0
        ]
        not_k_similarity = similarity_matrix[iid, not_k_idx][
            idx_at_least_high_probability_threshold
        ]

        closest_predicted_box = np.argmax(not_k_similarity)
        score = np.max([min_possible_similarity, 1 - not_k_similarity[closest_predicted_box]])
        scores_swap[iid] = score

    return scores_swap


def compute_swap_box_scores(
    *,
    labels: Optional[List[Dict[str, Any]]] = None,
    predictions: Optional[List[np.ndarray]] = None,
    alpha: Optional[float] = None,
    high_probability_threshold: Optional[float] = None,
    overlapping_label_check: Optional[bool] = True,
    auxiliary_inputs: Optional[List[AuxiliaryTypesDict]] = None,
) -> List[np.ndarray]:
    """
    Returns a numeric score for each annotated bounding box in each image, estimating the likelihood that the class label for this box was not accidentally swapped with another class.
    This is a helper method mostly for advanced users.

    A swapped box error occurs when a bounding box should be labeled as a class different to what the current label is.
    Score per high-confidence predicted bounding box is between 0 and 1, with lower values indicating boxes we are more confident were overlooked in the given label.

    Each image has ``L`` annotated bounding boxes and ``M`` predicted bounding boxes.
    A score is calculated for each predicted box in each of the ``N`` images in dataset.

    Note: ``M`` and ``L`` can be a different values for each image, as the number of annotated and predicted boxes varies.

    Parameters
    ----------
    labels:
        A list of ``N`` dictionaries such that ``labels[i]`` contains the given labels for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    predictions:
        A list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model predictions for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    alpha:
        Optional weighting between IoU and Euclidean distance when calculating similarity between predicted and annotated boxes. High alpha means weighting IoU more heavily over Euclidean distance. If no alpha is provided, a good default is used.

    high_probability_threshold:
        Optional probability threshold that determines which predicted boxes are considered high-confidence when computing overlooked scores. If not provided, a good default is used.

    overlapping_label_check : bool, default = True
        If True, boxes annotated with more than one class label have their swap score penalized. Set this to False if you are not concerned when two very similar boxes exist with different class labels in the given annotations.

    auxiliary_inputs:
        Optional list of ``N`` dictionaries containing keys for sub-parts of label and prediction per image. Useful to minimize computation when computing multiple box scores for a single set of images. For the `i`-th image, `auxiliary_inputs[i]` should contain following keys:

       * pred_labels: np.ndarray
            Array of predicted classes for `i`-th image of shape ``(M,)``.
       * pred_label_probs: np.ndarray
            Array of predicted class probabilities for `i`-th image of shape ``(M,)``.
       * pred_bboxes: np.ndarray
            Array of predicted bounding boxes for `i`-th image of shape ``(M, 4)``.
       * lab_labels: np.ndarray
            Array of given label classed for `i`-th image of shape ``(L,)``.
       * lab_bboxes: np.ndarray
            Array of given label bounding boxes for `i`-th image of shape ``(L, 4)``.
       * similarity_matrix: np.ndarray
            Similarity matrix between labels and predictions `i`-th image.
       * min_possible_similarity: float
            Minimum possible similarity value greater than 0 between labels and predictions for the entire dataset.
    Returns
    ---------
    scores_swap:
        A list of ``N`` numpy arrays where scores_swap[i] is an array of size ``L`` swap scores per annotated box for the `i`-th image.
    """
    (
        alpha,
        low_probability_threshold,
        high_probability_threshold,
        temperature,
    ) = _get_valid_subtype_score_params(alpha, None, high_probability_threshold, None)

    if auxiliary_inputs is None:
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(alpha, labels, predictions)

    scores_swap = []
    for auxiliary_inputs in auxiliary_inputs:
        scores_swap_per_box = _compute_swap_box_scores_for_image(
            alpha=alpha,
            high_probability_threshold=high_probability_threshold,
            overlapping_label_check=overlapping_label_check,
            **auxiliary_inputs,
        )
        scores_swap.append(scores_swap_per_box)
    return scores_swap


def pool_box_scores_per_image(
    box_scores: List[np.ndarray], *, temperature: Optional[float] = None
) -> np.ndarray:
    """
    Aggregates all per-box scores within an image to return a single quality score for the image rather than for individual boxes within it.
    This is a helper method mostly for advanced users to be used with the outputs of :py:func:`object_detection.rank.compute_overlooked_box_scores <cleanlab.object_detection.rank.compute_overlooked_box_scores>`, :py:func:`object_detection.rank.compute_badloc_box_scores <cleanlab.object_detection.rank.compute_badloc_box_scores>`, and :py:func:`object_detection.rank.compute_swap_box_scores <cleanlab.object_detection.rank.compute_swap_box_scores>`.

    Score per image is between 0 and 1, with lower values indicating we are more confident image contains an error.

    Parameters
    ----------
    box_scores:
        A list of ``N`` numpy arrays where box_scores[i] is an array of badly located scores per box for the `i`-th image.

    temperature:
        Optional temperature of the softmin function where a lower value suggests softmin acts closer to min. If not provided, a good default is used.

    Returns
    ---------
    image_scores:
        An array of size ``N`` where ``image_scores[i]`` represents the score for the `i`-th image.
    """

    (
        alpha,
        low_probability_threshold,
        high_probability_threshold,
        temperature,
    ) = _get_valid_subtype_score_params(None, None, None, temperature)

    image_scores = np.empty(
        shape=[
            len(box_scores),
        ]
    )
    for idx, box_score in enumerate(box_scores):
        image_score = _get_valid_score(box_score, temperature=temperature)
        image_scores[idx] = image_score
    return image_scores


def _get_subtype_label_quality_scores(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    *,
    alpha: Optional[float] = None,
    low_probability_threshold: Optional[float] = None,
    high_probability_threshold: Optional[float] = None,
    temperature: Optional[float] = None,
    aggregation_weights: Optional[Dict[str, float]] = None,
    overlapping_label_check: Optional[bool] = True,
) -> np.ndarray:
    """
    Returns a label quality score for each of the ``N`` images in the dataset.
    Score is between 0 and 1.

    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels:
        A list of ``N`` dictionaries such that ``labels[i]`` contains the given labels for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    predictions:
        A list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model predictions for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    alpha:
        Optional weighting between IoU and Euclidean distance when calculating similarity between predicted and annotated boxes. High alpha means weighting IoU more heavily over Euclidean distance. If no alpha is provided, a good default is used.

    low_probability_threshold:
        Optional minimum probability threshold that determines which predicted boxes are considered when computing badly located scores. If not provided, a good default is used.

    high_probability_threshold:
        Optional probability threshold that determines which predicted boxes are considered high-confidence when computing overlooked and swapped scores. If not provided, a good default is used.

    temperature:
        Optional temperature of the softmin function where a lower score suggests softmin acts closer to min. If not provided, a good default is used.

    overlapping_label_check : bool, default = True
        If True, boxes annotated with more than one class label have their swap score penalized. Set this to False if you are not concerned when two very similar boxes exist with different class labels in the given annotations.

    Returns
    ---------
    label_quality_scores:
        As returned by :py:func:`get_label_quality_scores <cleanlab.outlier.get_label_quality_scores>`. See function for more details.
    """
    (
        alpha,
        low_probability_threshold,
        high_probability_threshold,
        temperature,
    ) = _get_valid_subtype_score_params(
        alpha, low_probability_threshold, high_probability_threshold, temperature
    )
    auxiliary_inputs = _get_valid_inputs_for_compute_scores(alpha, labels, predictions)
    aggregation_weights = _get_aggregation_weights(aggregation_weights)

    overlooked_scores_per_box = compute_overlooked_box_scores(
        alpha=alpha,
        high_probability_threshold=high_probability_threshold,
        auxiliary_inputs=auxiliary_inputs,
    )
    overlooked_score_per_image = pool_box_scores_per_image(
        overlooked_scores_per_box, temperature=temperature
    )

    badloc_scores_per_box = compute_badloc_box_scores(
        alpha=alpha,
        low_probability_threshold=low_probability_threshold,
        auxiliary_inputs=auxiliary_inputs,
    )
    badloc_score_per_image = pool_box_scores_per_image(
        badloc_scores_per_box, temperature=temperature
    )

    swap_scores_per_box = compute_swap_box_scores(
        alpha=alpha,
        high_probability_threshold=high_probability_threshold,
        auxiliary_inputs=auxiliary_inputs,
        overlapping_label_check=overlapping_label_check,
    )
    swap_score_per_image = pool_box_scores_per_image(swap_scores_per_box, temperature=temperature)

    scores = (
        aggregation_weights["overlooked"] * np.log(TINY_VALUE + overlooked_score_per_image)
        + aggregation_weights["badloc"] * np.log(TINY_VALUE + badloc_score_per_image)
        + aggregation_weights["swap"] * np.log(TINY_VALUE + swap_score_per_image)
    )

    scores = np.exp(scores)

    return scores

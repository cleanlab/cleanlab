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
import warnings
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import copy

# for visualizing functions
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# for MAP
from multiprocessing import Pool

"""Methods to rank and score images in an object detection dataset (object detection data), based on how likely they
are to contain label errors. """


def get_label_quality_scores(
    annotations: List[Dict[Any, Any]],
    predictions: List[np.ndarray],
    *,
    method: str = "subtype",
    probability_threshold: Optional[float] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Returns a label quality score for each datapoint.

    This is a function to compute label quality scores for object detection datasets,
    where lower scores indicate image annotaion is less likely to be correct.

    Score is between 0 and 1.

    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    annotations:
        A list of `N` dictionaries for `N` images such that `annotations[i]` contains the given annotations for the `i`-th image in the format
       `{'bboxes': np.ndarray((M,4)), 'labels': np.ndarray((M,)), 'image_name': str}` where `M` is the number of annotated bounding boxes
       for the `i`-th image and `bboxes[j]` is in the format [x,y,x,y] with given label `labels[j]`. ('image_name' is optional here)

    predictions:
        A list of `N` `np.ndarray` for `N` images such that `predictions[i]` corresponds to the model predictions for the `i`-th image
        in the format `np.ndarray((K,))` where K is the number of classes and `predictions[i][k]` is of shape `np.ndarray(M,5)`
        where `M` is the number of predicted bounding boxes for class `K` and the five columns correspond to `[x,y,x,y,pred_prob]` returned
        by the model.

        Note: `M` number of predicted bounding boxes can be different from `M` number of annotated bounding boxes for class `K` of `i`-th image.

    method:
        The method used to calculate label_quality_scores.

    probability_threshold:
        Bounding boxes in `predictions` with `pred_prob` below the threshold are not considered for computing label_quality_scores.
        If you know what probability-threshold was used when producing predicted boxes from your trained object detector,
        please supply the value that was used here. If not provided, this value is inferred based on the smallest observed
        predicted probability for any of the predicted boxes.

    verbose : bool, default = True
      Set to ``False`` to suppress all print statements.

    Returns
    ---------
    label_quality_scores:
        Array of shape ``(N, )`` of scores between 0 and 1, one per image in the dataset.
        Lower scores indicate images are more likely to contain an incorrect annotation.
    """

    assert_valid_inputs(
        annotations=annotations,
        predictions=predictions,
        method=method,
        threshold=probability_threshold,
    )

    return _compute_label_quality_scores(
        annotations=annotations,
        predictions=predictions,
        method=method,
        threshold=probability_threshold,
        verbose=verbose,
    )


def issues_from_scores(label_quality_scores: np.ndarray, *, threshold: float = 0.1) -> np.ndarray:
    """Returns a list of indices images with issues sorted from most to least severe.

    Parameters
    ----------
    label_quality_scores:
        Array of shape ``(N, )`` of scores between 0 and 1, one per image in the dataset.
        Lower scores indicate images are more likely to contain a label issue.

    threshold:
        Label quality scores above the threshold are not considered as issues and their indices are omited from the return

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

    issue_indices = np.argwhere(label_quality_scores < threshold).flatten()
    issue_vals = label_quality_scores[issue_indices]
    sorted_idx = issue_vals.argsort()
    return issue_indices[sorted_idx]


def _compute_label_quality_scores(
    annotations: List[Dict[Any, Any]],
    predictions: List[np.ndarray],
    *,
    method: str = "subtype",
    threshold: Optional[float] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Internal function to prune extra bounding boxes and compute label quality scores based on passed in method."""

    pred_probs_prepruned = False
    min_pred_prob = _get_min_pred_prob(predictions)

    if threshold is not None:
        predictions = _prune_by_threshold(
            predictions=predictions, threshold=threshold, verbose=verbose
        )
        if np.abs(min_pred_prob - threshold) < 0.001 and threshold > 0:
            pred_probs_prepruned = True  # the provided threshold is the threshold used for pre_pruning the pred_probs during model prediction.
    else:
        threshold = min_pred_prob  # assume model was not pre_pruned if no threshold was provided

    if method == "subtype":
        scores = _compute_subtype_lqs(
            annotations,
            predictions,
            alpha=0.99,
            low_probability_threshold=0.7,
            high_probability_threshold=0.95,
            temperature=0.98,
        )
    elif method == "map":
        scores = _compute_map(predictions, annotations)
    return scores


def _get_min_pred_prob(
    predictions: List[np.ndarray],
) -> Tuple[List[Dict[Any, Any]], List[np.ndarray]]:
    """Returns min pred_prob out of all predictions."""
    pred_probs = [1.0]  # avoid calling np.min on empty array.
    for prediction in predictions:
        for class_prediction in prediction:
            pred_probs.extend(list(class_prediction[:, -1]))

    min_pred_prob = np.min(pred_probs)
    return min_pred_prob


def _prune_by_threshold(
    predictions: List[np.ndarray], threshold: float, verbose: bool = True
) -> Tuple[List[Dict[Any, Any]], List[np.ndarray]]:
    """Removes predicted bounding boxes from predictions who's pred_prob is below the cuttoff threshold."""

    max_allowed_box_prune = 0.97  # This is max allowed percent of prune for boxes below threhold before a warning is thrown.
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
    if p_ann_pruned > max_allowed_box_prune:
        warnings.warn(
            f"Pruning with threshold=={threshold} prunes {p_ann_pruned}% annotations. Consider lowering the threshold.",
            UserWarning,
        )
    if verbose:
        print(
            f"Pruning {num_ann_to_zero} annotations out of {total_ann} using threshold=={threshold}."
        )
    return predictions_copy


# Todo: make this more descriptive and assert better inputs
def assert_valid_inputs(annotations, predictions, method=None, threshold=None):
    """Asserts proper input format."""
    if len(annotations) != len(predictions):
        raise ValueError(
            f"Annotations and predictions length needs to match. len(annotations) == {len(annotations)} while len(predictions) == {len(predictions)}."
        )
    # Typecheck annotations and predictions
    if not isinstance(annotations[0], dict):
        raise ValueError(
            f"Annotations has to be a list of dicts. Instead it is list of {type(annotations[0])}."
        )
    # check last column of predictions is probabilities ( < 1.)?
    if not isinstance(predictions[0], np.ndarray):
        raise ValueError(
            f"Prediction has to be a list of np.ndarray. Instead it is list of {type(predictions[0])}."
        )
    if not predictions[0][0].shape[1] == 5:
        raise ValueError(f"Prediction values have to be of format [_,_,_,_,pred_prob].")

    valid_methods = ["map", "subtype"]
    if method is not None and method not in valid_methods:
        raise ValueError(
            f"""
            {method} is not a valid object detection scoring method!
            Please choose a valid scoring_method: {valid_methods}
            """
        )

    if threshold is not None and threshold > 1.0:
        raise ValueError(
            f"""
            Threshold is a cutoff of predicted probabilities and therefore should be <= 1.
            """
        )


def visualize(
    image_path: str,
    annotation: Dict[Any, Any],
    prediction: np.ndarray,
    *,
    prediction_threshold: Optional[float] = None,
    given_label_overlay: bool = True,
    class_labels: Optional[Dict[Any, Any]] = None,
    figsize: Optional[Tuple[int, int]] = None,
):
    """Visualize bounding box annotations (given labels) and model predictions for an image. The given label annotations
    are shown with red while the predicted annotations shown in blue.

        Parameters
        ----------
        image_path:
            Full path to the image file.

        annotation:
            The given annotation for a single image in the format {'bboxes': np.ndarray((N,4)), 'labels': np.ndarray((N,))}` where
            N is the number of bounding boxes for the `i`-th image and `bboxes[j]` is in the format [x,y,x,y] with given label `labels[j]`.

        prediction:
            A prediction for a single image in the format `np.ndarray((K,))` where K is the number of classes and `prediction[k]` is of shape `np.ndarray(N,5)`
            where `N` is the number of bounding boxes for class `K` and the five columns correspond to `[x,y,x,y,pred_prob]` returned
            by the model.

        prediction_threshold:
            Minimum pred_probs value of a bounding box output by the model. All bounding boxes with pred_probs below this threshold are
            omited from the visualization.

        given_label_overlay: bool
            If true, a single image with overlayed given label and predicted annotations is shown. If false, two images side
            by side are shown instead with the left image being given label and right being the ground truth annotation.

        class_labels:
            Optional dictionary mapping one-hot-encoded class labels back to their original class names in the format {"one-hot-label": "original-class-name"}.

        figsize:
            Optional figuresize for plotting the visualizations. Corresponds to matplotlib.figure.figsize.
    """

    prediction_type = _get_prediction_type(prediction)

    # Create figure and axes
    image = plt.imread(image_path)
    pbbox, plabels, pred_probs = _get_bbox_labels_prediction(
        prediction, prediction_type=prediction_type
    )
    abbox, alabels = _get_bbox_labels_annotation(annotation)

    if prediction_threshold is not None:
        keep_idx = np.where(pred_probs > prediction_threshold)
        pbbox = pbbox[keep_idx]
        plabels = plabels[keep_idx]

    if given_label_overlay:
        figsize = (8, 5) if figsize is None else figsize
        fig, ax = plt.subplots(frameon=False, figsize=figsize)
        plt.axis("off")
        ax.imshow(image)

        fig, ax = _draw_boxes(fig, ax, pbbox, plabels, edgecolor="b", linestyle="-", linewidth=1)
        _, _ = _draw_boxes(fig, ax, abbox, alabels, edgecolor="r", linestyle="-.", linewidth=1)
    else:
        figsize = (14, 10) if figsize is None else figsize
        fig, axes = plt.subplots(nrows=1, ncols=2, frameon=False, figsize=figsize)
        axes[0].axis("off")
        axes[0].imshow(image)
        axes[1].axis("off")
        axes[1].imshow(image)

        fig, ax = _draw_boxes(
            fig, axes[0], pbbox, plabels, edgecolor="b", linestyle="-", linewidth=2
        )
        _, _ = _draw_boxes(fig, axes[1], abbox, alabels, edgecolor="r", linestyle="-.", linewidth=2)

    _ = _plot_legend(class_labels)
    plt.show()


def _plot_legend(class_labels):
    MAX_CLASS_TO_SHOW = 10  # idea: we can also show top 10 most popular classes?

    colors = ["black"] + ["red", "blue"]
    markers = [None] + ["s", "s"]
    labels = [r"$\bf{Legend}$", "given label", "predicted label"]

    if class_labels:
        colors += ["black"] + ["black"] * min(len(class_labels), MAX_CLASS_TO_SHOW)
        markers += [None] + [f"${class_key}$" for class_key in class_labels.keys()]
        labels += [r"$\bf{classes}$"] + list(class_labels.values())

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f(marker, color) for marker, color in zip(markers, colors)]
    legend = plt.legend(
        handles, labels, bbox_to_anchor=(1.04, 0.05), loc="lower left", borderaxespad=0
    )

    return legend


def _draw_labels(ax, rect, label, edgecolor):
    """Helper function to draw labels on an axis."""

    rx, ry = rect.get_xy()
    c_xleft = rx + 10
    c_xright = rx + rect.get_width() - 10
    c_ytop = ry + 12
    c_ybottom = ry + rect.get_height() - 10

    if edgecolor == "r":
        cx, cy = c_xright, c_ytop
    elif edgecolor == "b":
        cx, cy = c_xleft, c_ytop
    else:
        cx, cy = c_xleft, c_ybottom

    l = ax.annotate(
        label, (cx, cy), fontsize=8, fontweight="bold", color="white", ha="center", va="center"
    )
    l.set_bbox(dict(facecolor=edgecolor, alpha=0.35, edgecolor=edgecolor, pad=2))
    return ax


def _draw_boxes(fig, ax, bboxes, labels, edgecolor="g", linestyle="-", linewidth=3):
    """Helper function to draw bboxes and labels on an axis."""
    bboxes = [_bbox_xyxy_to_xywh(box) for box in bboxes]
    for (x, y, w, h), label in zip(bboxes, labels):
        rect = Rectangle(
            (x, y),
            w,
            h,
            linewidth=linewidth,
            linestyle=linestyle,
            edgecolor=edgecolor,
            facecolor="none",
        )
        ax.add_patch(rect)

        if labels is not None:
            ax = _draw_labels(ax, rect, label, edgecolor)

    return fig, ax


def _get_bbox_labels_annotation(annotation):
    """Returns bbox and label values for annotation."""
    bboxes = annotation["bboxes"]
    labels = annotation["labels"]
    return bboxes, labels


def _get_bbox_labels_prediction_all_preds(prediction):
    det_bboxes, det_labels, det_probs = prediction
    return det_bboxes, det_labels, det_probs


def _get_bbox_labels_prediction_single_box(prediction):
    """Returns bbox, label and pred_prob values for prediction."""
    labels = []
    boxes = []
    for idx, prediction_class in enumerate(prediction):
        labels.extend([idx] * len(prediction_class))
        boxes.extend(prediction_class.tolist())
    bboxes = [box[:4] for box in boxes]
    pred_probs = [box[-1] for box in boxes]
    return np.array(bboxes), np.array(labels), np.array(pred_probs)


def _get_prediction_type(prediction):
    if (
        len(prediction) == 3
        and prediction[0].shape == prediction[2].shape
        and prediction[0].shape[0] == prediction[1].shape[0]
    ):
        return "all_pred"
    else:
        return "single_pred"


def _get_bbox_labels_prediction(prediction, prediction_type="single_pred"):
    """Returns bbox, label and pred_prob values for prediction."""

    if prediction_type == "all_pred":
        boxes, labels, pred_probs = _get_bbox_labels_prediction_all_preds(prediction)
    else:
        boxes, labels, pred_probs = _get_bbox_labels_prediction_single_box(prediction)
    return boxes, labels, pred_probs


def _bbox_xyxy_to_xywh(bbox):
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        return [x1, y1, w, h]
    else:
        print("Wrong bbox shape", len(bbox))
        return None


# ============= Simple label subtype algorithm ============


# setup labeling helper functions
def _seperate_annotations(annotation):
    """Seperates annotations into annotation and bounding box lists."""

    lab_bboxes = annotation["bboxes"]
    lab_annotations = annotation["labels"]
    return lab_bboxes, lab_annotations


def _separate_predictions(prediction):
    """Seperates predictions into annotation, bounding boxes and pred_probs"""

    det_bboxes = []
    det_annotations = []
    det_annotation_prob = []
    cnt = 0
    for i in prediction:
        for j in i:
            if len(j) != 0:
                det_bboxes.append(j[:-1])
                det_annotations.append(cnt)
                det_annotation_prob.append(j[-1])
        cnt += 1
    return det_bboxes, det_annotations, det_annotation_prob


def _mod_coordinates(x):
    """Takes is a list of xyxy coordinates and returns them in dictionary format."""
    wd = {}
    wd["x1"], wd["y1"], wd["x2"], wd["y2"] = x[0], x[1], x[2], x[3]
    return wd


# distance/similarity helped functions
def _get_overlap(bb1, bb2):
    """Takes in two bounding boxes `bb1` and `bb2` and returns their IoU overlap."""
    return _get_iou(_mod_coordinates(bb1), _mod_coordinates(bb2))


def _get_overlap_matrix(bb1_list, bb2_list):
    """Takes in two lists of bounding boxes and returns an IoU matrix where IoU[i][j] is the overlap between
    the i-th box in `bb1_list` and the j-th box in `bb2_list`."""
    wd = np.zeros(shape=(len(bb1_list), len(bb2_list)))
    for i in range(len(bb1_list)):
        for j in range(len(bb2_list)):
            wd[i][j] = _get_overlap(bb1_list[i], bb2_list[j])
    return wd


def _get_iou(bb1, bb2):
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
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

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
    assert intersection_area - 0.1 <= bb1_area
    assert intersection_area - 0.1 <= bb2_area

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    # There are some hyper-parameters here like consider tile area/object area
    assert iou >= 0.0
    assert iou - 0.01 <= 1.0
    return iou


def _euc_dis(box1, box2):
    """Calculates the euclidian distance between `box1` and `box2`."""
    euc_factor = 0.1  # this is a hyperparameter

    x1, y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    x2, y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    val2 = np.exp(-np.linalg.norm(p1 - p2) * euc_factor)
    return val2


def _get_dist_matrix(bb1_list, bb2_list):
    """Returns a distance matrix of distances from all of boxes in bb1_list to all of boxes in bb2_list."""
    wd = np.zeros(shape=(len(bb1_list), len(bb2_list)))
    for i in range(len(bb1_list)):
        for j in range(len(bb2_list)):
            wd[i][j] = _euc_dis(bb1_list[i], bb2_list[j])
    return wd


def _softmax(x, temperature=0.99, axis=0):
    """Gets softmax of scores."""
    x = x / temperature
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def _softmin1D(scores, temperature=0.99, axis=0):
    """Returns softmin of passed in scores."""
    scores = np.array(scores)
    softmax_scores = _softmax(-1 * scores, temperature, axis)
    return np.dot(softmax_scores, scores)


def _get_valid_score(scores_arr, temperature=0.99):
    """Given scores array, returns valid score (softmin) or 1. Checks validity of score."""
    scores_arr = np.array(scores_arr)
    if len(scores_arr) > 0:
        valid_score = _softmin1D(scores_arr, temperature=temperature)
    else:
        valid_score = 1.0
    return valid_score


def _get_min_possible_similarity(predictions, annotations, alpha):
    """Gets the min possible similarity score between two bounding boxes out of all examples."""
    min_possible_similarity = 1.0
    for prediction, annotation in zip(predictions, annotations):
        lab_bboxes, lab_annotations = _seperate_annotations(annotation)
        det_bboxes, det_annotations, det_annotation_prob = _separate_predictions(prediction)
        det_annotation_prob = np.array(det_annotation_prob)
        iou_matrix = _get_overlap_matrix(lab_bboxes, det_bboxes)
        dist_matrix = 1 - _get_dist_matrix(lab_bboxes, det_bboxes)
        similarity_matrix = iou_matrix * alpha + (1 - alpha) * (1 - dist_matrix)
        min_image_similarity = 1.0 if 0 in similarity_matrix.shape else np.min(similarity_matrix)
        if min_image_similarity > 0:
            min_possible_similarity = np.min([min_possible_similarity, min_image_similarity])
    return min_possible_similarity


def _compute_subtype_lqs(
    annotations,
    predictions,
    *,
    alpha,
    low_probability_threshold,
    high_probability_threshold,
    temperature,
):
    """
    Returns a label quality score for each datapoint.
    Score is between 0 and 1.

    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    annotations:
        A list of `N` dictionaries for `N` images such that `annotations[i]` contains the given annotations for the `i`-th image in the format
       `{'bboxes': np.ndarray((M,4)), 'labels': np.ndarray((M,)), 'image_name': str}` where `M` is the number of annotated bounding boxes
       for the `i`-th image and `bboxes[j]` is in the format [x,y,x,y] with given label `labels[j]`. ('image_name' is optional here)

    predictions:
        A list of `N` `np.ndarray` for `N` images such that `predictions[i]` corresponds to the model predictions for the `i`-th image
        in the format `np.ndarray((K,))` where K is the number of classes and `predictions[i][k]` is of shape `np.ndarray(M,5)`
        where `M` is the number of predicted bounding boxes for class `K` and the five columns correspond to `[x,y,x,y,pred_prob]` returned
        by the model.

        Note: `M` number of predicted bounding boxes can be different from `M` number of annotated bounding boxes for class `K` of `i`-th image.

    alpha:
        Weight between IoU and distance when considering similarity matrix. High alpha means considering IoU more strongly over distance.

    low_probability_threshold:
        The lowest prediction threshold allowed when considering predicted boxes to identify badly located annotation boxes.

    high_probability_threshold:
        The high probability threshold for considering predicted boxes to identify overlooked and swapped annotation boxes.

    temperature:
        Temperature of the softmin function where a lower score suggests softmin acts closer to min.

    Returns
    ---------
    label_quality_scores:
        Array of shape ``(N, )`` of scores between 0 and 1, one per image in the dataset.
        Lower scores indicate images are more likely to contain an incorrect annotation.
    """
    # scores_overlooked_per_image = []
    # scores_badloc_per_image = []
    # scores_swapped_per_image = []

    scores = []

    min_possible_similarity = _get_min_possible_similarity(predictions, annotations, alpha)
    for prediction, annotation in zip(predictions, annotations):
        lab_bboxes, lab_annotations = _seperate_annotations(annotation)
        det_bboxes, det_annotations, det_annotation_prob = _separate_predictions(prediction)
        det_annotation_prob = np.array(det_annotation_prob)
        iou_matrix = _get_overlap_matrix(lab_bboxes, det_bboxes)
        dist_matrix = 1 - _get_dist_matrix(lab_bboxes, det_bboxes)

        similarity_matrix = iou_matrix * alpha + (1 - alpha) * (1 - dist_matrix)
        assert (similarity_matrix.flatten() >= 0).all() and (similarity_matrix.flatten() <= 1).all()

        scores_overlooked = []
        for iid, k in enumerate(det_annotations):
            if det_annotation_prob[iid] < high_probability_threshold:
                continue

            k_similarity = similarity_matrix[lab_annotations == k, iid]
            if len(k_similarity) == 0:  # if there is no annotated box
                scores_overlooked.append(min_possible_similarity * (1 - det_annotation_prob[iid]))
            else:
                closest_annotated_box = np.argmax(k_similarity)
                scores_overlooked.append(k_similarity[closest_annotated_box])
        score_overlooked = _get_valid_score(scores_overlooked, temperature)
        # scores_overlooked_per_image.append(score_overlooked)

        scores_badloc = []
        for iid, k in enumerate(lab_annotations):  # for every annotated box
            k_similarity = similarity_matrix[iid, det_annotations == k]
            k_pred = det_annotation_prob[det_annotations == k]

            if len(k_pred) == 0:  # there are no predicted boxes of class k
                scores_badloc.append(min_possible_similarity)
                continue

            idx_at_least_low_probability_threshold = k_pred > low_probability_threshold
            k_similarity = k_similarity[idx_at_least_low_probability_threshold]
            k_pred = k_pred[idx_at_least_low_probability_threshold]
            assert len(k_pred) == len(k_similarity)
            if len(k_pred) == 0:
                scores_badloc.append(min_possible_similarity)
            else:
                scores_badloc.append(np.max(k_similarity))
        score_badloc = _get_valid_score(scores_badloc, temperature)
        # scores_badloc_per_image.append(score_badloc)

        scores_swap = []
        for iid, k in enumerate(lab_annotations):
            not_k_idx = det_annotations != k

            if len(not_k_idx) == 0:
                scores_swap.append(1.0)
                continue

            not_k_similarity = similarity_matrix[iid, not_k_idx]
            not_k_pred = det_annotation_prob[not_k_idx]

            idx_at_least_high_probability_threshold = not_k_pred > high_probability_threshold
            if len(idx_at_least_high_probability_threshold) == 0:
                scores_swap.append(1.0)
                continue

            not_k_similarity = not_k_similarity[idx_at_least_high_probability_threshold]
            if len(not_k_similarity) == 0:  # if there is no annotated box
                scores_swap.append(1.0)
            else:
                closest_predicted_box = np.argmax(not_k_similarity)
                score = np.max(
                    [min_possible_similarity, 1 - not_k_similarity[closest_predicted_box]]
                )
                scores_swap.append(score)
        score_swap = _get_valid_score(scores_swap, temperature)
        # scores_swapped_per_image.append(score_swap)
        scores.append(
            _softmin1D([score_overlooked, score_badloc, score_swap], temperature=temperature)
        )
    # scores = [_softmin1D(np.array([so,sbl,ssw]), temperature) for so,sbl,ssw in zip(SO,SBL,SSw)]

    # scores = _softmin1D(np.array((scores_overlooked_per_image, scores_badloc_per_image, scores_swapped_per_image)), temperature)
    return np.array(scores)


# # ==========TO BE DEPRECATED: Example score (calculate mAP)=============
# def _compute_map(annotations, predictions):
#     map = np.zeros((len(annotations),))
#     for i, (d, r) in enumerate(zip(annotations, predictions)):
#         map[i] = bbox_map_eval(d, r)
#     return map
#
#
# def bbox_map_eval(det_prediction, annotation, nproc=4):
#     """Evaluate mAP of single image det prediction.
#     Args:
#         det_prediction (list[list]): [[cls1_det, cls2_det, ...], ...].
#             The outer list indicates images, and the inner list indicates
#             per-class detected bboxes.
#         annotation (dict): Ground truth annotations where keys of
#              annotations are:
#             - bboxes: numpy array of shape (n, 4)
#             - annotations: numpy array of shape (n, )
#             - bboxes_ignore (optional): numpy array of shape (k, 4)
#             - annotations_ignore (optional): numpy array of shape (k, )
#         nproc (int): Processes used for computing mAP.
#             Default: 4.
#     Returns:
#         float: mAP
#     """
#     # use only bbox det prediction
#     if isinstance(det_prediction, tuple):
#         bbox_det_prediction = [det_prediction[0]]
#     else:
#         bbox_det_prediction = [det_prediction]
#     # mAP
#     iou_thrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
#
#     processes = []
#     workers = Pool(processes=nproc)
#     for thr in iou_thrs:
#         p = workers.apply_async(
#             eval_map,
#             (bbox_det_prediction, [annotation]),
#             {"iou_thr": thr, "logger": "silent", "nproc": 1},
#         )
#         processes.append(p)
#
#     workers.close()
#     workers.join()
#
#     mean_aps = []
#     for p in processes:
#         mean_aps.append(p.get()[0])
#
#     return sum(mean_aps) / len(mean_aps)
#
#
# def tpfp_default(
#     det_bboxes,
#     gt_bboxes,
#     gt_bboxes_ignore=None,
#     iou_thr=0.5,
#     area_ranges=None,
#     use_legacy_coordinate=False,
#     **kwargs,
# ):
#     """Check if detected bboxes are true positive or false positive.
#     Args:
#         det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
#         gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
#         gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
#             of shape (k, 4). Default: None
#         iou_thr (float): IoU threshold to be considered as matched.
#             Default: 0.5.
#         area_ranges (list[tuple] | None): Range of bbox areas to be
#             evaluated, in the format [(min1, max1), (min2, max2), ...].
#             Default: None.
#         use_legacy_coordinate (bool): Whether to use coordinate system in
#             mmdet v1.x. which means width, height should be
#             calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
#             Default: False.
#     Returns:
#         tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
#         each array is (num_scales, m).
#     """
#
#     if not use_legacy_coordinate:
#         extra_length = 0.0
#     else:
#         extra_length = 1.0
#
#     # an indicator of ignored gts
#     gt_ignore_inds = np.concatenate(
#         (np.zeros(gt_bboxes.shape[0], dtype=bool), np.ones(gt_bboxes_ignore.shape[0], dtype=bool))
#     )
#     # stack gt_bboxes and gt_bboxes_ignore for convenience
#     gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))
#
#     num_dets = det_bboxes.shape[0]
#     num_gts = gt_bboxes.shape[0]
#     if area_ranges is None:
#         area_ranges = [(None, None)]
#     num_scales = len(area_ranges)
#     # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
#     # a certain scale
#     tp = np.zeros((num_scales, num_dets), dtype=np.float32)
#     fp = np.zeros((num_scales, num_dets), dtype=np.float32)
#
#     # if there is no gt bboxes in this image, then all det bboxes
#     # within area range are false positives
#     if gt_bboxes.shape[0] == 0:
#         if area_ranges == [(None, None)]:
#             fp[...] = 1
#         else:
#             det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length) * (
#                 det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length
#             )
#             for i, (min_area, max_area) in enumerate(area_ranges):
#                 fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
#         return tp, fp
#
#     ious = bbox_overlaps(det_bboxes, gt_bboxes, use_legacy_coordinate=use_legacy_coordinate)
#     # for each det, the max iou with all gts
#     ious_max = ious.max(axis=1)
#     # for each det, which gt overlaps most with it
#     ious_argmax = ious.argmax(axis=1)
#     # sort all dets in descending order by scores
#     sort_inds = np.argsort(-det_bboxes[:, -1])
#     for k, (min_area, max_area) in enumerate(area_ranges):
#         gt_covered = np.zeros(num_gts, dtype=bool)
#         # if no area range is specified, gt_area_ignore is all False
#         if min_area is None:
#             gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
#         else:
#             gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + extra_length) * (
#                 gt_bboxes[:, 3] - gt_bboxes[:, 1] + extra_length
#             )
#             gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
#         for i in sort_inds:
#             if ious_max[i] >= iou_thr:
#                 matched_gt = ious_argmax[i]
#                 if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
#                     if not gt_covered[matched_gt]:
#                         gt_covered[matched_gt] = True
#                         tp[k, i] = 1
#                     else:
#                         fp[k, i] = 1
#                 # otherwise ignore this detected bbox, tp = 0, fp = 0
#             elif min_area is None:
#                 fp[k, i] = 1
#             else:
#                 bbox = det_bboxes[i, :4]
#                 area = (bbox[2] - bbox[0] + extra_length) * (bbox[3] - bbox[1] + extra_length)
#                 if area >= min_area and area < max_area:
#                     fp[k, i] = 1
#     return tp, fp
#
#
# def get_cls_group_ofs(annotations, class_id):
#     """Get `gt_group_of` of a certain class, which is used in Open Images.
#
#     Args:
#         annotations (list[dict]): Same as `eval_map()`.
#         class_id (int): ID of a specific class.
#
#     Returns:
#         list[np.ndarray]: `gt_group_of` of a certain class.
#     """
#     gt_group_ofs = []
#     for ann in annotations:
#         gt_inds = ann["annotations"] == class_id
#         if ann.get("gt_is_group_ofs", None) is not None:
#             gt_group_ofs.append(ann["gt_is_group_ofs"][gt_inds])
#         else:
#             gt_group_ofs.append(np.empty((0, 1), dtype=np.bool_))
#
#     return gt_group_ofs
#
#
# def get_cls_predictions(det_predictions, annotations, class_id):
#     """Get det predictions and gt information of a certain class.
#
#     Args:
#         det_predictions (list[list]): Same as `eval_map()`.
#         annotations (list[dict]): Same as `eval_map()`.
#         class_id (int): ID of a specific class.
#
#     Returns:
#         tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
#     """
#     cls_dets = [img_res[class_id] for img_res in det_predictions]
#     cls_gts = []
#     cls_gts_ignore = []
#     for ann in annotations:
#         gt_inds = ann["annotations"] == class_id
#         cls_gts.append(ann["bboxes"][gt_inds, :])
#
#         if ann.get("annotations_ignore", None) is not None:
#             ignore_inds = ann["annotations_ignore"] == class_id
#             cls_gts_ignore.append(ann["bboxes_ignore"][ignore_inds, :])
#         else:
#             cls_gts_ignore.append(np.empty((0, 4), dtype=np.float32))
#
#     return cls_dets, cls_gts, cls_gts_ignore
#
#
# def eval_map(
#     det_predictions,
#     annotations,
#     scale_ranges=None,
#     iou_thr=0.5,
#     ioa_thr=None,
#     dataset=None,
#     logger=None,
#     tpfp_fn=None,
#     nproc=4,
#     use_legacy_coordinate=False,
#     use_group_of=False,
# ):
#     """Evaluate mAP of a dataset.
#     Args:
#         det_predictions (list[list]): [[cls1_det, cls2_det, ...], ...].
#             The outer list indicates images, and the inner list indicates
#             per-class detected bboxes.
#         annotations (list[dict]): Ground truth annotations where each item of
#             the list indicates an image. Keys of annotations are:
#             - `bboxes`: numpy array of shape (n, 4)
#             - `annotations`: numpy array of shape (n, )
#             - `bboxes_ignore` (optional): numpy array of shape (k, 4)
#             - `annotations_ignore` (optional): numpy array of shape (k, )
#         scale_ranges (list[tuple] | None): Range of scales to be evaluated,
#             in the format [(min1, max1), (min2, max2), ...]. A range of
#             (32, 64) means the area range between (32**2, 64**2).
#             Default: None.
#         iou_thr (float): IoU threshold to be considered as matched.
#             Default: 0.5.
#         ioa_thr (float | None): IoA threshold to be considered as matched,
#             which only used in OpenImages evaluation. Default: None.
#         dataset (list[str] | str | None): Dataset name or dataset classes,
#             there are minor differences in metrics for different datasets, e.g.
#             "voc07", "imagenet_det", etc. Default: None.
#         logger (logging.Logger | str | None): The way to print the mAP
#             summary. See `mmcv.utils.print_log()` for details. Default: None.
#         tpfp_fn (callable | None): The function used to determine true/
#             false positives. If None, :func:`tpfp_default` is used as default
#             unless dataset is 'det' or 'vid' (:func:`tpfp_imagenet` in this
#             case). If it is given as a function, then this function is used
#             to evaluate tp & fp. Default None.
#         nproc (int): Processes used for computing TP and FP.
#             Default: 4.
#         use_legacy_coordinate (bool): Whether to use coordinate system in
#             mmdet v1.x. which means width, height should be
#             calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
#             Default: False.
#         use_group_of (bool): Whether to use group of when calculate TP and FP,
#             which only used in OpenImages evaluation. Default: False.
#     Returns:
#         tuple: (mAP, [dict, dict, ...])
#     """
#     assert len(det_predictions) == len(annotations)
#     if not use_legacy_coordinate:
#         extra_length = 0.0
#     else:
#         extra_length = 1.0
#
#     num_imgs = len(det_predictions)
#     num_scales = len(scale_ranges) if scale_ranges is not None else 1
#     num_classes = len(det_predictions[0])  # positive class num
#     area_ranges = (
#         [(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges] if scale_ranges is not None else None
#     )
#
#     # There is no need to use multi processes to process
#     # when num_imgs = 1 .
#     if num_imgs > 1:
#         assert nproc > 0, "nproc must be at least one."
#         nproc = min(nproc, num_imgs)
#         pool = Pool(nproc)
#
#     eval_predictions = []
#     for i in range(num_classes):
#         # get gt and det bboxes of this class
#         cls_dets, cls_gts, cls_gts_ignore = get_cls_predictions(det_predictions, annotations, i)
#         # choose proper function according to datasets to compute tp and fp
#         if tpfp_fn is None:
#             if dataset in ["det", "vid"]:
#                 tpfp_fn = tpfp_imagenet
#             elif dataset in ["oid_challenge", "oid_v6"] or use_group_of is True:
#                 tpfp_fn = tpfp_openimages
#             else:
#                 tpfp_fn = tpfp_default
#         if not callable(tpfp_fn):
#             raise ValueError(f"tpfp_fn has to be a function or None, but got {tpfp_fn}")
#
#         if num_imgs > 1:
#             # compute tp and fp for each image with multiple processes
#             args = []
#             if use_group_of:
#                 # used in Open Images Dataset evaluation
#                 gt_group_ofs = get_cls_group_ofs(annotations, i)
#                 args.append(gt_group_ofs)
#                 args.append([use_group_of for _ in range(num_imgs)])
#             if ioa_thr is not None:
#                 args.append([ioa_thr for _ in range(num_imgs)])
#
#             tpfp = pool.starmap(
#                 tpfp_fn,
#                 zip(
#                     cls_dets,
#                     cls_gts,
#                     cls_gts_ignore,
#                     [iou_thr for _ in range(num_imgs)],
#                     [area_ranges for _ in range(num_imgs)],
#                     [use_legacy_coordinate for _ in range(num_imgs)],
#                     *args,
#                 ),
#             )
#         else:
#             tpfp = tpfp_fn(
#                 cls_dets[0],
#                 cls_gts[0],
#                 cls_gts_ignore[0],
#                 iou_thr,
#                 area_ranges,
#                 use_legacy_coordinate,
#                 # gt_bboxes_group_of=(get_cls_group_ofs(annotations, i)[0]
#                 #                     if use_group_of else None),
#                 # use_group_of=use_group_of, # todo: figure out why commented out
#                 # ioa_thr=ioa_thr
#             )
#             tpfp = [tpfp]
#
#         if use_group_of:
#             tp, fp, cls_dets = tuple(zip(*tpfp))
#         else:
#             tp, fp = tuple(zip(*tpfp))
#         # calculate gt number of each scale
#         # ignored gts or gts beyond the specific scale are not counted
#         num_gts = np.zeros(num_scales, dtype=int)
#         for j, bbox in enumerate(cls_gts):
#             if area_ranges is None:
#                 num_gts[0] += bbox.shape[0]
#             else:
#                 gt_areas = (bbox[:, 2] - bbox[:, 0] + extra_length) * (
#                     bbox[:, 3] - bbox[:, 1] + extra_length
#                 )
#                 for k, (min_area, max_area) in enumerate(area_ranges):
#                     num_gts[k] += np.sum((gt_areas >= min_area) & (gt_areas < max_area))
#         # sort all det bboxes by score, also sort tp and fp
#         cls_dets = np.vstack(cls_dets)
#         num_dets = cls_dets.shape[0]
#         sort_inds = np.argsort(-cls_dets[:, -1])
#         tp = np.hstack(tp)[:, sort_inds]
#         fp = np.hstack(fp)[:, sort_inds]
#         # calculate recall and precision with tp and fp
#         tp = np.cumsum(tp, axis=1)
#         fp = np.cumsum(fp, axis=1)
#         eps = np.finfo(np.float32).eps
#         recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
#         precisions = tp / np.maximum((tp + fp), eps)
#         # calculate AP
#         if scale_ranges is None:
#             recalls = recalls[0, :]
#             precisions = precisions[0, :]
#             num_gts = num_gts.item()
#         mode = "area" if dataset != "voc07" else "11points"
#         ap = average_precision(recalls, precisions, mode)
#         eval_predictions.append(
#             {
#                 "num_gts": num_gts,
#                 "num_dets": num_dets,
#                 "recall": recalls,
#                 "precision": precisions,
#                 "ap": ap,
#             }
#         )
#
#     if num_imgs > 1:
#         pool.close()
#
#     if scale_ranges is not None:
#         # shape (num_classes, num_scales)
#         all_ap = np.vstack([cls_prediction["ap"] for cls_prediction in eval_predictions])
#         all_num_gts = np.vstack([cls_prediction["num_gts"] for cls_prediction in eval_predictions])
#         mean_ap = []
#         for i in range(num_scales):
#             if np.any(all_num_gts[:, i] > 0):
#                 mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
#             else:
#                 mean_ap.append(0.0)
#     else:
#         aps = []
#         for cls_prediction in eval_predictions:
#             if cls_prediction["num_gts"] > 0:
#                 aps.append(cls_prediction["ap"])
#         mean_ap = np.array(aps).mean().item() if aps else 0.0
#
#     return mean_ap, eval_predictions
#
#
# def tpfp_imagenet(
#     det_bboxes,
#     gt_bboxes,
#     gt_bboxes_ignore=None,
#     default_iou_thr=0.5,
#     area_ranges=None,
#     use_legacy_coordinate=False,
# ):
#     """Check if detected bboxes are true positive or false positive.
#
#     Args:
#         det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
#         gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
#         gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
#             of shape (k, 4). Default: None
#         default_iou_thr (float): IoU threshold to be considered as matched for
#             medium and large bboxes (small ones have special rules).
#             Default: 0.5.
#         area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
#             in the format [(min1, max1), (min2, max2), ...]. Default: None.
#         use_legacy_coordinate (bool): Whether to use coordinate system in
#             mmdet v1.x. which means width, height should be
#             calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
#             Default: False.
#
#     Returns:
#         tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
#         each array is (num_scales, m).
#     """
#
#     if not use_legacy_coordinate:
#         extra_length = 0.0
#     else:
#         extra_length = 1.0
#
#     # an indicator of ignored gts
#     gt_ignore_inds = np.concatenate(
#         (
#             np.zeros(gt_bboxes.shape[0], dtype=np.bool_),
#             np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool_),
#         )
#     )
#     # stack gt_bboxes and gt_bboxes_ignore for convenience
#     gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))
#
#     num_dets = det_bboxes.shape[0]
#     num_gts = gt_bboxes.shape[0]
#     if area_ranges is None:
#         area_ranges = [(None, None)]
#     num_scales = len(area_ranges)
#     # tp and fp are of shape (num_scales, num_gts), each row is tp or fp
#     # of a certain scale.
#     tp = np.zeros((num_scales, num_dets), dtype=np.float32)
#     fp = np.zeros((num_scales, num_dets), dtype=np.float32)
#     if gt_bboxes.shape[0] == 0:
#         if area_ranges == [(None, None)]:
#             fp[...] = 1
#         else:
#             det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length) * (
#                 det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length
#             )
#             for i, (min_area, max_area) in enumerate(area_ranges):
#                 fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
#         return tp, fp
#     ious = bbox_overlaps(det_bboxes, gt_bboxes - 1, use_legacy_coordinate=use_legacy_coordinate)
#     gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0] + extra_length
#     gt_h = gt_bboxes[:, 3] - gt_bboxes[:, 1] + extra_length
#     iou_thrs = np.minimum((gt_w * gt_h) / ((gt_w + 10.0) * (gt_h + 10.0)), default_iou_thr)
#     # sort all detections by scores in descending order
#     sort_inds = np.argsort(-det_bboxes[:, -1])
#     for k, (min_area, max_area) in enumerate(area_ranges):
#         gt_covered = np.zeros(num_gts, dtype=bool)
#         # if no area range is specified, gt_area_ignore is all False
#         if min_area is None:
#             gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
#         else:
#             gt_areas = gt_w * gt_h
#             gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
#         for i in sort_inds:
#             max_iou = -1
#             matched_gt = -1
#             # find best overlapped available gt
#             for j in range(num_gts):
#                 # different from PASCAL VOC: allow finding other gts if the
#                 # best overlapped ones are already matched by other det bboxes
#                 if gt_covered[j]:
#                     continue
#                 elif ious[i, j] >= iou_thrs[j] and ious[i, j] > max_iou:
#                     max_iou = ious[i, j]
#                     matched_gt = j
#             # there are 4 cases for a det bbox:
#             # 1. it matches a gt, tp = 1, fp = 0
#             # 2. it matches an ignored gt, tp = 0, fp = 0
#             # 3. it matches no gt and within area range, tp = 0, fp = 1
#             # 4. it matches no gt but is beyond area range, tp = 0, fp = 0
#             if matched_gt >= 0:
#                 gt_covered[matched_gt] = 1
#                 if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
#                     tp[k, i] = 1
#             elif min_area is None:
#                 fp[k, i] = 1
#             else:
#                 bbox = det_bboxes[i, :4]
#                 area = (bbox[2] - bbox[0] + extra_length) * (bbox[3] - bbox[1] + extra_length)
#                 if area >= min_area and area < max_area:
#                     fp[k, i] = 1
#     return tp, fp
#
#
# def tpfp_default(
#     det_bboxes,
#     gt_bboxes,
#     gt_bboxes_ignore=None,
#     iou_thr=0.5,
#     area_ranges=None,
#     use_legacy_coordinate=False,
# ):
#     """Check if detected bboxes are true positive or false positive.
#
#     Args:
#         det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
#         gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
#         gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
#             of shape (k, 4). Default: None
#         iou_thr (float): IoU threshold to be considered as matched.
#             Default: 0.5.
#         area_ranges (list[tuple] | None): Range of bbox areas to be
#             evaluated, in the format [(min1, max1), (min2, max2), ...].
#             Default: None.
#         use_legacy_coordinate (bool): Whether to use coordinate system in
#             mmdet v1.x. which means width, height should be
#             calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
#             Default: False.
#
#     Returns:
#         tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
#         each array is (num_scales, m).
#     """
#
#     if not use_legacy_coordinate:
#         extra_length = 0.0
#     else:
#         extra_length = 1.0
#
#     # an indicator of ignored gts
#     gt_ignore_inds = np.concatenate(
#         (
#             np.zeros(gt_bboxes.shape[0], dtype=np.bool_),
#             np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool_),
#         )
#     )
#     # stack gt_bboxes and gt_bboxes_ignore for convenience
#     gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))
#
#     num_dets = det_bboxes.shape[0]
#     num_gts = gt_bboxes.shape[0]
#     if area_ranges is None:
#         area_ranges = [(None, None)]
#     num_scales = len(area_ranges)
#     # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
#     # a certain scale
#     tp = np.zeros((num_scales, num_dets), dtype=np.float32)
#     fp = np.zeros((num_scales, num_dets), dtype=np.float32)
#
#     # if there is no gt bboxes in this image, then all det bboxes
#     # within area range are false positives
#     if gt_bboxes.shape[0] == 0:
#         if area_ranges == [(None, None)]:
#             fp[...] = 1
#         else:
#             det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length) * (
#                 det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length
#             )
#             for i, (min_area, max_area) in enumerate(area_ranges):
#                 fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
#         return tp, fp
#
#     ious = bbox_overlaps(det_bboxes, gt_bboxes, use_legacy_coordinate=use_legacy_coordinate)
#     # for each det, the max iou with all gts
#     ious_max = ious.max(axis=1)
#     # for each det, which gt overlaps most with it
#     ious_argmax = ious.argmax(axis=1)
#     # sort all dets in descending order by scores
#     sort_inds = np.argsort(-det_bboxes[:, -1])
#     for k, (min_area, max_area) in enumerate(area_ranges):
#         gt_covered = np.zeros(num_gts, dtype=bool)
#         # if no area range is specified, gt_area_ignore is all False
#         if min_area is None:
#             gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
#         else:
#             gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + extra_length) * (
#                 gt_bboxes[:, 3] - gt_bboxes[:, 1] + extra_length
#             )
#             gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
#         for i in sort_inds:
#             if ious_max[i] >= iou_thr:
#                 matched_gt = ious_argmax[i]
#                 if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
#                     if not gt_covered[matched_gt]:
#                         gt_covered[matched_gt] = True
#                         tp[k, i] = 1
#                     else:
#                         fp[k, i] = 1
#                 # otherwise ignore this detected bbox, tp = 0, fp = 0
#             elif min_area is None:
#                 fp[k, i] = 1
#             else:
#                 bbox = det_bboxes[i, :4]
#                 area = (bbox[2] - bbox[0] + extra_length) * (bbox[3] - bbox[1] + extra_length)
#                 if area >= min_area and area < max_area:
#                     fp[k, i] = 1
#     return tp, fp
#
#
# def tpfp_openimages(
#     det_bboxes,
#     gt_bboxes,
#     gt_bboxes_ignore=None,
#     iou_thr=0.5,
#     area_ranges=None,
#     use_legacy_coordinate=False,
#     gt_bboxes_group_of=None,
#     use_group_of=True,
#     ioa_thr=0.5,
# ):
#     """Check if detected bboxes are true positive or false positive.
#
#     Args:
#         det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
#         gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
#         gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
#             of shape (k, 4). Default: None
#         iou_thr (float): IoU threshold to be considered as matched.
#             Default: 0.5.
#         area_ranges (list[tuple] | None): Range of bbox areas to be
#             evaluated, in the format [(min1, max1), (min2, max2), ...].
#             Default: None.
#         use_legacy_coordinate (bool): Whether to use coordinate system in
#             mmdet v1.x. which means width, height should be
#             calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
#             Default: False.
#         gt_bboxes_group_of (ndarray): GT group_of of this image, of shape
#             (k, 1). Default: None
#         use_group_of (bool): Whether to use group of when calculate TP and FP,
#             which only used in OpenImages evaluation. Default: True.
#         ioa_thr (float | None): IoA threshold to be considered as matched,
#             which only used in OpenImages evaluation. Default: 0.5.
#
#     Returns:
#         tuple[np.ndarray]: Returns a tuple (tp, fp, det_bboxes), where
#         (tp, fp) whose elements are 0 and 1. The shape of each array is
#         (num_scales, m). (det_bboxes) whose will filter those are not
#         matched by group of gts when processing Open Images evaluation.
#         The shape is (num_scales, m).
#     """
#
#     if not use_legacy_coordinate:
#         extra_length = 0.0
#     else:
#         extra_length = 1.0
#
#     # an indicator of ignored gts
#     gt_ignore_inds = np.concatenate(
#         (
#             np.zeros(gt_bboxes.shape[0], dtype=np.bool_),
#             np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool_),
#         )
#     )
#     # stack gt_bboxes and gt_bboxes_ignore for convenience
#     gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))
#
#     num_dets = det_bboxes.shape[0]
#     num_gts = gt_bboxes.shape[0]
#     if area_ranges is None:
#         area_ranges = [(None, None)]
#     num_scales = len(area_ranges)
#     # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
#     # a certain scale
#     tp = np.zeros((num_scales, num_dets), dtype=np.float32)
#     fp = np.zeros((num_scales, num_dets), dtype=np.float32)
#
#     # if there is no gt bboxes in this image, then all det bboxes
#     # within area range are false positives
#     if gt_bboxes.shape[0] == 0:
#         if area_ranges == [(None, None)]:
#             fp[...] = 1
#         else:
#             det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length) * (
#                 det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length
#             )
#             for i, (min_area, max_area) in enumerate(area_ranges):
#                 fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
#         return tp, fp, det_bboxes
#
#     if gt_bboxes_group_of is not None and use_group_of:
#         # if handle group-of boxes, divided gt boxes into two parts:
#         # non-group-of and group-of.Then calculate ious and ioas through
#         # non-group-of group-of gts respectively. This only used in
#         # OpenImages evaluation.
#         assert gt_bboxes_group_of.shape[0] == gt_bboxes.shape[0]
#         non_group_gt_bboxes = gt_bboxes[~gt_bboxes_group_of]
#         group_gt_bboxes = gt_bboxes[gt_bboxes_group_of]
#         num_gts_group = group_gt_bboxes.shape[0]
#         ious = bbox_overlaps(det_bboxes, non_group_gt_bboxes)
#         ioas = bbox_overlaps(det_bboxes, group_gt_bboxes, mode="iof")
#     else:
#         # if not consider group-of boxes, only calculate ious through gt boxes
#         ious = bbox_overlaps(det_bboxes, gt_bboxes, use_legacy_coordinate=use_legacy_coordinate)
#         ioas = None
#
#     if ious.shape[1] > 0:
#         # for each det, the max iou with all gts
#         ious_max = ious.max(axis=1)
#         # for each det, which gt overlaps most with it
#         ious_argmax = ious.argmax(axis=1)
#         # sort all dets in descending order by scores
#         sort_inds = np.argsort(-det_bboxes[:, -1])
#         for k, (min_area, max_area) in enumerate(area_ranges):
#             gt_covered = np.zeros(num_gts, dtype=bool)
#             # if no area range is specified, gt_area_ignore is all False
#             if min_area is None:
#                 gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
#             else:
#                 gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + extra_length) * (
#                     gt_bboxes[:, 3] - gt_bboxes[:, 1] + extra_length
#                 )
#                 gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
#             for i in sort_inds:
#                 if ious_max[i] >= iou_thr:
#                     matched_gt = ious_argmax[i]
#                     if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
#                         if not gt_covered[matched_gt]:
#                             gt_covered[matched_gt] = True
#                             tp[k, i] = 1
#                         else:
#                             fp[k, i] = 1
#                     # otherwise ignore this detected bbox, tp = 0, fp = 0
#                 elif min_area is None:
#                     fp[k, i] = 1
#                 else:
#                     bbox = det_bboxes[i, :4]
#                     area = (bbox[2] - bbox[0] + extra_length) * (bbox[3] - bbox[1] + extra_length)
#                     if area >= min_area and area < max_area:
#                         fp[k, i] = 1
#     else:
#         # if there is no no-group-of gt bboxes in this image,
#         # then all det bboxes within area range are false positives.
#         # Only used in OpenImages evaluation.
#         if area_ranges == [(None, None)]:
#             fp[...] = 1
#         else:
#             det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length) * (
#                 det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length
#             )
#             for i, (min_area, max_area) in enumerate(area_ranges):
#                 fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
#
#     if ioas is None or ioas.shape[1] <= 0:
#         return tp, fp, det_bboxes
#     else:
#         # The evaluation of group-of TP and FP are done in two stages:
#         # 1. All detections are first matched to non group-of boxes; true
#         #    positives are determined.
#         # 2. Detections that are determined as false positives are matched
#         #    against group-of boxes and calculated group-of TP and FP.
#         # Only used in OpenImages evaluation.
#         det_bboxes_group = np.zeros((num_scales, ioas.shape[1], det_bboxes.shape[1]), dtype=float)
#         match_group_of = np.zeros((num_scales, num_dets), dtype=bool)
#         tp_group = np.zeros((num_scales, num_gts_group), dtype=np.float32)
#         ioas_max = ioas.max(axis=1)
#         # for each det, which gt overlaps most with it
#         ioas_argmax = ioas.argmax(axis=1)
#         # sort all dets in descending order by scores
#         sort_inds = np.argsort(-det_bboxes[:, -1])
#         for k, (min_area, max_area) in enumerate(area_ranges):
#             box_is_covered = tp[k]
#             # if no area range is specified, gt_area_ignore is all False
#             if min_area is None:
#                 gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
#             else:
#                 gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
#                 gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
#             for i in sort_inds:
#                 matched_gt = ioas_argmax[i]
#                 if not box_is_covered[i]:
#                     if ioas_max[i] >= ioa_thr:
#                         if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
#                             if not tp_group[k, matched_gt]:
#                                 tp_group[k, matched_gt] = 1
#                                 match_group_of[k, i] = True
#                             else:
#                                 match_group_of[k, i] = True
#
#                             if det_bboxes_group[k, matched_gt, -1] < det_bboxes[i, -1]:
#                                 det_bboxes_group[k, matched_gt] = det_bboxes[i]
#
#         fp_group = (tp_group <= 0).astype(float)
#         tps = []
#         fps = []
#         # concatenate tp, fp, and det-boxes which not matched group of
#         # gt boxes and tp_group, fp_group, and det_bboxes_group which
#         # matched group of boxes respectively.
#         for i in range(num_scales):
#             tps.append(np.concatenate((tp[i][~match_group_of[i]], tp_group[i])))
#             fps.append(np.concatenate((fp[i][~match_group_of[i]], fp_group[i])))
#             det_bboxes = np.concatenate((det_bboxes[~match_group_of[i]], det_bboxes_group[i]))
#
#         tp = np.vstack(tps)
#         fp = np.vstack(fps)
#         return tp, fp, det_bboxes
#
#
# def bbox_overlaps(bboxes1, bboxes2, mode="iou", eps=1e-6, use_legacy_coordinate=False):
#     """Calculate the ious between each bbox of bboxes1 and bboxes2.
#     Args:
#         bboxes1 (ndarray): Shape (n, 4)
#         bboxes2 (ndarray): Shape (k, 4)
#         mode (str): IOU (intersection over union) or IOF (intersection
#             over foreground)
#         use_legacy_coordinate (bool): Whether to use coordinate system in
#             mmdet v1.x. which means width, height should be
#             calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
#             Note when function is used in `VOCDataset`, it should be
#             True to align with the official implementation
#             `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
#             Default: False.
#     Returns:
#         ious (ndarray): Shape (n, k)
#     """
#
#     assert mode in ["iou", "iof"]
#     if not use_legacy_coordinate:
#         extra_length = 0.0
#     else:
#         extra_length = 1.0
#     bboxes1 = bboxes1.astype(np.float32)
#     bboxes2 = bboxes2.astype(np.float32)
#     rows = bboxes1.shape[0]
#     cols = bboxes2.shape[0]
#     ious = np.zeros((rows, cols), dtype=np.float32)
#     if rows * cols == 0:
#         return ious
#     exchange = False
#     if bboxes1.shape[0] > bboxes2.shape[0]:
#         bboxes1, bboxes2 = bboxes2, bboxes1
#         ious = np.zeros((cols, rows), dtype=np.float32)
#         exchange = True
#     area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
#         bboxes1[:, 3] - bboxes1[:, 1] + extra_length
#     )
#     area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
#         bboxes2[:, 3] - bboxes2[:, 1] + extra_length
#     )
#     for i in range(bboxes1.shape[0]):
#         x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
#         y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
#         x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
#         y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
#         overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
#             y_end - y_start + extra_length, 0
#         )
#         if mode == "iou":
#             union = area1[i] + area2 - overlap
#         else:
#             union = area1[i] if not exchange else area2
#         union = np.maximum(union, eps)
#         ious[i, :] = overlap / union
#     if exchange:
#         ious = ious.T
#     return ious
#
#
# def get_cls_predictions(det_predictions, annotations, class_id):
#     """Get det predictions and gt information of a certain class.
#     Args:
#         det_predictions (list[list]): Same as `eval_map()`.
#         annotations (list[dict]): Same as `eval_map()`.
#         class_id (int): ID of a specific class.
#     Returns:
#         tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
#     """
#     cls_dets = [img_res[class_id] for img_res in det_predictions]
#     cls_gts = []
#     cls_gts_ignore = []
#     for ann in annotations:
#         gt_inds = ann["annotations"] == class_id
#         cls_gts.append(ann["bboxes"][gt_inds, :])
#
#         if ann.get("annotations_ignore", None) is not None:
#             ignore_inds = ann["annotations_ignore"] == class_id
#             cls_gts_ignore.append(ann["bboxes_ignore"][ignore_inds, :])
#         else:
#             cls_gts_ignore.append(np.empty((0, 4), dtype=np.float32))
#
#     return cls_dets, cls_gts, cls_gts_ignore
#
#
# def average_precision(recalls, precisions, mode="area"):
#     """Calculate average precision (for single or multiple scales).
#     Args:
#         recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
#         precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
#         mode (str): 'area' or '11points', 'area' means calculating the area
#             under precision-recall curve, '11points' means calculating
#             the average precision of recalls at [0, 0.1, ..., 1]
#     Returns:
#         float or ndarray: calculated average precision
#     """
#     no_scale = False
#     if recalls.ndim == 1:
#         no_scale = True
#         recalls = recalls[np.newaxis, :]
#         precisions = precisions[np.newaxis, :]
#     assert recalls.shape == precisions.shape and recalls.ndim == 2
#     num_scales = recalls.shape[0]
#     ap = np.zeros(num_scales, dtype=np.float32)
#     if mode == "area":
#         zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
#         ones = np.ones((num_scales, 1), dtype=recalls.dtype)
#         mrec = np.hstack((zeros, recalls, ones))
#         mpre = np.hstack((zeros, precisions, zeros))
#         for i in range(mpre.shape[1] - 1, 0, -1):
#             mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
#         for i in range(num_scales):
#             ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
#             ap[i] = np.sum((mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
#     elif mode == "11points":
#         for i in range(num_scales):
#             for thr in np.arange(0, 1 + 1e-3, 0.1):
#                 precs = precisions[i, recalls[i, :] >= thr]
#                 prec = precs.max() if precs.size > 0 else 0
#                 ap[i] += prec
#         ap /= 11
#     else:
#         raise ValueError('Unrecognized mode, only "area" and "11points" are supported')
#     if no_scale:
#         ap = ap[0]
#     return ap

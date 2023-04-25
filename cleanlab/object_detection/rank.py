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
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import copy


def get_label_quality_scores(
    labels: List[Dict[Any, Any]],
    predictions: List[np.ndarray],
    *,
    method: str = "subtype",
    probability_threshold: Optional[float] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Computes a label quality score for each image in the dataset.

    For object detection datasets, the label quality score for an image estimates how likely it has been correctly labeled.
    Lower scores indicate images whose annotation is more likely imperfect.
    Annotators may have mislabeled an image because they:
    - overlooked an object (missing annotated bounding box),
    - chose the wrong class label for an annotated box in the correct location,
    - imperfectly annotated the location/edges of a bounding box.
    Any of these annotation errors should lead to an image with a lower label quality score.

    Score is between 0 and 1.

        - 1 - clean label (given label is likely correct).
        - 0 - dirty label (given label is likely incorrect).

    A score is calculated for each of ``N`` images, with ``K`` total classes in the data.
    Each image has ``L`` annotated bounding boxes and ``M`` predicted bounding boxes.

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

    method:
        The method used to calculate label_quality_scores. Options:

        - ``subtype_lqs``: calculates image score as a composite score of the quality of badly located, swapped and missing bounding boxes.

    probability_threshold:
        Bounding boxes in ``predictions`` with ``pred_prob`` below the threshold are not considered for computing `label_quality_scores`.
        If you know what probability-threshold was used when producing predicted boxes from your trained object detector,
        please supply the value that was used here. If not provided, this value is inferred based on the smallest observed
        predicted probability for any of the predicted boxes.

    verbose : bool, default = True
      Set to ``False`` to suppress all print statements.

    Returns
    ---------
    label_quality_scores:
        Array of shape ``(N, )`` of scores between 0 and 1, one per image in the dataset.
        Lower scores indicate images that are more likely mislabeled.
    """

    _assert_valid_inputs(
        labels=labels,
        predictions=predictions,
        method=method,
        threshold=probability_threshold,
    )

    return _compute_label_quality_scores(
        labels=labels,
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

    issue_indices = np.argwhere(label_quality_scores <= threshold).flatten()
    issue_vals = label_quality_scores[issue_indices]
    sorted_idx = issue_vals.argsort()
    return issue_indices[sorted_idx]


def _compute_label_quality_scores(
    labels: List[Dict[Any, Any]],
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
        alpha = 0.91
        low_probability_threshold = 0.10
        high_probability_threshold = 0.6
        temperature = 0.98
        scores = _compute_subtype_lqs(
            labels,
            predictions,
            alpha=alpha,
            low_probability_threshold=low_probability_threshold,
            high_probability_threshold=high_probability_threshold,
            temperature=temperature,
        )

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

    max_allowed_box_prune = 0.97  # This is max allowed percent of prune for boxes below threshold before a warning is thrown.
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
            f"Pruning with threshold=={threshold} prunes {p_ann_pruned}% labels. Consider lowering the threshold.",
            UserWarning,
        )
    if verbose:
        print(f"Pruning {num_ann_to_zero} labels out of {total_ann} using threshold=={threshold}.")
    return predictions_copy


# Todo: make this more descriptive and assert better inputs
def _assert_valid_inputs(labels, predictions, method=None, threshold=None):
    """Asserts proper input format."""
    if len(labels) != len(predictions):
        raise ValueError(
            f"labels and predictions length needs to match. len(labels) == {len(labels)} while len(predictions) == {len(predictions)}."
        )
    # Typecheck labels and predictions
    if not isinstance(labels[0], dict):
        raise ValueError(
            f"Labels has to be a list of dicts. Instead it is list of {type(labels[0])}."
        )
    # check last column of predictions is probabilities ( < 1.)?
    if not isinstance(predictions[0], np.ndarray):
        raise ValueError(
            f"Prediction has to be a list of np.ndarray. Instead it is list of {type(predictions[0])}."
        )
    if not predictions[0][0].shape[1] == 5:
        raise ValueError(f"Prediction values have to be of format [_,_,_,_,pred_prob].")

    valid_methods = ["subtype"]
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
    label: Dict[Any, Any],
    prediction: np.ndarray,
    *,
    prediction_threshold: Optional[float] = None,
    given_label_overlay: bool = True,
    class_labels: Optional[Dict[Any, Any]] = None,
    figsize: Optional[Tuple[int, int]] = None,
):
    """Visualize bounding box labels (given labels) and model predictions for an image. The given labels
    are shown with red while the predictions are shown in blue.

    A single image is visualized with ``L`` annotated bounding boxes, ``M`` predicted bounding boxes and ``K`` total classes.


    Parameters
    ----------
    image_path:
        Full path to the image file.

    label:
        The given label for a single image in the format ``{'bboxes': np.ndarray((L,4)), 'labels': np.ndarray((L,))}`` where
        ``L`` is the number of bounding boxes for the `i`-th image and ``bboxes[j]`` is in the format ``[x,y,x,y]`` with given label ``labels[j]``.

    prediction:
        A prediction for a single image in the format ``np.ndarray((K,))`` and ``prediction[k]`` is of shape ``np.ndarray(N,5)``
        where ``M`` is the number of bounding boxes for class ``k`` and the five columns correspond to ``[x,y,x,y,pred_prob]`` returned
        by the model.

    prediction_threshold:
        Minimum `pred_probs` value of a bounding box output by the model. All bounding boxes with `pred_probs` below this threshold are
        omitted from the visualization.

    given_label_overlay: bool
        If true, a single image with overlaid given labels and predictions is shown. If false, two images side
        by side are shown instead with the left image being the prediction and right being the given label.

    class_labels:
        Optional dictionary mapping one-hot-encoded class labels back to their original class names in the format ``{"one-hot-label": "original-class-name"}``.

    figsize:
        Optional figuresize for plotting the visualizations. Corresponds to ``matplotlib.figure.figsize``.
    """

    prediction_type = _get_prediction_type(prediction)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError(
            "This functionality requires matplotlib. Install it via: `pip install matplotlib`"
        )

    # Create figure and axes
    image = plt.imread(image_path)
    pbbox, plabels, pred_probs = _separate_prediction(prediction, prediction_type=prediction_type)
    abbox, alabels = _separate_label(label)

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

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError(
            "This functionality requires matplotlib. Install it via: `pip install matplotlib`"
        )

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

    try:
        from matplotlib.patches import Rectangle
    except Exception as e:
        raise ImportError(
            "This functionality requires matplotlib. Install it via: `pip install matplotlib`"
        )

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


def _separate_label(label):
    """Seperates labels into bounding box and class label lists."""
    bboxes = label["bboxes"]
    labels = label["labels"]
    return bboxes, labels


# TODO: make object detection work for all predicted probabilities
def _separate_prediction_all_preds(prediction):
    pred_bboxes, pred_labels, det_probs = prediction
    return pred_bboxes, pred_labels, det_probs


def _separate_prediction_single_box(prediction):
    """Seperates predictions into class labels, bounding boxes and pred_prob lists"""
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


def _separate_prediction(prediction, prediction_type="single_pred"):
    """Returns bbox, label and pred_prob values for prediction."""

    if prediction_type == "all_pred":
        boxes, labels, pred_probs = _separate_prediction_all_preds(prediction)
    else:
        boxes, labels, pred_probs = _separate_prediction_single_box(prediction)
    return boxes, labels, pred_probs


def _bbox_xyxy_to_xywh(bbox):
    """Converts bounding box coodrinate types from x1y1,x2y2 to x,y,w,h"""
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        return [x1, y1, w, h]
    else:
        print("Wrong bbox shape", len(bbox))
        return None


# ============= Simple label subtype algorithm ============
def _mod_coordinates(x):
    """Takes is a list of xyxy coordinates and returns them in dictionary format."""

    wd = {"x1": x[0], "y1": x[1], "x2": x[2], "y2": x[3]}
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
    euc_factor = 100  # this is a hyperparameter

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


def _get_min_possible_similarity(predictions, labels, alpha):
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
    *,
    alpha: float,
    label: Optional[dict] = None,
    prediction: Optional[np.ndarray] = None,
    pred_labels: Optional[np.ndarray] = None,
    pred_label_probs: Optional[np.ndarray] = None,
    pred_bboxes=None,
    lab_labels: Optional[np.ndarray] = None,
    lab_bboxes=None,
    similarity_matrix=None,
    min_possible_similarity: Optional[float] = None,
):
    if lab_labels is None or lab_bboxes is None:
        if label is None:
            raise ValueError(f"Pass in either one of label or label labels. Both can not be None.")
        lab_bboxes, lab_labels = _separate_label(label)

    if pred_labels is None or pred_label_probs is None or pred_bboxes is None:
        if prediction is None:
            raise ValueError(
                f"Pass in either one of prediction or prediction labels and prediction probabilities. Both can not be None."
            )
        pred_bboxes, pred_labels, pred_label_probs = _separate_prediction(prediction)

    if similarity_matrix is None:
        iou_matrix = _get_overlap_matrix(lab_bboxes, pred_bboxes)
        dist_matrix = 1 - _get_dist_matrix(lab_bboxes, pred_bboxes)
        similarity_matrix = iou_matrix * alpha + (1 - alpha) * (1 - dist_matrix)

    if min_possible_similarity is None:
        min_possible_similarity = (
            1.0
            if 0 in similarity_matrix.shape
            else np.min(similarity_matrix[np.nonzero(similarity_matrix)])
        )

    return (
        pred_labels,
        pred_label_probs,
        pred_bboxes,
        lab_labels,
        lab_bboxes,
        similarity_matrix,
        min_possible_similarity,
    )


def _get_valid_inputs_for_compute_scores(
    labels,
    predictions,
    alpha,
):
    min_possible_similarity = _get_min_possible_similarity(predictions, labels, alpha)

    lab_labels_list = []
    lab_bboxes_list = []
    pred_labels_list = []
    pred_label_probs_list = []
    pred_bboxes_list = []
    similarity_matrix_list = []

    for prediction, label in zip(predictions, labels):
        (
            pred_labels,
            pred_label_probs,
            pred_bboxes,
            lab_labels,
            lab_bboxes,
            similarity_matrix,
            _,
        ) = _get_valid_inputs_for_compute_scores_per_image(
            alpha=alpha,
            label=label,
            prediction=prediction,
            min_possible_similarity=min_possible_similarity,
        )
        lab_labels_list.append(lab_labels)
        lab_bboxes_list.append(lab_bboxes)
        pred_labels_list.append(pred_labels)
        pred_label_probs_list.append(pred_label_probs)
        pred_bboxes_list.append(pred_bboxes)
        similarity_matrix_list.append(similarity_matrix)
    return (
        pred_labels_list,
        pred_label_probs_list,
        pred_bboxes_list,
        lab_labels_list,
        lab_bboxes_list,
        similarity_matrix_list,
        min_possible_similarity,
    )


def _get_valid_score(scores_arr, temperature=0.99):
    """Given scores array, returns valid score (softmin) or 1. Checks validity of score."""
    scores_arr = np.array(scores_arr)
    scores_arr = scores_arr[~np.isnan(scores_arr)]
    if len(scores_arr) > 0:
        valid_score = _softmin1D(scores_arr, temperature=temperature)
    else:
        valid_score = 1.0
    return valid_score


def _pool_box_scores_per_image(box_scores, temperature=0.99) -> np.ndarray:
    image_scores = np.empty(
        shape=[
            len(box_scores),
        ]
    )
    for idx, box_score in enumerate(box_scores):
        image_score = _get_valid_score(box_score, temperature=temperature)
        image_scores[idx] = image_score
    return image_scores


def _compute_overlooked_box_scores_for_image(
    alpha: float,
    high_probability_threshold: float,
    label: Optional[dict] = None,
    prediction: Optional[np.ndarray] = None,
    pred_labels: Optional[np.ndarray] = None,
    pred_label_probs: Optional[np.ndarray] = None,
    pred_bboxes: Optional[list] = None,
    lab_labels: Optional[np.ndarray] = None,
    lab_bboxes: Optional[list] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    min_possible_similarity: Optional[float] = None,
):
    """This method returns one score per predicted box (above threshold) in an image. Score from 0 to 1 ranking how overlooked the box is."""

    (
        pred_labels,
        pred_label_probs,
        pred_bboxes,
        lab_labels,
        lab_bboxes,
        similarity_matrix,
        min_possible_similarity,
    ) = _get_valid_inputs_for_compute_scores_per_image(
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

    scores_overlooked = np.empty(
        shape=[
            len(pred_labels),
        ]
    )  # same length as num of predicted boxes

    for iid, k in enumerate(pred_labels):
        if pred_label_probs[iid] < high_probability_threshold:
            scores_overlooked[iid] = np.nan
            continue

        k_similarity = similarity_matrix[lab_labels == k, iid]
        if len(k_similarity) == 0:  # if there is no annotated box
            score = min_possible_similarity * (1 - pred_label_probs[iid])
        else:
            closest_annotated_box = np.argmax(k_similarity)
            score = k_similarity[closest_annotated_box]
        scores_overlooked[iid] = score

    return scores_overlooked


def _compute_overlooked_box_scores(
    alpha: float,
    high_probability_threshold: float,
    labels: Optional[dict] = None,
    predictions: Optional[np.ndarray] = None,
    pred_labels_list: Optional[list] = None,
    pred_label_probs_list: Optional[list] = None,
    pred_bboxes_list: Optional[list] = None,
    lab_labels_list: Optional[list] = None,
    lab_bboxes_list: Optional[list] = None,
    similarity_matrix_list: Optional[list] = None,
    min_possible_similarity: Optional[float] = None,
):
    if (
        pred_labels_list is None
        or pred_label_probs_list is None
        or lab_labels_list is None
        or similarity_matrix_list is None
        or min_possible_similarity is None
    ):
        (
            pred_labels_list,
            pred_label_probs_list,
            pred_bboxes_list,
            lab_labels_list,
            lab_bboxes_list,
            similarity_matrix_list,
            min_possible_similarity,
        ) = _get_valid_inputs_for_compute_scores(labels, predictions, alpha)

    scores_overlooked = []
    for (
        pred_labels,
        pred_label_probs,
        pred_bboxes,
        lab_labels,
        lab_bboxes,
        similarity_matrix,
    ) in zip(
        pred_labels_list,
        pred_label_probs_list,
        pred_bboxes_list,
        lab_labels_list,
        lab_bboxes_list,
        similarity_matrix_list,
    ):
        scores_overlooked_per_box = _compute_overlooked_box_scores_for_image(
            alpha=alpha,
            high_probability_threshold=high_probability_threshold,
            pred_labels=pred_labels,
            pred_label_probs=pred_label_probs,
            pred_bboxes=pred_bboxes,
            lab_labels=lab_labels,
            lab_bboxes=lab_bboxes,
            similarity_matrix=similarity_matrix,
            min_possible_similarity=min_possible_similarity,
        )
        scores_overlooked.append(scores_overlooked_per_box)
    return scores_overlooked


def _compute_badloc_box_scores_for_image(
    alpha: float,
    low_probability_threshold: float,
    label: Optional[dict] = None,
    prediction: Optional[np.ndarray] = None,
    pred_labels: Optional[np.ndarray] = None,
    pred_label_probs: Optional[np.ndarray] = None,
    pred_bboxes: Optional[list] = None,
    lab_labels: Optional[np.ndarray] = None,
    lab_bboxes: Optional[list] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    min_possible_similarity: Optional[float] = None,
):
    """This method returns one score per labeled box in an image. Score from 0 to 1 ranking how badly located the box is."""

    (
        pred_labels,
        pred_label_probs,
        pred_bboxes,
        lab_labels,
        lab_bboxes,
        similarity_matrix,
        min_possible_similarity,
    ) = _get_valid_inputs_for_compute_scores_per_image(
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

    scores_badloc = np.empty(
        shape=[
            len(lab_labels),
        ]
    )  # same length as number of labeled boxes
    for iid, k in enumerate(lab_labels):  # for every annotated box
        k_similarity = similarity_matrix[iid, pred_labels == k]
        k_pred = pred_label_probs[pred_labels == k]

        if len(k_pred) == 0:  # there are no predicted boxes of class k
            scores_badloc[iid] = min_possible_similarity
            continue

        idx_at_least_low_probability_threshold = k_pred > low_probability_threshold
        k_similarity = k_similarity[idx_at_least_low_probability_threshold]
        k_pred = k_pred[idx_at_least_low_probability_threshold]
        assert len(k_pred) == len(k_similarity)
        if len(k_pred) == 0:
            scores_badloc[iid] = min_possible_similarity
        else:
            scores_badloc[iid] = np.max(k_similarity)
    return scores_badloc


def _compute_badloc_box_scores(
    alpha: float,
    low_probability_threshold: float,
    labels: Optional[dict] = None,
    predictions: Optional[np.ndarray] = None,
    pred_labels_list: Optional[list] = None,
    pred_label_probs_list: Optional[list] = None,
    pred_bboxes_list: Optional[list] = None,
    lab_labels_list: Optional[list] = None,
    lab_bboxes_list: Optional[list] = None,
    similarity_matrix_list: Optional[list] = None,
    min_possible_similarity: Optional[float] = None,
):
    if (
        pred_labels_list is None
        or pred_label_probs_list is None
        or lab_labels_list is None
        or similarity_matrix_list is None
        or min_possible_similarity is None
    ):
        (
            pred_labels_list,
            pred_label_probs_list,
            lab_labels_list,
            similarity_matrix_list,
            min_possible_similarity,
        ) = _get_valid_inputs_for_compute_scores(labels, predictions, alpha)

    scores_badloc = []
    for (
        pred_labels,
        pred_label_probs,
        pred_bboxes,
        lab_labels,
        lab_bboxes,
        similarity_matrix,
    ) in zip(
        pred_labels_list,
        pred_label_probs_list,
        pred_bboxes_list,
        lab_labels_list,
        lab_bboxes_list,
        similarity_matrix_list,
    ):
        scores_badloc_per_box = _compute_badloc_box_scores_for_image(
            alpha=alpha,
            low_probability_threshold=low_probability_threshold,
            pred_labels=pred_labels,
            pred_label_probs=pred_label_probs,
            pred_bboxes=pred_bboxes,
            lab_labels=lab_labels,
            lab_bboxes=lab_bboxes,
            similarity_matrix=similarity_matrix,
            min_possible_similarity=min_possible_similarity,
        )
        scores_badloc.append(scores_badloc_per_box)
    return scores_badloc


def _compute_swap_box_scores_for_image(
    alpha: float,
    high_probability_threshold: float,
    label: Optional[dict] = None,
    prediction: Optional[np.ndarray] = None,
    pred_labels: Optional[np.ndarray] = None,
    pred_label_probs: Optional[np.ndarray] = None,
    pred_bboxes: Optional[list] = None,
    lab_labels: Optional[np.ndarray] = None,
    lab_bboxes: Optional[list] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    min_possible_similarity: Optional[float] = None,
):
    """This method returns one score per labeled box in an image. Score from 0 to 1 ranking how likeley swapped the box is."""

    (
        pred_labels,
        pred_label_probs,
        pred_bboxes,
        lab_labels,
        lab_bboxes,
        similarity_matrix,
        min_possible_similarity,
    ) = _get_valid_inputs_for_compute_scores_per_image(
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

    scores_swap = np.empty(
        shape=[
            len(lab_labels),
        ]
    )  # same length as number of labeled boxes
    for iid, k in enumerate(lab_labels):
        not_k_idx = pred_labels != k

        if len(not_k_idx) == 0:
            scores_swap[iid] = 1.0
            continue

        not_k_similarity = similarity_matrix[iid, not_k_idx]
        not_k_pred = pred_label_probs[not_k_idx]

        idx_at_least_high_probability_threshold = not_k_pred > high_probability_threshold
        if len(idx_at_least_high_probability_threshold) == 0:
            scores_swap[iid] = 1.0
            continue

        not_k_similarity = not_k_similarity[idx_at_least_high_probability_threshold]
        if len(not_k_similarity) == 0:  # if there is no annotated box
            scores_swap[iid] = 1.0
        else:
            closest_predicted_box = np.argmax(not_k_similarity)
            score = np.max([min_possible_similarity, 1 - not_k_similarity[closest_predicted_box]])
            scores_swap[iid] = score
    return scores_swap


def _compute_swap_box_scores(
    alpha: float,
    high_probability_threshold: float,
    labels: Optional[dict] = None,
    predictions: Optional[np.ndarray] = None,
    pred_labels_list: Optional[list] = None,
    pred_label_probs_list: Optional[list] = None,
    pred_bboxes_list: Optional[list] = None,
    lab_labels_list: Optional[list] = None,
    lab_bboxes_list: Optional[list] = None,
    similarity_matrix_list: Optional[list] = None,
    min_possible_similarity: Optional[float] = None,
):
    if (
        pred_labels_list is None
        or pred_label_probs_list is None
        or lab_labels_list is None
        or similarity_matrix_list is None
        or min_possible_similarity is None
    ):
        (
            pred_labels_list,
            pred_label_probs_list,
            lab_labels_list,
            similarity_matrix_list,
            min_possible_similarity,
        ) = _get_valid_inputs_for_compute_scores(labels, predictions, alpha)

    scores_swap = []
    for (
        pred_labels,
        pred_label_probs,
        pred_bboxes,
        lab_labels,
        lab_bboxes,
        similarity_matrix,
    ) in zip(
        pred_labels_list,
        pred_label_probs_list,
        pred_bboxes_list,
        lab_labels_list,
        lab_bboxes_list,
        similarity_matrix_list,
    ):
        scores_swap_per_box = _compute_swap_box_scores_for_image(
            alpha=alpha,
            high_probability_threshold=high_probability_threshold,
            pred_labels=pred_labels,
            pred_label_probs=pred_label_probs,
            pred_bboxes=pred_bboxes,
            lab_labels=lab_labels,
            lab_bboxes=lab_bboxes,
            similarity_matrix=similarity_matrix,
            min_possible_similarity=min_possible_similarity,
        )
        scores_swap.append(scores_swap_per_box)
    return scores_swap


def _compute_subtype_lqs(
    labels,
    predictions,
    *,
    alpha,
    low_probability_threshold,
    high_probability_threshold,
    temperature,
):
    """
    Returns a label quality score for each image.
    Score is between 0 and 1.

    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels:
        A list of `N` dictionaries for `N` images such that `labels[i]` contains the given labels for the `i`-th image in the format
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
        The lowest prediction threshold allowed when considering predicted boxes to identify badly located label boxes.

    high_probability_threshold:
        The high probability threshold for considering predicted boxes to identify overlooked and swapped label boxes.

    temperature:
        Temperature of the softmin function where a lower score suggests softmin acts closer to min.

    Returns
    ---------
    label_quality_scores:
        Array of shape ``(N, )`` of scores between 0 and 1, one per image in the dataset.
        Lower scores indicate images are more likely to contain an incorrect label.
    """
    (
        pred_labels_list,
        pred_label_probs_list,
        pred_bboxes_list,
        lab_labels_list,
        lab_bboxes_list,
        similarity_matrix_list,
        min_possible_similarity,
    ) = _get_valid_inputs_for_compute_scores(labels, predictions, alpha)

    overlooked_scores_per_box = _compute_overlooked_box_scores(
        alpha=alpha,
        high_probability_threshold=high_probability_threshold,
        pred_labels_list=pred_labels_list,
        pred_label_probs_list=pred_label_probs_list,
        pred_bboxes_list=pred_bboxes_list,
        lab_labels_list=lab_labels_list,
        lab_bboxes_list=lab_bboxes_list,
        similarity_matrix_list=similarity_matrix_list,
        min_possible_similarity=min_possible_similarity,
    )
    overlooked_score_per_image = _pool_box_scores_per_image(overlooked_scores_per_box, temperature)

    badloc_scores_per_box = _compute_badloc_box_scores(
        alpha=alpha,
        low_probability_threshold=low_probability_threshold,
        pred_labels_list=pred_labels_list,
        pred_label_probs_list=pred_label_probs_list,
        pred_bboxes_list=pred_bboxes_list,
        lab_labels_list=lab_labels_list,
        lab_bboxes_list=lab_bboxes_list,
        similarity_matrix_list=similarity_matrix_list,
        min_possible_similarity=min_possible_similarity,
    )
    badloc_score_per_image = _pool_box_scores_per_image(badloc_scores_per_box, temperature)

    swap_scores_per_box = _compute_swap_box_scores(
        alpha=alpha,
        high_probability_threshold=high_probability_threshold,
        pred_labels_list=pred_labels_list,
        pred_label_probs_list=pred_label_probs_list,
        pred_bboxes_list=pred_bboxes_list,
        lab_labels_list=lab_labels_list,
        lab_bboxes_list=lab_bboxes_list,
        similarity_matrix_list=similarity_matrix_list,
        min_possible_similarity=min_possible_similarity,
    )
    swap_score_per_image = _pool_box_scores_per_image(swap_scores_per_box, temperature)

    scores = (
        0.6 * overlooked_score_per_image + 0.2 * badloc_score_per_image + 0.2 * swap_score_per_image
    )
    return scores

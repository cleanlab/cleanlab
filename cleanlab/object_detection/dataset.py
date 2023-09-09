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

"""
Methods to display examples and their label issues in an object detection dataset.
Here each image can have multiple objects, each with its own bounding box and class label.
"""
from typing import Optional, Any, Dict, Tuple, Union, List, TYPE_CHECKING, TypeVar

import numpy as np
import collections

from cleanlab.internal.constants import (
    MAX_CLASS_TO_SHOW,
    ALPHA,
)
from cleanlab.object_detection.rank import (
    _get_valid_inputs_for_compute_scores,
    _separate_prediction,
    _separate_label,
    _get_prediction_type,
    AuxiliaryTypesDict,
)

from cleanlab.internal.object_detection_utils import bbox_xyxy_to_xywh

if TYPE_CHECKING:
    from PIL.Image import Image as Image  # pragma: no cover
else:
    Image = TypeVar("Image")


def get_object_count(
    labels=None,
    predictions=None,
    *,
    auxiliary_inputs=None,
) -> Tuple[List, List]:
    """Return the number of annotated and predicted objects in the dataset.

    Parameters
    ----------
    labels :
        Annotated boxes and class labels in the original dataset, which may contain some errors.
        This is a list of ``N`` dictionaries such that ``labels[i]`` contains the given labels for the `i`-th image in the following format:
        ``{'bboxes': np.ndarray((L,4)), 'labels': np.ndarray((L,)), 'image_name': str}`` where ``L`` is the number of annotated bounding boxes
        for the `i`-th image and ``bboxes[l]`` is a bounding box of coordinates in ``[x1,y1,x2,y2]`` format with given class label ``labels[j]``.
        ``image_name`` is an optional part of the labels that can be used to later refer to specific images.

       For more information on proper labels formatting, check out the `MMDetection library <https://mmdetection.readthedocs.io/en/dev-3.x/advanced_guides/customize_dataset.html>`_.

    predictions :
        Predictions output by a trained object detection model.
        For the most accurate results, predictions should be out-of-sample to avoid overfitting, eg. obtained via :ref:`cross-validation <pred_probs_cross_val>`.
        This is a list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model prediction for the `i`-th image.
        For each possible class ``k`` in 0, 1, ..., K-1: ``predictions[i][k]`` is a ``np.ndarray`` of shape ``(M,5)``,
        where ``M`` is the number of predicted bounding boxes for class ``k``. Here the five columns correspond to ``[x1,y1,x2,y2,pred_prob]``,
        where ``[x1,y1,x2,y2]`` are coordinates of the bounding box predicted by the model
        and ``pred_prob`` is the model's confidence in the predicted class label for this bounding box.

        Note: Here, ``[x1,y1]`` corresponds to the coordinates of the bottom-left corner of the bounding box, while ``[x2,y2]`` corresponds to the coordinates of the top-right corner of the bounding box. The last column, pred_prob, represents the predicted probability that the bounding box contains an object of the class k.

        For more information see the `MMDetection package <https://github.com/open-mmlab/mmdetection>`_ for an example object detection library that outputs predictions in the correct format.

    auxiliary_inputs : optional
        Auxiliary inputs to be used in the computation of counts.
        The `auxiliary_inputs` can be computed using :py:func:`rank._get_valid_inputs_for_compute_scores <cleanlab.object_detection.rank._get_valid_inputs_for_compute_scores>`.
        It is internally computed from the given `labels` and `predictions`.

    Returns
    -------
    object_counts: Tuple[np.ndarray, label_issues : np.ndarray]
        A tuple containing two lists. The first is an array of shape ``(N,)`` containing the number of annotated objects for each image in the dataset.
        The second is an array of shape ``(N,)`` containing the number of predicted objects for each image in the dataset.
    """
    if auxiliary_inputs is None:
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(ALPHA, labels, predictions)
    return (
        [len(sample["lab_bboxes"]) for sample in auxiliary_inputs],
        [len(sample["pred_bboxes"]) for sample in auxiliary_inputs],
    )


def get_bbox_sizes(
    labels=None,
    predictions=None,
    *,
    auxiliary_inputs=None,
) -> Tuple[Dict[Any, List], Dict[Any, List]]:
    """Return the sizes of annotated and predicted bounding boxes in the dataset for each class.

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

    auxiliary_inputs : optional
        Auxiliary inputs to be used in the computation of counts.
        The `auxiliary_inputs` can be computed using :py:func:`rank._get_valid_inputs_for_compute_scores <cleanlab.object_detection.rank._get_valid_inputs_for_compute_scores>`.
        It is internally computed from the given `labels` and `predictions`.

    Returns
    -------
    bbox_sizes: Tuple[Dict[Any, List], Dict[Any, List]]
        A tuple containing two dictionaries. Each maps each class label to a list of the sizes of annotated bounding boxes for that class in the label and prediction datasets, respectively.
    """
    if auxiliary_inputs is None:
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(ALPHA, labels, predictions)

    label_area, pred_area = collections.defaultdict(list), collections.defaultdict(list)
    for sample in auxiliary_inputs:
        _get_bbox_area(sample["lab_labels"], sample["lab_bboxes"], label_area)
        _get_bbox_area(sample["pred_labels"], sample["pred_bboxes"], pred_area)

    return label_area, pred_area


def get_class_distribution(
    labels=None,
    predictions=None,
    *,
    auxiliary_inputs=None,
) -> Tuple[Dict[Any, float], Dict[Any, float]]:
    """Return the distribution of classes in the dataset.

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

    auxiliary_inputs : optional
        Auxiliary inputs to be used in the computation of counts.
        The `auxiliary_inputs` can be computed using :py:func:`rank._get_valid_inputs_for_compute_scores <cleanlab.object_detection.rank._get_valid_inputs_for_compute_scores>`.
        It is internally computed from the given `labels` and `predictions`.

    Returns
    -------
    class_distribution: Tuple[Dict[Any, float], Dict[Any, float]]
        A tuple containing two dictionaries. The first is a dictionary mapping each class label to its frequency in the dataset.
        The second is a dictionary mapping each class label to its frequency in the predictions.
    """
    if auxiliary_inputs is None:
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(ALPHA, labels, predictions)

    lab_freq, pred_freq = collections.defaultdict(int), collections.defaultdict(int)
    for sample in auxiliary_inputs:
        for cl in sample["lab_labels"]:
            lab_freq[cl] += 1
        for cl in sample["pred_labels"]:
            pred_freq[cl] += 1

    lab_total, pred_total = sum(lab_freq.values()), sum(pred_freq.values())
    return (
        {k: round(v / lab_total, 2) for k, v in lab_freq.items()},
        {k: round(v / pred_total, 2) for k, v in pred_freq.items()},
    )


def visualize(
    image: Union[str, Image],
    *,
    label: Optional[Dict[str, Any]] = None,
    prediction: Optional[np.ndarray] = None,
    prediction_threshold: Optional[float] = None,
    overlay: bool = True,
    class_names: Optional[Dict[Any, Any]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Display the annotated bounding boxes (given labels) and predicted bounding boxes (model predictions) for a particular image.
    Given labels are shown in red, model predictions in blue.


    Parameters
    ----------
    image:
        Image object loaded into memory or full path to the image file. If path is provided, image is loaded into memory.

    label:
        The given label for a single image in the format ``{'bboxes': np.ndarray((L,4)), 'labels': np.ndarray((L,))}`` where
        ``L`` is the number of bounding boxes for the `i`-th image and ``bboxes[j]`` is in the format ``[x1,y1,x2,y2]`` with given label ``labels[j]``.

        Note: Here, ``[x1,y1]`` corresponds to the coordinates of the bottom-left corner of the bounding box, while ``[x2,y2]`` corresponds to the coordinates of the top-right corner of the bounding box. The last column, pred_prob, represents the predicted probability that the bounding box contains an object of the class k.

    prediction:
        A prediction for a single image in the format ``np.ndarray((K,))`` and ``prediction[k]`` is of shape ``np.ndarray(N,5)``
        where ``M`` is the number of predicted bounding boxes for class ``k`` and the five columns correspond to ``[x,y,x,y,pred_prob]`` where
        ``[x,y,x,y]`` are the bounding box coordinates predicted by the model and ``pred_prob`` is the model's confidence in ``predictions[i]``.

    prediction_threshold:
        All model-predicted bounding boxes with confidence (`pred_prob`)
        below this threshold are omitted from the visualization.

    overlay: bool
        If True, display a single image with given labels and predictions overlaid.
        If False, display two images (side by side) with the left image showing  the model predictions and the right image showing the given label.

    class_names:
        Optional dictionary mapping one-hot-encoded class labels back to their original class names in the format ``{"integer-label": "original-class-name"}``.

    save_path:
        Path to save figure at. If a path is provided, the figure is saved. To save in a specific image format, add desired file extension to the end of `save_path`. Allowed file extensions are: 'png', 'pdf', 'ps', 'eps', and 'svg'.

    figsize:
        Optional figure size for plotting the image.
        Corresponds to ``matplotlib.figure.figsize``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "This functionality requires matplotlib. Install it via: `pip install matplotlib`"
        )

    # Create figure and axes
    if isinstance(image, str):
        image = plt.imread(image)

    if prediction is not None:
        prediction_type = _get_prediction_type(prediction)
        pbbox, plabels, pred_probs = _separate_prediction(
            prediction, prediction_type=prediction_type
        )

        if prediction_threshold is not None:
            keep_idx = np.where(pred_probs > prediction_threshold)
            pbbox = pbbox[keep_idx]
            plabels = plabels[keep_idx]

    if label is not None:
        abbox, alabels = _separate_label(label)

    if overlay:
        figsize = (8, 5) if figsize is None else figsize
        fig, ax = plt.subplots(frameon=False, figsize=figsize)
        plt.axis("off")
        ax.imshow(image)
        if label is not None:
            fig, ax = _draw_boxes(
                fig, ax, abbox, alabels, edgecolor="r", linestyle="-", linewidth=1
            )
        if prediction is not None:
            _, _ = _draw_boxes(fig, ax, pbbox, plabels, edgecolor="b", linestyle="-.", linewidth=1)
    else:
        figsize = (14, 10) if figsize is None else figsize
        fig, axes = plt.subplots(nrows=1, ncols=2, frameon=False, figsize=figsize)
        axes[0].axis("off")
        axes[0].imshow(image)
        axes[1].axis("off")
        axes[1].imshow(image)

        if label is not None:
            fig, ax = _draw_boxes(
                fig, axes[0], abbox, alabels, edgecolor="r", linestyle="-", linewidth=1
            )
        if prediction is not None:
            _, _ = _draw_boxes(
                fig, axes[1], pbbox, plabels, edgecolor="b", linestyle="-.", linewidth=1
            )
    bbox_extra_artists = None
    if label or prediction is not None:
        legend, plt = _plot_legend(class_names, label, prediction)
        bbox_extra_artists = (legend,)

    if save_path:
        allowed_image_formats = set(["png", "pdf", "ps", "eps", "svg"])
        image_format: Optional[str] = None
        if save_path.split(".")[-1] in allowed_image_formats and "." in save_path:
            image_format = save_path.split(".")[-1]
        plt.savefig(
            save_path,
            format=image_format,
            bbox_extra_artists=bbox_extra_artists,
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.5,
        )
    plt.show()


def _get_bbox_area(labels, boxes, class_area_dict) -> None:
    """Helper function to compute the area of bounding boxes for each class."""
    for cl, bbox in zip(labels, boxes):
        class_area_dict[cl].append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _plot_legend(class_names, label, prediction):
    colors = ["black"]
    colors.extend(["red"] if label is not None else [])
    colors.extend(["blue"] if prediction is not None else [])

    markers = [None]
    markers.extend(["s"] if label is not None else [])
    markers.extend(["s"] if prediction is not None else [])

    labels = [r"$\bf{Legend}$"]
    labels.extend(["given label"] if label is not None else [])
    labels.extend(["predicted label"] if prediction is not None else [])

    if class_names:
        colors += ["black"] + ["black"] * min(len(class_names), MAX_CLASS_TO_SHOW)
        markers += [None] + [f"${class_key}$" for class_key in class_names.keys()]
        labels += [r"$\bf{classes}$"] + list(class_names.values())

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "This functionality requires matplotlib. Install it via: `pip install matplotlib`"
        )

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f(marker, color) for marker, color in zip(markers, colors)]
    legend = plt.legend(
        handles, labels, bbox_to_anchor=(1.04, 0.05), loc="lower left", borderaxespad=0
    )

    return legend, plt


def _draw_labels(ax, rect, label, edgecolor):
    """Helper function to draw labels on an axis."""

    rx, ry = rect.get_xy()
    c_xleft = rx + 10
    c_xright = rx + rect.get_width() - 10
    c_ytop = ry + 12

    if edgecolor == "r":
        cx, cy = c_xleft, c_ytop
    else:  # edgecolor == b
        cx, cy = c_xright, c_ytop

    l = ax.annotate(
        label, (cx, cy), fontsize=8, fontweight="bold", color="white", ha="center", va="center"
    )
    l.set_bbox(dict(facecolor=edgecolor, alpha=0.35, edgecolor=edgecolor, pad=2))
    return ax


def _draw_boxes(fig, ax, bboxes, labels, edgecolor="g", linestyle="-", linewidth=3):
    """Helper function to draw bboxes and labels on an axis."""
    bboxes = [bbox_xyxy_to_xywh(box) for box in bboxes]

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

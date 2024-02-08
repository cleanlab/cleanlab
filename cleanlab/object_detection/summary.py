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
from multiprocessing import Pool
from typing import Optional, Any, Dict, Tuple, Union, List, TYPE_CHECKING, TypeVar, DefaultDict

import numpy as np
import collections

from cleanlab.internal.constants import (
    MAX_CLASS_TO_SHOW,
    ALPHA,
    EPSILON,
    TINY_VALUE,
)
from cleanlab.object_detection.filter import (
    _filter_by_class,
    _calculate_true_positives_false_positives,
)
from cleanlab.object_detection.rank import (
    _get_valid_inputs_for_compute_scores,
    _separate_prediction,
    _separate_label,
    _get_prediction_type,
)

from cleanlab.internal.object_detection_utils import bbox_xyxy_to_xywh

if TYPE_CHECKING:
    from PIL.Image import Image as Image  # pragma: no cover
else:
    Image = TypeVar("Image")


def object_counts_per_image(
    labels=None,
    predictions=None,
    *,
    auxiliary_inputs=None,
) -> Tuple[List, List]:
    """Return the number of annotated and predicted objects for each image in the dataset.

    This method can help you discover images with abnormally many/few object annotations.

    Parameters
    ----------
    labels :
        Annotated boxes and class labels in the original dataset, which may contain some errors.
        This is a list of ``N`` dictionaries such that ``labels[i]`` contains the given labels for the `i`-th image in the following format:
        ``{'bboxes': np.ndarray((L,4)), 'labels': np.ndarray((L,)), 'image_name': str}`` where ``L`` is the number of annotated bounding boxes
        for the `i`-th image and ``bboxes[l]`` is a bounding box of coordinates in ``[x1,y1,x2,y2]`` format with given class label ``labels[j]``.
        ``image_name`` is an optional part of the labels that can be used to later refer to specific images.

        Note: Here, ``(x1,y1)`` corresponds to the top-left and ``(x2,y2)`` corresponds to the bottom-right corner of the bounding box with respect to the image matrix [e.g. `XYXY in Keras <https://keras.io/api/keras_cv/bounding_box/formats/>`, `Detectron 2 <https://detectron2.readthedocs.io/en/latest/modules/utils.html#detectron2.utils.visualizer.Visualizer.draw_box>`].

        For more information on proper labels formatting, check out the `MMDetection library <https://mmdetection.readthedocs.io/en/dev-3.x/advanced_guides/customize_dataset.html>`_.

    predictions :
        Predictions output by a trained object detection model.
        For the most accurate results, predictions should be out-of-sample to avoid overfitting, eg. obtained via :ref:`cross-validation <pred_probs_cross_val>`.
        This is a list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model prediction for the `i`-th image.
        For each possible class ``k`` in 0, 1, ..., K-1: ``predictions[i][k]`` is a ``np.ndarray`` of shape ``(M,5)``,
        where ``M`` is the number of predicted bounding boxes for class ``k``. Here the five columns correspond to ``[x1,y1,x2,y2,pred_prob]``,
        where ``[x1,y1,x2,y2]`` are coordinates of the bounding box predicted by the model
        and ``pred_prob`` is the model's confidence in the predicted class label for this bounding box.

        Note: Here, ``(x1,y1)`` corresponds to the top-left and ``(x2,y2)`` corresponds to the bottom-right corner of the bounding box with respect to the image matrix [e.g. `XYXY in Keras <https://keras.io/api/keras_cv/bounding_box/formats/>`, `Detectron 2 <https://detectron2.readthedocs.io/en/latest/modules/utils.html#detectron2.utils.visualizer.Visualizer.draw_box>`]. The last column, pred_prob, represents the predicted probability that the bounding box contains an object of the class k.

        For more information see the `MMDetection package <https://github.com/open-mmlab/mmdetection>`_ for an example object detection library that outputs predictions in the correct format.

    auxiliary_inputs: optional
        Auxiliary inputs to be used in the computation of counts.
        The `auxiliary_inputs` can be computed using :py:func:`rank._get_valid_inputs_for_compute_scores <cleanlab.object_detection.rank._get_valid_inputs_for_compute_scores>`.
        It is internally computed from the given `labels` and `predictions`.

    Returns
    -------
    object_counts: Tuple[List, List]
        A tuple containing two lists. The first is an array of shape ``(N,)`` containing the number of annotated objects for each image in the dataset.
        The second is an array of shape ``(N,)`` containing the number of predicted objects for each image in the dataset.
    """
    if auxiliary_inputs is None:
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(ALPHA, labels, predictions)
    return (
        [len(sample["lab_bboxes"]) for sample in auxiliary_inputs],
        [len(sample["pred_bboxes"]) for sample in auxiliary_inputs],
    )


def bounding_box_size_distribution(
    labels=None,
    predictions=None,
    *,
    auxiliary_inputs=None,
    class_names: Optional[Dict[Any, Any]] = None,
    sort: bool = False,
) -> Tuple[Dict[Any, List], Dict[Any, List]]:
    """Return the distribution over sizes of annotated and predicted bounding boxes across the dataset, broken down by each class.

    This method can help you find annotated/predicted boxes for a particular class that are abnormally big/small.

    Parameters
    ----------
    labels:
        Annotated boxes and class labels in the original dataset, which may contain some errors.
        Refer to documentation for this argument in :py:func:`object_counts_per_image <cleanlab.object_detection.summary.object_counts_per_image>` for further details.

    predictions:
        Predictions output by a trained object detection model.
        Refer to documentation for this argument in :py:func:`object_counts_per_image <cleanlab.object_detection.summary.object_counts_per_image>` for further details.

    auxiliary_inputs: optional
        Auxiliary inputs to be used in the computation of counts.
        Refer to documentation for this argument in :py:func:`object_counts_per_image <cleanlab.object_detection.summary.object_counts_per_image>` for further details.

    class_names: optional
        A dictionary mapping one-hot-encoded class labels back to their original class names in the format ``{"integer-label": "original-class-name"}``.
        You can use this argument to control the classes for which the size distribution is computed.

    sort: bool
        If True, the returned dictionaries are sorted by the number of instances of each class in the dataset in descending order.

    Returns
    -------
    bbox_sizes: Tuple[Dict[Any, List], Dict[Any, List]]
        A tuple containing two dictionaries. Each maps each class label to a list of the sizes of annotated bounding boxes for that class in the label and prediction datasets, respectively.
    """
    if auxiliary_inputs is None:
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(ALPHA, labels, predictions)

    lab_area: Dict[Any, list] = collections.defaultdict(list)
    pred_area: Dict[Any, list] = collections.defaultdict(list)
    for sample in auxiliary_inputs:
        _get_bbox_areas(sample["lab_labels"], sample["lab_bboxes"], lab_area, class_names)
        _get_bbox_areas(sample["pred_labels"], sample["pred_bboxes"], pred_area, class_names)

    if sort:
        lab_area = dict(sorted(lab_area.items(), key=lambda x: -len(x[1])))
        pred_area = dict(sorted(pred_area.items(), key=lambda x: -len(x[1])))

    return lab_area, pred_area


def class_label_distribution(
    labels=None,
    predictions=None,
    *,
    auxiliary_inputs=None,
    class_names: Optional[Dict[Any, Any]] = None,
) -> Tuple[Dict[Any, float], Dict[Any, float]]:
    """Returns the distribution of class labels associated with all annotated bounding boxes (or predicted bounding boxes) in the dataset.

    This method can help you understand which classes are: rare or over/under-predicted by the model overall.

    Parameters
    ----------
    labels:
        Annotated boxes and class labels in the original dataset, which may contain some errors.
        Refer to documentation for this argument in :py:func:`object_counts_per_image <cleanlab.object_detection.summary.object_counts_per_image>` for further details.

    predictions:
        Predictions output by a trained object detection model.
        Refer to documentation for this argument in :py:func:`object_counts_per_image <cleanlab.object_detection.summary.object_counts_per_image>` for further details.

    auxiliary_inputs: optional
        Auxiliary inputs to be used in the computation of counts.
        Refer to documentation for this argument in :py:func:`object_counts_per_image <cleanlab.object_detection.summary.object_counts_per_image>` for further details.

    class_names: optional
        Optional dictionary mapping one-hot-encoded class labels back to their original class names in the format ``{"integer-label": "original-class-name"}``.

    Returns
    -------
    class_distribution: Tuple[Dict[Any, float], Dict[Any, float]]
        A tuple containing two dictionaries. The first is a dictionary mapping each class label to its frequency in the dataset annotations.
        The second is a dictionary mapping each class label to its frequency in the model predictions across all images in the dataset.
    """
    if auxiliary_inputs is None:
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(ALPHA, labels, predictions)

    lab_freq: DefaultDict[Any, int] = collections.defaultdict(int)
    pred_freq: DefaultDict[Any, int] = collections.defaultdict(int)
    for sample in auxiliary_inputs:
        _get_class_instances(sample["lab_labels"], lab_freq, class_names)
        _get_class_instances(sample["pred_labels"], pred_freq, class_names)

    label_norm = _normalize_by_total(lab_freq)
    pred_norm = _normalize_by_total(pred_freq)

    return label_norm, pred_norm


def get_sorted_bbox_count_idxs(labels, predictions):
    """
    Returns a tuple of idxs and bounding box counts of images sorted from highest to lowest number of bounding boxes.

    This plot can help you discover images with abnormally many/few object annotations.

    Parameters
    ----------
    labels:
        Annotated boxes and class labels in the original dataset, which may contain some errors.
        Refer to documentation for this argument in :py:func:`object_counts_per_image <cleanlab.object_detection.summary.object_counts_per_image>` for further details.

    predictions:
        Predictions output by a trained object detection model.
        Refer to documentation for this argument in :py:func:`object_counts_per_image <cleanlab.object_detection.summary.object_counts_per_image>` for further details.


    Returns
    -------
    sorted_idxs: List[Tuple[int, int]], List[Tuple[int, int]]
        A tuple containing two lists. The first is an array of shape ``(N,)`` containing the number of annotated objects for each image in the dataset.
        The second is an array of shape ``(N,)`` containing the number of predicted objects for each image in the dataset.
    """
    lab_count, pred_count = object_counts_per_image(labels, predictions)
    lab_grouped = list(enumerate(lab_count))
    pred_grouped = list(enumerate(pred_count))

    sorted_lab = sorted(lab_grouped, key=lambda x: x[1], reverse=True)
    sorted_pred = sorted(pred_grouped, key=lambda x: x[1], reverse=True)

    return sorted_lab, sorted_pred


def plot_class_size_distributions(
    labels, predictions, class_names=None, class_to_show=MAX_CLASS_TO_SHOW
):
    """
    Plots the size distributions for bounding boxes for each class.

    This plot can help you find annotated/predicted boxes for a particular class that are abnormally big/small.

    Parameters
    ----------
    labels:
        Annotated boxes and class labels in the original dataset, which may contain some errors.
        Refer to documentation for this argument in :py:func:`object_counts_per_image <cleanlab.object_detection.summary.object_counts_per_image>` for further details.

    predictions:
        Predictions output by a trained object detection model.
        Refer to documentation for this argument in :py:func:`object_counts_per_image <cleanlab.object_detection.summary.object_counts_per_image>` for further details.

    class_names: optional
        Optional dictionary mapping one-hot-encoded class labels back to their original class names in the format ``{"integer-label": "original-class-name"}``.
        You can use this argument to control the classes for which the size distribution is plotted.

    class_to_show: optional
        The number of classes to show in the plots. Classes over `class_to_show` are hidden. If this argument is provided, then the classes are sorted by the number of instances in the dataset.
        Defaults to `MAX_CLASS_TO_SHOW` which is set to 10.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "This functionality requires matplotlib. Install it via: `pip install matplotlib`"
        )

    lab_boxes, pred_boxes = bounding_box_size_distribution(
        labels,
        predictions,
        class_names=class_names,
        sort=True if class_to_show is not None else False,
    )

    for i, c in enumerate(lab_boxes.keys()):
        if i >= class_to_show:
            break
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Size distributions for bounding box for class {c}")
        for i, l in enumerate([lab_boxes, pred_boxes]):
            axs[i].hist(l[c], bins="auto")
            axs[i].set_xlabel("box area (pixels)")
            axs[i].set_ylabel("count")
            axs[i].set_title("annotated" if i == 0 else "predicted")

        plt.show()


def plot_class_distribution(labels, predictions, class_names=None):
    """
    Plots the distribution of class labels associated with all annotated bounding boxes and predicted bounding boxes in the dataset.

    This plot can help you understand which classes are rare or over/under-predicted by the model overall.

    Parameters
    ----------
    labels:
        Annotated boxes and class labels in the original dataset, which may contain some errors.
        Refer to documentation for this argument in :py:func:`object_counts_per_image <cleanlab.object_detection.summary.object_counts_per_image>` for further details.

    predictions:
        Predictions output by a trained object detection model.
        Refer to documentation for this argument in :py:func:`object_counts_per_image <cleanlab.object_detection.summary.object_counts_per_image>` for further details.

    class_names: optional
        Optional dictionary mapping one-hot-encoded class labels back to their original class names in the format ``{"integer-label": "original-class-name"}``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "This functionality requires matplotlib. Install it via: `pip install matplotlib`"
        )

    lab_dist, pred_dist = class_label_distribution(labels, predictions, class_names=class_names)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Distribution of classes in the dataset")
    for i, d in enumerate([lab_dist, pred_dist]):
        axs[i].pie(d.values(), labels=d.keys(), autopct="%1.1f%%")
        axs[i].set_title("Annotated" if i == 0 else "Predicted")

    plt.show()


def visualize(
    image: Union[str, np.ndarray, Image],
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

        Note: Here, ``(x1,y1)`` corresponds to the top-left and ``(x2,y2)`` corresponds to the bottom-right corner of the bounding box with respect to the image matrix [e.g. `XYXY in Keras <https://keras.io/api/keras_cv/bounding_box/formats/>`, `Detectron 2 <https://detectron2.readthedocs.io/en/latest/modules/utils.html#detectron2.utils.visualizer.Visualizer.draw_box>`].

    prediction:
        A prediction for a single image in the format ``np.ndarray((K,))`` and ``prediction[k]`` is of shape ``np.ndarray(N,5)``
        where ``M`` is the number of predicted bounding boxes for class ``k`` and the five columns correspond to ``[x,y,x,y,pred_prob]`` where
        ``[x1,y1,x2,y2]`` are the bounding box coordinates predicted by the model and ``pred_prob`` is the model's confidence in ``predictions[i]``.

        Note: Here, ``(x1,y1)`` corresponds to the top-left and ``(x2,y2)`` corresponds to the bottom-right corner of the bounding box with respect to the image matrix [e.g. `XYXY in Keras <https://keras.io/api/keras_cv/bounding_box/formats/>`, `Detectron 2 <https://detectron2.readthedocs.io/en/latest/modules/utils.html#detectron2.utils.visualizer.Visualizer.draw_box>`]. The last column, pred_prob, represents the predicted probability that the bounding box contains an object of the class k.

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


def _get_per_class_confusion_matrix_dict_(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    iou_threshold: Optional[float] = 0.5,
    num_procs: int = 1,
) -> DefaultDict[int, Dict[str, int]]:
    """
    Returns a confusion matrix dictionary for each class containing the number of True Positive, False Positive, and False Negative detections from the object detection model.
    """
    num_classes = len(predictions[0])
    num_images = len(predictions)
    pool = Pool(num_procs)
    counter_dict: DefaultDict[int, dict[str, int]] = collections.defaultdict(
        lambda: {"TP": 0, "FP": 0, "FN": 0}
    )

    for class_num in range(num_classes):
        pred_bboxes, lab_bboxes = _filter_by_class(labels, predictions, class_num)
        tpfpfn = pool.starmap(
            _calculate_true_positives_false_positives,
            zip(
                pred_bboxes,
                lab_bboxes,
                [iou_threshold for _ in range(num_images)],
                [True for _ in range(num_images)],
            ),
        )

        for image_idx, (tp, fp, fn) in enumerate(tpfpfn):  # type: ignore
            counter_dict[class_num]["TP"] += np.sum(tp)
            counter_dict[class_num]["FP"] += np.sum(fp)
            counter_dict[class_num]["FN"] += np.sum(fn)

    return counter_dict


def _sort_dict_to_list(index_value_dict):
    """
    Convert a dictionary to a list sorted by index and return the values in that order.

    Parameters:
    - index_value_dict (dict): The input dictionary where keys represent indices and values are the corresponding elements.

    Returns:
    list: A list containing the values from the input dictionary, sorted by index.

    Example:
    >>> my_dict = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4'}
    >>> sort_dict_to_list(my_dict)
    ['0', '1', '2', '3', '4']
    """
    sorted_list = [
        value for key, value in sorted(index_value_dict.items(), key=lambda x: int(x[0]))
    ]
    return sorted_list


def get_average_per_class_confusion_matrix(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    num_procs: int = 1,
    class_names: Optional[Dict[Any, Any]] = None,
) -> Dict[Union[int, str], Dict[str, float]]:
    """
    Compute a confusion matrix dictionary for each class containing the average number of True Positive, False Positive, and False Negative detections from the object detection model across a range of Intersection over Union thresholds.

    At each IoU threshold, the metrics are calculated as follows:
    - True Positive (TP): Instances where the model correctly identifies the class with IoU above the threshold.
    - False Positive (FP): Instances where the model predicts the class, but IoU is below the threshold.
    - False Negative (FN): Instances where the ground truth class is not predicted by the model.

    The average confusion matrix provides insights into the model strengths and potential biases.

    Note:  lower TP at certain IoU thresholds does not necessarily imply that everything else is FP, instead it indicates that, at those specific IoU thresholds, the model is not performing as well in terms of correctly identifying class instances. The other metrics (FP and FN) provide additional information about the model's behavior.

    Note: Since we average over many IoU thresholds, 'TP', 'FP', and 'FN' may contain float values representing the average across these thresholds.

    Parameters
    ----------
    labels:
        A list of ``N`` dictionaries such that ``labels[i]`` contains the given labels for the `i`-th image.
        Refer to documentation for this argument in :py:func:`object_detection.filter.find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    predictions:
        A list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model predictions for the `i`-th image.
        Refer to documentation for this argument in :py:func:`object_detection.filter.find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    num_procs:
        Number of processes for parallelization. Default is 1.

    class_names:
        Optional dictionary mapping one-hot-encoded class labels back to their original class names in the format ``{"integer-label": "original-class-name"}``


    Returns
    -------
    avg_metrics: dict
        A distionary containing the average confusion matrix.

        The default range of Intersection over Union thresholds is from 0.5 to 0.95 with a step size of 0.05.
    """
    iou_thrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
    num_classes = len(predictions[0])
    if class_names is None:
        class_names = {str(i): int(i) for i in list(range(num_classes))}
    class_names = _sort_dict_to_list(class_names)
    avg_metrics = {class_num: {"TP": 0.0, "FP": 0.0, "FN": 0.0} for class_num in class_names}

    for iou_threshold in iou_thrs:
        results_dict = _get_per_class_confusion_matrix_dict_(
            labels, predictions, iou_threshold, num_procs
        )

        for class_num in results_dict:
            tp = results_dict[class_num]["TP"]
            fp = results_dict[class_num]["FP"]
            fn = results_dict[class_num]["FN"]

            avg_metrics[class_names[class_num]]["TP"] += tp
            avg_metrics[class_names[class_num]]["FP"] += fp
            avg_metrics[class_names[class_num]]["FN"] += fn

    num_thresholds = len(iou_thrs) * len(results_dict)
    for class_name in avg_metrics:
        avg_metrics[class_name]["TP"] /= num_thresholds
        avg_metrics[class_name]["FP"] /= num_thresholds
        avg_metrics[class_name]["FN"] /= num_thresholds
    return avg_metrics


def calculate_per_class_metrics(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    num_procs: int = 1,
    class_names=None,
) -> Dict[Union[int, str], Dict[str, float]]:
    """
    Calculate the object detection model's precision, recall, and F1 score for each class in the dataset.

    These metrics can help you identify model strengths and weaknesses, and provide reference statistics for model evaluation and comparisons.

    Parameters
    ----------
    labels:
        A list of ``N`` dictionaries such that ``labels[i]`` contains the given labels for the `i`-th image.
        Refer to documentation for this argument in :py:func:`object_detection.filter.find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    predictions:
        A list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model predictions for the `i`-th image.
        Refer to documentation for this argument in :py:func:`object_detection.filter.find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    num_procs:
        Number of processes for parallelization. Default is 1.

    class_names:
        Optional dictionary mapping one-hot-encoded class labels back to their original class names in the format ``{"integer-label": "original-class-name"}``


    Returns
    -------
    per_class_metrics: dict
        A dictionary containing per-class metrics computed from the object detection model's average confusion matrix values across a range of Intersection over Union thresholds.

        The default range of Intersection over Union thresholds is from 0.5 to 0.95 with a step size of 0.05.
    """
    avg_metrics = get_average_per_class_confusion_matrix(
        labels, predictions, num_procs, class_names=class_names
    )

    avg_metrics_dict = {}
    for class_name in avg_metrics:
        tp = avg_metrics[class_name]["TP"]
        fp = avg_metrics[class_name]["FP"]
        fn = avg_metrics[class_name]["FN"]

        precision = tp / (tp + fp + TINY_VALUE)  # Avoid division by zero
        recall = tp / (tp + fn + TINY_VALUE)  # Avoid division by zero
        f1 = 2 * (precision * recall) / (precision + recall + TINY_VALUE)  # Avoid division by zero

        avg_metrics_dict[class_name] = {
            "average precision": precision,
            "average recall": recall,
            "average f1": f1,
        }

    return avg_metrics_dict


def _normalize_by_total(freq):
    """Helper function to normalize a frequency distribution."""
    total = sum(freq.values())
    return {k: round(v / (total + EPSILON), 2) for k, v in freq.items()}


def _get_bbox_areas(labels, boxes, class_area_dict, class_names=None) -> None:
    """Helper function to compute the area of bounding boxes for each class."""
    for cl, bbox in zip(labels, boxes):
        if class_names is not None:
            if str(cl) not in class_names:
                continue
            cl = class_names[str(cl)]
        class_area_dict[cl].append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _get_class_instances(labels, class_instances_dict, class_names=None) -> None:
    """Helper function to count the number of class instances in each image."""
    for cl in labels:
        if class_names is not None:
            cl = class_names[str(cl)]
        class_instances_dict[cl] += 1


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

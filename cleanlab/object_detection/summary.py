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

"""Methods to display examples and their label issues in an object detection dataset."""
from typing import List, Optional, Any, Dict, Tuple, Union

from PIL import Image
import numpy as np

from cleanlab.internal.constants import MAX_CLASS_TO_SHOW
from cleanlab.object_detection.rank import (
    _separate_prediction,
    _separate_label,
    _get_prediction_type,
)


def visualize(
    image: Union[str, Image.Image],
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
        ``L`` is the number of bounding boxes for the `i`-th image and ``bboxes[j]`` is in the format ``[x,y,x,y]`` with given label ``labels[j]``.

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
        Path to save figure at. If a path is provided, the figure is saved instead of displayed.

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
                fig, ax, abbox, alabels, edgecolor="r", linestyle="-.", linewidth=1
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

    if label or prediction is not None:
        _ = _plot_legend(class_names, label, prediction)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


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

    return legend


def _draw_labels(ax, rect, label, edgecolor):
    """Helper function to draw labels on an axis."""

    rx, ry = rect.get_xy()
    c_xleft = rx + 10
    c_xright = rx + rect.get_width() - 10
    c_ytop = ry + 12

    if edgecolor == "r":
        cx, cy = c_xright, c_ytop
    else:  # edgecolor == b
        cx, cy = c_xleft, c_ytop

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


def _bbox_xyxy_to_xywh(bbox: List[float]) -> Optional[List[float]]:
    """Converts bounding box coodrinate types from x1y1,x2y2 to x,y,w,h"""
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        return [x1, y1, w, h]
    else:
        print("Wrong bbox shape", len(bbox))
        return None

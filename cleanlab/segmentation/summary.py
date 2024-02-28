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
Methods to display images and their label issues in a semantic segmentation dataset, as well as summarize the overall types of issues identified.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from cleanlab.internal.segmentation_utils import _get_summary_optional_params


def display_issues(
    issues: np.ndarray,
    *,
    labels: Optional[np.ndarray] = None,
    pred_probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    exclude: Optional[List[int]] = None,
    top: Optional[int] = None,
) -> None:
    """
    Display semantic segmentation label issues, showing images with problematic pixels highlighted.

    Can also show given and predicted masks for each image identified to have label issue.

    Parameters
    ----------
    issues:
      Boolean **mask** for the entire dataset
      where ``True`` represents a pixel label issue and ``False`` represents an example that is
      accurately labeled.

      Same format as output by :py:func:`segmentation.filter.find_label_issues <cleanlab.segmentation.filter.find_label_issues>`
      or :py:func:`segmentation.rank.issues_from_scores <cleanlab.segmentation.rank.issues_from_scores>`.

    labels:
      Optional discrete array of noisy labels for a segmantic segmentation dataset, in the shape ``(N,H,W,)``,
      where each pixel must be integer in 0, 1, ..., K-1.
      If `labels` is provided, this function also displays given label of the pixel identified with issue.
      Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.segmentation.filter.find_label_issues>` for more information.

    pred_probs:
      Optional array of shape ``(N,K,H,W,)`` of model-predicted class probabilities.
      If `pred_probs` is provided, this function also displays predicted label of the pixel identified with issue.
      Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.segmentation.filter.find_label_issues>` for more information.

      Tip
      ---
      If your labels are one hot encoded you can `np.argmax(labels_one_hot, axis=1)` assuming that `labels_one_hot` is of dimension (N,K,H,W)
      before entering in the function

    class_names:
      Optional list of strings, where each string represents the name of a class in the semantic segmentation problem.
      The order of the names should correspond to the numerical order of the classes. The list length should be
      equal to the number of unique classes present in the labels.
      If provided, this function will generate a legend
      showing the color mapping of each class in the provided colormap.

      Example:
      If there are three classes in your labels, represented by 0, 1, 2, then class_names might look like this:

      .. code-block:: python

            class_names = ['background', 'person', 'dog']

    top:
        Optional maximum number of issues to be printed. If not provided, a good default is used.

    exclude:
        Optional list of label classes that can be ignored in the errors, each element must be 0, 1, ..., K-1

    """
    class_names, exclude, top = _get_summary_optional_params(class_names, exclude, top)
    if labels is None and len(exclude) > 0:
        raise ValueError("Provide labels to allow class exclusion")

    top = min(top, len(issues))

    correct_ordering = np.argsort(-np.sum(issues, axis=(1, 2)))[:top]

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap
    except:
        raise ImportError('try "pip install matplotlib"')

    output_plots = (pred_probs is not None) + (labels is not None) + 1

    # Colormap for errors
    error_cmap = ListedColormap(["none", "red"])
    _, h, w = issues.shape
    if output_plots > 1:
        if pred_probs is not None:
            _, num_classes, _, _ = pred_probs.shape
            cmap = _generate_colormap(num_classes)
        elif labels is not None:
            num_classes = max(np.unique(labels)) + 1
            cmap = _generate_colormap(num_classes)
    else:
        cmap = None

    # Show a legend
    if class_names is not None and cmap is not None:
        patches = [
            mpatches.Patch(color=cmap[i], label=class_names[i]) for i in range(len(class_names))
        ]
        legend = plt.figure()  # adjust figsize for larger legend
        legend.legend(
            handles=patches, loc="center", ncol=len(class_names), facecolor="white", fontsize=20
        )  # adjust fontsize for larger text
        plt.axis("off")
        plt.show()

    for i in correct_ordering:
        # Show images
        fig, axes = plt.subplots(1, output_plots, figsize=(5 * output_plots, 5))
        plot_index = 0

        # First image - Given truth labels
        if labels is not None:
            axes[plot_index].imshow(cmap[labels[i]])
            axes[plot_index].set_title("Given Labels")
            plot_index += 1

        # Second image - Argmaxed pred_probs
        if pred_probs is not None:
            axes[plot_index].imshow(cmap[np.argmax(pred_probs[i], axis=0)])
            axes[plot_index].set_title("Argmaxed Prediction Probabilities")
            plot_index += 1

        # Third image - Errors
        if output_plots == 1:
            ax = axes
        else:
            ax = axes[plot_index]

        mask = np.full((h, w), True)
        if labels is not None and len(exclude) != 0:
            mask = ~np.isin(labels[i], exclude)
        ax.imshow(issues[i] & mask, cmap=error_cmap, vmin=0, vmax=1)
        ax.set_title(f"Image {i}: Suggested Errors (in Red)")
        plt.show()

    return None


def common_label_issues(
    issues: np.ndarray,
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    class_names: Optional[List[str]] = None,
    exclude: Optional[List[int]] = None,
    top: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Display the frequency of which label are swapped in the dataset.

    These may correspond to pixels that are ambiguous or systematically misunderstood by the data annotators.

    * N - Number of images in the dataset
    * K - Number of classes in the dataset
    * H - Height of each image
    * W - Width of each image

    Parameters
    ----------
    issues:
      Boolean **mask** for the entire dataset
      where ``True`` represents a pixel label issue and ``False`` represents an example that is
      accurately labeled.

      Same format as output by :py:func:`segmentation.filter.find_label_issues <cleanlab.segmentation.filter.find_label_issues>`
      or :py:func:`segmentation.rank.issues_from_scores <cleanlab.segmentation.rank.issues_from_scores>`.

    labels:
      A discrete array of noisy labels for a segmantic segmentation dataset, in the shape ``(N,H,W,)``.
      where each pixel must be integer in 0, 1, ..., K-1.
      Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.segmentation.filter.find_label_issues>` for more information.

    pred_probs:
      An array of shape ``(N,K,H,W,)`` of model-predicted class probabilities.
      Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.segmentation.filter.find_label_issues>` for more information.

      Tip
      ---
      If your labels are one hot encoded you can `np.argmax(labels_one_hot, axis=1)` assuming that `labels_one_hot` is of dimension (N,K,H,W)
      before entering in the function

    class_names:
      Optional length K list of names of each class, such that `class_names[i]` is the string name of the class corresponding to `labels` with value `i`.
      If `class_names` is provided, display these string names for predicted and given labels, otherwise display the integer index of classes.

    exclude:
      Optional list of label classes that can be ignored in the errors, each element must be in 0, 1, ..., K-1.

    top:
      Optional maximum number of tokens to print information for. If not provided, a good default is used.

    verbose:
      Set to ``False`` to suppress all print statements.

    Returns
    -------
    issues_df:
      DataFrame with columns ``['given_label', 'predicted_label', 'num_label_issues']``
      where each row contains information about a particular given/predicted label swap.
      Rows are ordered by the number of label issues inferred to exhibit this type of label swap.
    """
    try:
        N, K, H, W = pred_probs.shape
    except:
        raise ValueError("pred_probs must be of shape (N, K, H, W)")

    assert labels.shape == (N, H, W), "labels must be of shape (N, H, W)"

    class_names, exclude, top = _get_summary_optional_params(class_names, exclude, top)
    # Find issues by pixel coordinates
    issue_coords = np.column_stack(np.where(issues))

    # Count issues per class (given label)
    count: Dict[int, Any] = {}
    for i, j, k in tqdm(issue_coords):
        label = labels[i, j, k]
        pred = pred_probs[i, :, j, k].argmax()
        if label not in count:
            count[label] = np.zeros(K, dtype=int)
        if pred not in exclude:
            count[label][pred] += 1

    # Prepare output DataFrame
    if class_names is None:
        class_names = [str(i) for i in range(K)]

    info = []
    for given_label, class_name in enumerate(class_names):
        if given_label in count:
            for pred_label, num_issues in enumerate(count[given_label]):
                if num_issues > 0:
                    info.append([class_name, class_names[pred_label], num_issues])

    info = sorted(info, key=lambda x: x[2], reverse=True)[:top]
    issues_df = pd.DataFrame(info, columns=["given_label", "predicted_label", "num_pixel_issues"])

    if verbose:
        for idx, row in issues_df.iterrows():
            print(
                f"Class '{row['given_label']}' is potentially mislabeled as class for '{row['predicted_label']}' "
                f"{row['num_pixel_issues']} pixels in the dataset"
            )

    return issues_df


def filter_by_class(
    class_index: int, issues: np.ndarray, labels: np.ndarray, pred_probs: np.ndarray
) -> np.ndarray:
    """
    Return label issues involving particular class. Note that this includes errors where the given label is the class of interest, and the predicted label is any other class.

    Parameters
    ----------
    class_index:
      The specific class you are interested in.

    issues:
      Boolean **mask** for the entire dataset where ``True`` represents a pixel label issue and ``False`` represents an example that is
      accurately labeled.

      Same format as output by :py:func:`segmentation.filter.find_label_issues <cleanlab.segmentation.filter.find_label_issues>`
      or :py:func:`segmentation.rank.issues_from_scores <cleanlab.segmentation.rank.issues_from_scores>`.

    labels:
      A discrete array of noisy labels for a segmantic segmentation dataset, in the shape ``(N,H,W,)``,
      where each pixel must be integer in 0, 1, ..., K-1.
      Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.segmentation.filter.find_label_issues>` for further details.

    pred_probs:
      An array of shape ``(N,K,H,W,)`` of model-predicted class probabilities.
      Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.segmentation.filter.find_label_issues>` for further details.

    Returns
    ----------
    issues_subset:
      Boolean **mask** for the subset dataset where ``True`` represents a pixel label issue and ``False`` represents an example that is
      accurately labeled for the labeled class.

      Returned mask shows **all** instances that involve the particular class of interest.


    """
    issues_subset = (issues & np.isin(labels, class_index)) | (
        issues & np.isin(pred_probs.argmax(1), class_index)
    )
    return issues_subset


def _generate_colormap(num_colors):
    """
    Finds a unique color map based on the number of colors inputted ideal for semantic segmentation.
    Parameters
    ----------
    num_colors:
        How many unique colors you want

    Returns
    -------
    colors:
        colors with num_colors distinct colors
    """

    try:
        from matplotlib.cm import hsv
    except:
        raise ImportError('try "pip install matplotlib"')

    num_shades = 7
    num_colors_with_shades = -(-num_colors // num_shades) * num_shades
    linear_nums = np.linspace(0, 1, num_colors_with_shades, endpoint=False)

    arr_by_shade_rows = linear_nums.reshape(num_shades, -1)
    arr_by_shade_columns = arr_by_shade_rows.T
    num_partitions = arr_by_shade_columns.shape[0]
    nums_distributed_like_rising_saw = arr_by_shade_columns.flatten()

    initial_cm = hsv(nums_distributed_like_rising_saw)
    lower_partitions_half = num_partitions // 2
    upper_partitions_half = num_partitions - lower_partitions_half

    lower_half = lower_partitions_half * num_shades
    initial_cm[:lower_half, :3] *= np.linspace(0.2, 1, lower_half)[:, np.newaxis]

    upper_half_indices = np.arange(lower_half, num_colors_with_shades).reshape(
        upper_partitions_half, num_shades
    )
    modifier = (
        (1 - initial_cm[upper_half_indices, :3])
        * np.arange(upper_partitions_half)[:, np.newaxis, np.newaxis]
        / upper_partitions_half
    )
    initial_cm[upper_half_indices, :3] += modifier
    colors = initial_cm[:num_colors]
    return colors

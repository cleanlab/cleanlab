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
Methods to display images and their label issues in a semantic segmentation classification dataset, as well as summarize the types of issues identified.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cleanlab.internal.token_classification_utils import color_sentence, get_sentence


def display_issues(
    issues: np.ndarray,
    *,
    labels: np.ndarray,
    pred_probs: np.ndarray,
    exclude: List[Tuple[int, int]] = [],
    class_names: Optional[List[str]] = None,
    top: int = 20
) -> None:
    """
    Display semantic segmentation label issues, showing images with problematic pixels highlighted.

    Can also shows given and predicted label for each pixel identified to have label issue.

    Parameters
    ----------
    issues:
        Boolean **mask** for the entire dataset
        where ``True`` represents a pixel label issue and ``False`` represents an example that is
        accurately labeled.

        Same format as output by :py:func:`segmentation.filter.find_label_issues <cleanlab.segmentation.filter.find_label_issues>`
        or :py:func:`segmentation.rank.issues_from_scores <cleanlab.segmentation.rank.issues_from_scores>`.
        
    labels : np.ndarray 
       Optional discrete array of noisy labels for a classification dataset, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, each pixel must be integer in 0, 1, ..., K-1.
      For a standard (multi-class) classification dataset where each example is labeled with one class,
      `labels` should be 3-D array of shape ``(N,H,W,)``. 

      If `labels` is provided, this function also displays given label of the pixel identified with issue.
      
    Tip: If your labels are one hot encoded you can `np.argmax(labels_one_hot,axis=1)` assuming that `labels_one_hot` is of dimension (N,K,H,W)
    before entering in the function

    pred_probs : np.ndarray
      Optional array of shape ``(N,K,H,W,)`` of model-predicted class probabilities,
      ``P(label=k|x)``. Each pixel contains an array of K classes, where for 
      an example `x` the array at each pixel contains the model-predicted probabilities 
      that `x` belongs to each of the K classes.
      
    If `pred_probs` is provided, this function also displays predicted label of the pixel identified with issue.

    exclude:
        Optional list of given/predicted label swaps (tuples) to be ignored. For example, if `exclude=[(0, 1), (1, 0)]`,
        works whose label was likely swapped between class 0 and 1 are not displayed. Class labels must be in 0, 1, ..., K-1.

    class_names:
        Optional length K list of names of each class, such that `class_names[i]` is the string name of the class corresponding to `labels` with value `i`.

        If `class_names` is provided, display these string names for predicted and given labels, otherwise display the integer index of classes.

    top: int, default=20
        Maximum number of issues to be printed.

    """
    if not class_names:
    print(
        "Classes will be printed in terms of their integer index since `class_names` was not provided. "
    )
    print("Specify this argument to see the string names of each class. \n")

    top = min(top, len(issues))
    shown = 0
    is_tuple = isinstance(issues[0], tuple)

    for i in range(len(issues)):
        if shown >= top:
            break
        
        if is_tuple:
            x, y = issues[i]
        else:
            x, y = np.unravel_index(i, issues.shape)

        given_label = labels[x, y]
        pred_label = np.argmax(pred_probs[x, y])

        if (given_label, pred_label) in exclude:
            continue

        # Show images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # First image - Ground truth labels
        axes[0].imshow(labels[x], cmap='jet')
        axes[0].set_title("Ground Truth Labels")
        
        # Second image - Argmaxed pred_probs
        axes[1].imshow(np.argmax(pred_probs[x], axis=0), cmap='jet')
        axes[1].set_title("Argmaxed Prediction Probabilities")
        
        # Third image - Errors
        error_map = (labels[x] != np.argmax(pred_probs[x], axis=0)).astype(int)
        axes[2].imshow(error_map, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title("Errors")
        
        plt.show()
        shown += 1




def common_label_issues(
    issues: List[Tuple[int, int]],
    tokens: List[List[str]],
    *,
    labels: Optional[list] = None,
    pred_probs: Optional[list] = None,
    class_names: Optional[List[str]] = None,
    top: int = 10,
    exclude: List[Tuple[int, int]] = [],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Display the pixels that most commonly have label issues.

    These may correspond to pixels that are ambiguous or systematically misunderstood by the data annotators.

    Parameters
    ----------
    issues:
        Boolean **mask** for the entire dataset
        where ``True`` represents a pixel label issue and ``False`` represents an example that is
        accurately labeled.

        Same format as output by :py:func:`segmentation.filter.find_label_issues <cleanlab.segmentation.filter.find_label_issues>`
        or :py:func:`segmentation.rank.issues_from_scores <cleanlab.segmentation.rank.issues_from_scores>`.

    labels : np.ndarray 
       Optional discrete array of noisy labels for a classification dataset, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, each pixel must be integer in 0, 1, ..., K-1.
      For a standard (multi-class) classification dataset where each example is labeled with one class,
      `labels` should be 3-D array of shape ``(N,H,W,)``. 

      If `labels` is provided, this function also displays given label of the pixel identified with issue.
      
    Tip: If your labels are one hot encoded you can `np.argmax(labels_one_hot,axis=1)` assuming that `labels_one_hot` is of dimension (N,K,H,W)
    before entering in the function

    pred_probs : np.ndarray
      Optional array of shape ``(N,K,H,W,)`` of model-predicted class probabilities,
      ``P(label=k|x)``. Each pixel contains an array of K classes, where for 
      an example `x` the array at each pixel contains the model-predicted probabilities 
      that `x` belongs to each of the K classes.
      
    If `pred_probs` is provided, this function also displays predicted label of the pixel identified with issue.
        If both `labels` and `pred_probs` are provided, also reports each type of given/predicted label swap for tokens identified to commonly suffer from label issues.

    class_names:
        Optional length K list of names of each class, such that `class_names[i]` is the string name of the class corresponding to `labels` with value `i`.

        If `class_names` is provided, display these string names for predicted and given labels, otherwise display the integer index of classes.

    top:
        Maximum number of tokens to print information for.

    exclude:
        Optional list of given/predicted label swaps (tuples) to be ignored in the same format as `exclude` for
        :py:func:`segmentation.summary.display_issues <cleanlab.segmentation.summary.display_issues>`.

    verbose:
        Whether to also print out the token information in the returned DataFrame `df`.

    Returns
    -------
    df:
        If both `labels` and `pred_probs` are provided, DataFrame `df` contains columns ``['class', 'given_label',
        'predicted_label', 'num_label_issues']``, and each row contains information for a specific class and
        given/predicted label swap, ordered by the number of label issues inferred for this type of label swap.

        Otherwise, `df` only has columns ['class', 'num_label_issues'], and each row contains the information for a specific
        token, ordered by the number of total label issues involving this token.


    """
    if not class_names:
    print(
        "Classes will be printed in terms of their integer index since `class_names` was not provided. "
    )
    print("Specify this argument to see the string names of each class. \n")

    class_counter = Counter()
    label_swap_counter = Counter()

    for issue in issues:
        x, y = issue
        class_name = tokens[x][y]
        class_counter[class_name] += 1

        if labels is not None and pred_probs is not None:
            given_label = labels[x, y]
            pred_label = np.argmax(pred_probs[x, y])

            if (given_label, pred_label) not in exclude:
                label_swap_counter[(class_name, given_label, pred_label)] += 1

    top_classes = class_counter.most_common(top)
    columns = ['class', 'num_label_issues']
    data = top_classes

    if labels is not None and pred_probs is not None:
        top_label_swaps = label_swap_counter.most_common(top)
        columns.extend(['given_label', 'predicted_label'])
        data = [item[0] + (item[1],) for item in top_label_swaps]

        if class_names:
            data = [(class_name, class_names[given], class_names[pred], count)
                    for (class_name, given, pred, count) in data]

    df = pd.DataFrame(data, columns=columns)

    if verbose:
        print(df)

    return df
    

def filter_by_class(
    class_index: int, issues: np.ndarray, labels: np.ndarray 
) -> np.ndarray :
    """
    Return subset of label issues involving a particular class in labels.

    Parameters
    ----------
    class_index:
        A specific class you are interested in.

    issues:
        Boolean **mask** for the entire dataset
        where ``True`` represents a pixel label issue and ``False`` represents an example that is
        accurately labeled.

        Same format as output by :py:func:`segmentation.filter.find_label_issues <cleanlab.segmentation.filter.find_label_issues>`
        or :py:func:`segmentation.rank.issues_from_scores <cleanlab.segmentation.rank.issues_from_scores>`.

    labels : np.ndarray 
       Optional discrete array of noisy labels for a classification dataset, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, each pixel must be integer in 0, 1, ..., K-1.
      For a standard (multi-class) classification dataset where each example is labeled with one class,
      `labels` should be 3-D array of shape ``(N,H,W,)``. 

      If `labels` is provided, this function also displays given label of the pixel identified with issue.
      
    Tip: If your labels are one hot encoded you can `np.argmax(labels_one_hot,axis=1)` assuming that `labels_one_hot` is of dimension (N,K,H,W)
    before entering in the function


    Returns
    ----------
    issues_subset:
        Boolean **mask** for the subset dataset
        where ``True`` represents a pixel label issue and ``False`` represents an example that is
        accurately labeled for the labeled class.


    """
    mask = labels==class_index
    return np.logical_and(mask, issues)
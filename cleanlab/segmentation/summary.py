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
from tqdm import tqdm

from cleanlab.internal.token_classification_utils import color_sentence, get_sentence


def display_issues(
    issues: np.ndarray,
    labels: np.ndarray=None,
    pred_probs: np.ndarray=None,
    exclude: List[int] = [],
    top: int = 20
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
        Optional list of label classes that can be ignored in the errors, each element must be 0, 1, ..., K-1

    top: int, default=20
        Maximum number of issues to be printed.

    """
    if labels is None and len(exclude)>0:
        raise ValueError("Provide labels to allow class exclusion")

    top = min(top, len(issues))

    correct_ordering = np.argsort(-np.sum(issues, axis=(1,2)))[:top]

    try:
        import matplotlib.pyplot as plt
    except:
        raise ImportError("try \"pip install matplotlib\"")
        
        
    output_plots = (pred_probs is not None) + (labels is not None) +1
    
    _,h,w = issues.shape
    if output_plots >1: 
        if pred_probs is not None:
            _,num_classes,_,_ = pred_probs.shape
        else:
            num_classes = max(np.unique(labels))+1
        cmap = generate_colormap(num_classes)
    
    for i in correct_ordering:

        # Show images
        fig, axes = plt.subplots(1, output_plots, figsize=(5*output_plots, 5))
        plot_index = 0
        
        # First image - Given truth labels
        if labels is not None:
            axes[plot_index].imshow(cmap[labels[i]])
            axes[plot_index].set_title("Given Labels")
            plot_index+=1
            
        # Second image - Argmaxed pred_probs
        if pred_probs is not None:
            axes[plot_index].imshow(cmap[np.argmax(pred_probs[i], axis=0)])
            axes[plot_index].set_title("Argmaxed Prediction Probabilities")
            plot_index+=1
        
        # Third image - Errors
        mask = np.full((h,w), True) if len(exclude)== 0 else ~np.isin(labels[i], exclude)
        axes[plot_index].imshow(issues[i]& mask, cmap='gray', vmin=0, vmax=1)
        axes[plot_index].set_title(f"Suggested Errors in image index {i}")
        plt.show()
        
        plot_index = 0

    return None



def common_label_issues(
    issues: np.ndarray,
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    exclude: List[int] = [],
    top: int = 20,
    class_names: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Display the frequency of which label are swapped in the dataset.

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
      Discrete array of noisy labels for a classification dataset, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, each pixel must be integer in 0, 1, ..., K-1.
      For a standard (multi-class) classification dataset where each example is labeled with one class,
      `labels` should be 3-D array of shape ``(N,H,W,)``. 
      
    Tip: If your labels are one hot encoded you can `np.argmax(labels_one_hot,axis=1)` assuming that `labels_one_hot` is of dimension (N,K,H,W)
    before entering in the function

    pred_probs : np.ndarray
      Array of shape ``(N,K,H,W,)`` of model-predicted class probabilities,
      ``P(label=k|x)``. Each pixel contains an array of K classes, where for 
      an example `x` the array at each pixel contains the model-predicted probabilities 
      that `x` belongs to each of the K classes.

    class_names:
        Optional length K list of names of each class, such that `class_names[i]` is the string name of the class corresponding to `labels` with value `i`.

        If `class_names` is provided, display these string names for predicted and given labels, otherwise display the integer index of classes.

    top:
        Maximum number of tokens to print information for.

    exclude:
        Optional list of label classes that can be ignored in the errors, each element must be 0, 1, ..., K-1

    verbose:
        Whether to also print out the token information in the returned DataFrame `df`.

    Returns
    -------
    df:
        DataFrame `df` contains columns ``['given_label',
        'predicted_label', 'num_label_issues']``, and each row contains information for a
        given/predicted label swap, ordered by the number of label issues inferred for this type of label swap.


    """
    try:
        N, K, H, W = pred_probs.shape
    except:
        raise ValueError("pred_probs must be of shape (N, K, H, W)")

    assert labels.shape == (N, H, W), "labels must be of shape (N, H, W)"

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
    df = pd.DataFrame(info, columns=["given_label", "predicted_label", "num_label_issues"])

    if verbose:
        for idx, row in df.iterrows():
            print(f"Class '{row['given_label']}' is potentially mislabeled as class '{row['predicted_label']}' "
                f"{row['num_label_issues']} times throughout the dataset")

    return df
    

def filter_by_class(
    class_index: int, issues: np.ndarray, labels: np.ndarray 
) -> np.ndarray :
    """
    Return subset of label issues involving a particular class.

    Parameters
    ----------
    class_index:
        A specific class you are interested in

    issues:
        Boolean **mask** for the entire dataset
        where ``True`` represents a pixel label issue and ``False`` represents an example that is
        accurately labeled.

        Same format as output by :py:func:`segmentation.filter.find_label_issues <cleanlab.segmentation.filter.find_label_issues>`
        or :py:func:`segmentation.rank.issues_from_scores <cleanlab.segmentation.rank.issues_from_scores>`.
    
    pred_probs : np.ndarray
      Array of shape ``(N,K,H,W,)`` of model-predicted class probabilities,
      ``P(label=k|x)``. Each pixel contains an array of K classes, where for 
      an example `x` the array at each pixel contains the model-predicted probabilities 
      that `x` belongs to each of the K classes.
      
      
    labels : np.ndarray 
       Optional discrete array of noisy labels for a classification dataset, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, each pixel must be integer in 0, 1, ..., K-1.
      For a standard (multi-class) classification dataset where each example is labeled with one class,
      `labels` should be 3-D array of shape ``(N,H,W,)``. 

      
    Tip: If your labels are one hot encoded you can `np.argmax(labels_one_hot,axis=1)` assuming that `labels_one_hot` is of dimension (N,K,H,W)
    before entering in the function


    Returns
    ----------
    issues_subset:
        Boolean **mask** for the subset dataset
        where ``True`` represents a pixel label issue and ``False`` represents an example that is
        accurately labeled for the labeled class.


    """
    issues_subset = issues & np.isin(labels, class_index) & np.isin(pred_probs.argmax(1), class_index)
    return issues_subset
    
def generate_colormap(num_colors):
    
    """
    Finds a unique color map based on the number of colors inputted ideal for semantic segmentation.
    Parameters
    ----------
    num_colors:int 
        How many unique colors you want 
        
    Returns
    -------
    colors:
        colors with num_colors distinct colors 
    
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.cm import hsv
    except:
        raise ImportError("try \"pip install matplotlib\"")
    
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

    upper_half_indices = np.arange(lower_half, num_colors_with_shades).reshape(upper_partitions_half, num_shades)
    modifier = (1 - initial_cm[upper_half_indices, :3]) * np.arange(upper_partitions_half)[:, np.newaxis, np.newaxis] / upper_partitions_half
    initial_cm[upper_half_indices, :3] += modifier
    colors = (initial_cm[:num_colors])
    return colors

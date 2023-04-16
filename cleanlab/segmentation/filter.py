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

import numpy as np 
from typing import Optional, List, Tuple, Any
from tqdm import tqdm


def get_label_quality_per_pixel(region_scores,method=None,temperature=0.1):
    from cleanlab.internal.multilabel_scorer import softmin
    """
    Region scores - 0->1 score on a different region:
    """
    region_scores_64 = region_scores.astype("float64")
    if method=="softmin":
        return softmin(np.expand_dims(region_scores_64,axis=0), axis=1, temperature=temperature)[0]
    else:
        raise Exception("Invalid Method: Specify correct method")
    return None   


from cleanlab.segmentation.rank import find_label_issues

def get_label_quality_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    method: str = "softmin",
    batch_size: int = 10000,
    n_jobs: Optional[int] = 1,
    verbose: bool = True,
    **kwargs
) -> np.ndarray:
    """Returns a label quality score for each image.

    This is a function to compute label quality scores for standard (multi-class) classification datasets,
    where lower scores indicate labels less likely to be correct.

    Score is between 0 and 1.

    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels : np.ndarray 
      A discrete array of noisy labels for a classification dataset, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, each pixel must be integer in 0, 1, ..., K-1.
      For a standard (multi-class) classification dataset where each example is labeled with one class,
      `labels` should be 3-D array of shape ``(N,H,W,)``. 
      
    Tip: If your labels are one hot encoded you can `np.argmax(labels_one_hot,axis=1)` assuming that `labels_one_hot` is of dimension (N,K,H,W)
    before entering in the function

    pred_probs : np.ndarray
      An array of shape ``(N,K,H,W,)`` of model-predicted class probabilities,
      ``P(label=k|x)``. Each pixel contains an array of K classes, where for 
      an example `x` the array at each pixel contains the model-predicted probabilities 
      that `x` belongs to each of the K classes.
      
    method : {"softmin", "num_pixel_issues"}, default="softmin"
      Label quality scoring method.

    batch_size : int, optional
      For num_pixel_issues:
      Size of mini-batches to use for estimating the label issues.
      To maximize efficiency, try to use the largest `batch_size` your memory allows.
      
    n_jobs: int, optional
      For num_pixel_issues:
      Number of processes for multiprocessing (default value = 1). Only used on Linux.
      If `n_jobs=None`, will use either the number of: physical cores if psutil is installed, or logical cores otherwise.
    
    verbose : bool, optional
      For num_pixel_issues:
      Whether to suppress print statements or not.
      
    **kwargs:
      downsample : int, optional
      Factor to shrink labels and pred_probs by for 'num_pixel_issues' only . Default ``16``
      Must be a factor divisible by both the labels and the pred_probs. Note that larger factors result in a linear 
      decrease in performance



    Returns
    -------
    label_quality_scores : np.ndarray
      Contains one score (between 0 and 1) per image.
      Lower scores indicate more likely mislabeled examples.

    """
        
    def check_input(labels: np.ndarray, pred_probs: np.ndarray) -> None:
        if len(labels.shape) != 3:
            raise ValueError("labels must have a shape of (N, H, W)")

        if len(pred_probs.shape) != 4:
            raise ValueError("pred_probs must have a shape of (N, K, H, W)")

        num_images, height, width = labels.shape
        num_images_pred, num_classes, height_pred, width_pred = pred_probs.shape

        if num_images != num_images_pred or height != height_pred or width != width_pred:
            raise ValueError("labels and pred_probs must have matching dimensions for N, H, and W")

    check_input(labels, pred_probs)

    softmin_temperature = kwargs.get("temperature", 0.1)
    downsample_num_pixel_issues = kwargs.get("downsample", 16)

    if method == "num_pixel_issues":
        return find_label_issues(labels, pred_probs, downsample=downsample_num_pixel_issues, n_jobs=n_jobs, scores_only=True, verbose=verbose,)

    if downsample_num_pixel_issues != 16:
        import warnings
        warnings.warn(f"image will not downsample for method {method} is only for method: num_pixel_issues")

    num_im, num_class, h, w = pred_probs.shape
    final_scores = []

    if verbose:
        from tqdm.auto import tqdm
        pbar = tqdm(desc=f"images processed using {method}", total=num_im)

    for image in range(num_im):
        mask = [labels[image] == cls for cls in range(num_class)]
        image_scores = pred_probs[image][mask]
        final_scores.append(get_label_quality_per_pixel(np.array(image_scores), method=method, temperature=softmin_temperature))

        if verbose:
            pbar.update(1)

    return np.array(final_scores)

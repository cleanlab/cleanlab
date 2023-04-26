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
Methods to rank and score images in a semantic segmentation dataset , based on how likely they are to contain label errors.


"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple

from cleanlab.segmentation.filter import find_label_issues


def get_label_quality_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    method: str = "softmin",
    batch_size: int = 10000,
    n_jobs: Optional[int] = 1,
    verbose: bool = True,
    **kwargs
) -> Tuple[np.ndarray,np.ndarray]:
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

    image_scores: np.ndarray
        Array of shape ``(N, )`` of scores between 0 and 1, one per image in the dataset.
        Lower scores indicate image more likely to contain a label issue.
    pixel_scores:
        Array of shape ``(N,H,W)`` of scores between 0 and 1, one per pixel in the dataset.

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
            
    check_input(labels,pred_probs)
    
    softmin_temperature = kwargs.get("temperature", 0.1) 
    downsample_num_pixel_issues = kwargs.get("downsample", 16) 

    if method == "num_pixel_issues":
        _,K,_,_ = pred_probs.shape
        labels_expanded = labels[:, np.newaxis, :, :]
        mask = (np.arange(K)[np.newaxis, :, np.newaxis, np.newaxis] == labels_expanded)
        # Calculate pixel_scores
        masked_pred_probs = np.where(mask, pred_probs, 0)
        pixel_scores = masked_pred_probs.sum(axis=1)
        
        return find_label_issues(labels,pred_probs, downsample=downsample_num_pixel_issues, n_jobs=n_jobs, scores_only=True, verbose=verbose,), pixel_scores
        
    if downsample_num_pixel_issues != 16:
        import warnings
        warnings.warn(f"image will not downsample for method {method} is only for method: num_pixel_issues")
        
    num_im, num_class, h,w = pred_probs.shape
    image_scores = []
    pixel_scores = []
    if verbose:
        from tqdm.auto import tqdm
        pbar = tqdm(desc=f"images processed using {method}", total=num_im)
    for image in range(num_im):
        mask = [labels[image]==cls for cls in range(num_class)]
        image_probs = pred_probs[image][mask]
        pixel_scores.append(np.where(mask, pred_probs[image], 0).sum(axis=0))
        image_scores.append(get_label_quality_per_image(np.array(image_probs),method = method, temperature=softmin_temperature))
        if verbose:
            pbar.update(1)
    return np.array(image_scores), np.array(pixel_scores)


def get_label_quality_per_image(image_scores,method=None,temperature=0.1):
    from cleanlab.internal.multilabel_scorer import softmin
    """
    Image overall label quality scoring, curently using the "softmin" method.

    Parameters
    ----------
    image_scores:
        Per-pixel label quality scores in flattened array of shape ``(N, )``, where N is the number of pixels in the image.


    temperature:
        Temperature of the softmax function.

        Lower values encourage this method to converge toward the label quality score of the pixel with the lowest quality label in the image.

        Higher values encourage this method to converge toward the average label quality score of all pixels in the image.

    Returns
    ---------
    image_score:
        Float of the image's label quality score from 0 to 1, 0 being the lowest quality and 1 being the highest quality.

    """
    image_scores_64 = image_scores.astype("float64")
    if method=="softmin":
        return softmin(np.expand_dims(image_scores_64,axis=0), axis=1, temperature=temperature)[0]
    else:
        raise Exception("Invalid Method: Specify correct method")



def issues_from_scores(
    image_scores: np.ndarray, pixel_scores: np.ndarray = None, threshold: float = 0.1
) -> Union[list, np.ndarray]:
    """
    Converts scores output by :py:func:`segmentation.rank.get_label_quality_scores <cleanlab.segmentation.rank.get_label_quality_scores>`
    to a list of issues of similar format as output by :py:func:`segmentation.filter.find_label_issues <cleanlab.segmentation.filter.find_label_issues>`.

    Issues are sorted by label quality score, from most to least severe.

    Only considers as issues those tokens with label quality score lower than `threshold`,
    so this parameter determines the number of issues that are returned.
    This method is intended for converting the most severely mislabeled examples to a format compatible with
    ``summary`` methods like :py:func:`segmentation.summary.display_issues <cleanlab.segmentation.summary.display_issues>`.
    This method does not estimate the number of label errors since the `threshold` is arbitrary,
    for that instead use :py:func:`segmentation.filter.find_label_issues <cleanlab.segmentation.filter.find_label_issues>`,
    which estimates the label errors via Confident Learning rather than score thresholding.

    Parameters
    ----------
    image_scores:
        Array of shape `(N, )` of overall image scores, where `N` is the number of images in the dataset.

        Same format as the `image_scores` returned by :py:func:`segmentation.rank.get_label_quality_scores <cleanlab.segmentation.rank.get_label_quality_scores>`.

    pixel_scores:
        Array of shape ``(N,H,W)`` of scores between 0 and 1, one per pixel in the dataset.

        Same format as the `pixel_scores` returned by :py:func:`segmentation.rank.get_label_quality_scores <cleanlab.segmentation.rank.get_label_quality_scores>`.

    threshold:
        Pixels with quality scores above the `threshold` are not
        included in the result.

    Returns
    ---------
    issues : np.ndarray
      Returns a boolean **mask** for the entire dataset
      where ``True`` represents a pixel label issue and ``False`` represents an example that is
      accurately labeled with using the threshold provided by the user.
      Use :py:func:`segmentation.summary.display_issues <cleanlab.segmentation.summary.display_issues>`
      to view these issues within the original images.
      
      If `pixel_scores` is not provided, returns array of integer indices (rather than boolean mask) of the images whose label quality score
        falls below the `threshold` (also sorted by overall label quality score of each image).

    """
    if pixel_scores is not None:
        issues = np.where(pixel_scores < threshold, True, False)
        num_issues = np.sum(issues, axis=(1, 2))
        ranking = np.argsort(num_issues)
        return issues[ranking]

    else:
        ranking = np.argsort(image_scores)
        cutoff = 0
        while image_scores[ranking[cutoff]] < threshold and cutoff < len(ranking)-1:
            cutoff += 1
        return ranking[:cutoff]




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
Methods to find label issues in semantic segmentation datasets, where each pixel in a image receives its own class label.

"""

from cleanlab.experimental.label_issues_batched import find_label_issues_batched
import numpy as np 
from typing import Optional, List, Any, Tuple


def find_label_issues(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    downsample: int = 16,
    batch_size: int = 10000,
    n_jobs: Optional[int] = 1,
    verbose: bool = True,
    **kwargs) -> np.ndarray:
    """
    Returns a boolean mask for the entire dataset, per pixel where ``True`` represents
    an example identified with a label issue and ``False`` represents an example of a pixel correctly labeled.

    N - Number of images in the dataset
    K - Number of classes in the dataset
    H - Height of each image
    W - Width of each image

    Tip: if you encounter the error "pred_probs is not defined", try setting
    ``n_jobs=1``.

    Parameters
    ----------
    labels : np.ndarray 
      A discrete array of shape ``(N,H,W,)`` of noisy labels for a classification dataset, i.e. some labels may be erroneous.
      *Format requirements*: for a dataset with K classes, each pixel must be integer in 0, 1, ..., K-1.
      
    Tip: If your labels are one hot encoded you can `np.argmax(labels_one_hot,axis=1)` assuming that `labels_one_hot` is of dimension (N,K,H,W)
    before entering in the function

    pred_probs : np.ndarray
      An array of shape ``(N,K,H,W,)`` of model-predicted class probabilities,
      ``P(label=k|x)``. Each pixel contains an array of K classes, where for 
      an example `x` the array at each pixel contains the model-predicted probabilities 
      that `x` belongs to each of the K classes. The columns must be ordered such that these probabilities 
      correspond to class 0, 1, ..., K-1.
      
    downsample : int, optional
      Factor to shrink labels and pred_probs by. Default ``16``
      Must be a factor divisible by both the labels and the pred_probs. Note that larger factors result in a linear 
      decrease in performance

    batch_size : int, optional
      Size of mini-batches to use for estimating the label issues.
      To maximize efficiency, try to use the largest `batch_size` your memory allows.
      
    n_jobs: int, optional
      Number of processes for multiprocessing (default value = 1). Only used on Linux.
      If `n_jobs=None`, will use either the number of: physical cores if psutil is installed, or logical cores otherwise.
    
    verbose : bool, optional
      Set to ``False`` to suppress all print statements.

    **kwargs: 
      scores_only: optional
      Set to True to return a score for each image. Meant for internal call in 
      ``cleanlab.semantic_segmentation.rank.get_label_quality_scores``

    
    Returns
    -------
    label_issues : np.ndarray
      Returns a boolean **mask** for the entire dataset of length `(N,H,W)`
      where ``True`` represents a pixel label issue and ``False`` represents an example that is correctly labeled.
      

    """
    scores_only = kwargs.get("scores_only", False)
    
    def downsample_arrays(labels: np.ndarray, pred_probs: np.ndarray, factor: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if factor == 1:
            return labels, pred_probs
        
        num_image, num_classes, h, w = pred_probs.shape

        #Check if possible to downsample
        if h%downsample!=0 or w%downsample!=0:
            raise ValueError(f"Height {h} and width {w} not divisible by downsample value of {downsample}")
        small_labels = np.round(labels.reshape((num_image, h // factor, factor,
                                                w // factor, factor)).mean(4).mean(2))
        small_pred_probs = pred_probs.reshape((num_image, num_classes, h // factor, factor,
                                               w // factor, factor)).mean(5).mean(3)

        # We want to make sure that pred_probs are renormalized
        row_sums = small_pred_probs.sum(axis=1)
        renorm_small_pred_probs = (small_pred_probs / np.expand_dims(row_sums, 1))

        return small_labels, renorm_small_pred_probs
    

    def flatten_and_preprocess_masks(labels: np.ndarray, pred_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, num_classes, _, _ = pred_probs.shape
        labels_flat = labels.flatten().astype(int)
        pred_probs_flat = np.moveaxis(pred_probs, 0, 1).reshape(num_classes, -1)

        return labels_flat, pred_probs_flat.T
    
    
    ##
    _check_input(labels, pred_probs)
    
    
    #Added Downsampling
    pre_labels, pre_pred_probs = downsample_arrays(labels, pred_probs, downsample)
    
    num_image, num_classes, h, w = pre_pred_probs.shape
    #flatten images just preps labels and pred_probs
    
    pre_labels, pre_pred_probs = flatten_and_preprocess_masks(pre_labels, pre_pred_probs)
    
    ranked_label_issues = find_label_issues_batched(pre_labels, pre_pred_probs, batch_size=batch_size, n_jobs=n_jobs,verbose=verbose)

    
    #Finding the right indicies
    relative_index = ranked_label_issues % (h*w)
    pixel_coor_i,pixel_coor_j  = np.unravel_index(relative_index, (h,w))
    image_number = ranked_label_issues//(h*w)
    
    if scores_only:
        return 1 - np.bincount(image_number)/(h*w)

    #Upsample carefully maintaining indicies
    image = np.full((num_image, h, w), False)

    for num,ii,jj in zip(image_number,pixel_coor_i, pixel_coor_j):
        #only want to call it an error if pred_probs doesnt match the label at that pixel
        image[num,ii,jj]=True 
    
    if downsample==1:
        return image
    else:
        image = image.repeat(downsample, axis = 1).repeat(downsample, axis = 2)
        
        for num, ii, jj in zip(image_number, pixel_coor_i, pixel_coor_j):
            # Upsample the coordinates
            upsampled_ii = ii * downsample
            upsampled_jj = jj * downsample
            # Iterate over the upsampled region
            for row in range(upsampled_ii, upsampled_ii + downsample):
                for col in range(upsampled_jj, upsampled_jj + downsample):
                    # Check if the predicted class (argmax) at the identified issue location matches the true label
                    if np.argmax(pred_probs[num, :, row, col]) == labels[num, row, col]:
                        # If they match, set the corresponding entry in the image array to False
                        image[num, row, col] = False

        return image
    
def _check_input(labels: np.ndarray, pred_probs: np.ndarray) -> None:
        """
        Checks that the input labels and predicted probabilities are valid.

        Parameters
        ----------
        labels:
            Array of shape ``(N, H, W)`` of integer labels, where `N` is the number of images in the dataset and `H` and `W` are the height and width of the images.

        pred_probs:
            Array of shape ``(N, K, H, W)`` of predicted probabilities, where `N` is the number of images in the dataset, `K` is the number of classes, and `H` and `W` are the height and width of the images.
        """
        if len(labels.shape) != 3:
            raise ValueError("labels must have a shape of (N, H, W)")

        if len(pred_probs.shape) != 4:
            raise ValueError("pred_probs must have a shape of (N, K, H, W)")

        num_images, height, width = labels.shape
        num_images_pred, num_classes, height_pred, width_pred = pred_probs.shape

        if num_images != num_images_pred or height != height_pred or width != width_pred:
            raise ValueError("labels and pred_probs must have matching dimensions for N, H, and W")
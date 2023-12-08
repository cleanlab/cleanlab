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
Methods to rank and score images in a semantic segmentation dataset based on how likely they are to contain mislabeled pixels.
"""
import numpy as np
import warnings
from typing import Optional, Union, Tuple
from cleanlab.segmentation.filter import find_label_issues

from cleanlab.internal.segmentation_utils import _get_valid_optional_params, _check_input


def get_label_quality_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    method: str = "softmin",
    batch_size: Optional[int] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a label quality score for each image.

    This is a function to compute label quality scores for semantic segmentation datasets,
    where lower scores indicate labels less likely to be correct.

    * N - Number of images in the dataset
    * K - Number of classes in the dataset
    * H - Height of each image
    * W - Width of each image

    Parameters
    ----------
    labels:
      A discrete array of noisy labels for a segmantic segmentation dataset, in the shape ``(N,H,W,)``,
      where each pixel must be integer in 0, 1, ..., K-1.
      Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.segmentation.filter.find_label_issues>` for further details.

    pred_probs:
      An array of shape ``(N,K,H,W,)`` of model-predicted class probabilities.
      Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.segmentation.filter.find_label_issues>` for further details.

    method: {"softmin", "num_pixel_issues"}, default="softmin"
      Label quality scoring method.

      - "softmin" - Calculates the inner product between scores and softmax(1-scores). For efficiency, use instead of "num_pixel_issues".
      - "num_pixel_issues" - Uses the number of pixels with label issues for each image using :py:func:`find_label_issues <cleanlab.segmentation.filter.find_label_issues>`

    batch_size :
      Optional size of mini-batches to use for estimating the label issues for 'num_pixel_issues' only, not 'softmin'.
      To maximize efficiency, try to use the largest `batch_size` your memory allows. If not provided, a good default is used.

    n_jobs:
      Optional number of processes for multiprocessing (default value = 1). Only used on Linux. For 'num_pixel_issues' only, not 'softmin'
      If `n_jobs=None`, will use either the number of: physical cores if psutil is installed, or logical cores otherwise.

    verbose:
      Set to ``False`` to suppress all print statements.

    **kwargs:
      * downsample : int,
          Factor to shrink labels and pred_probs by for 'num_pixel_issues' only, not 'softmin' . Default ``16``
          Must be a factor divisible by both the labels and the pred_probs. Larger values of `downsample` produce faster runtimes but potentially less accurate results due to over-compression. Set to 1 to avoid any downsampling.
      * temperature : float,
          Temperature for softmin. Default ``0.1``


    Returns
    -------
    image_scores:
        Array of shape ``(N, )`` of scores between 0 and 1, one per image in the dataset.
        Lower scores indicate image more likely to contain a label issue.
    pixel_scores:
        Array of shape ``(N,H,W)`` of scores between 0 and 1, one per pixel in the dataset.
    """
    batch_size, n_jobs = _get_valid_optional_params(batch_size, n_jobs)
    _check_input(labels, pred_probs)

    softmin_temperature = kwargs.get("temperature", 0.1)
    downsample_num_pixel_issues = kwargs.get("downsample", 1)

    if method == "num_pixel_issues":
        _, K, _, _ = pred_probs.shape
        labels_expanded = labels[:, np.newaxis, :, :]
        mask = np.arange(K)[np.newaxis, :, np.newaxis, np.newaxis] == labels_expanded
        # Calculate pixel_scores
        masked_pred_probs = np.where(mask, pred_probs, 0)
        pixel_scores = masked_pred_probs.sum(axis=1)
        scores = find_label_issues(
            labels,
            pred_probs,
            downsample=downsample_num_pixel_issues,
            n_jobs=n_jobs,
            verbose=verbose,
            batch_size=batch_size,
        )
        img_scores = 1 - np.mean(scores, axis=(1, 2))
        return (img_scores, pixel_scores)

    if downsample_num_pixel_issues != 1:
        warnings.warn(
            f"image will not downsample for method {method} is only for method: num_pixel_issues"
        )

    num_im, num_class, h, w = pred_probs.shape
    image_scores = []
    pixel_scores = []
    if verbose:
        from tqdm.auto import tqdm

        pbar = tqdm(desc=f"images processed using {method}", total=num_im)
    for image in range(num_im):
        image_probs = pred_probs[image][
            labels[image],
            np.arange(h)[:, None],
            np.arange(w),
        ]
        pixel_scores.append(image_probs)
        image_scores.append(
            _get_label_quality_per_image(
                np.array(image_probs.flatten()), method=method, temperature=softmin_temperature
            )
        )
        if verbose:
            pbar.update(1)
    return np.array(image_scores), np.array(pixel_scores)


def issues_from_scores(
    image_scores: np.ndarray, pixel_scores: Optional[np.ndarray] = None, threshold: float = 0.1
) -> Union[list, np.ndarray]:
    """
    Converts scores output by `~cleanlab.segmentation.rank.get_label_quality_scores`
    to a list of issues of similar format as output by :py:func:`segmentation.filter.find_label_issues <cleanlab.segmentation.filter.find_label_issues>`.

    Only considers as issues those tokens with label quality score lower than `threshold`,
    so this parameter determines the number of issues that are returned.

    Note
    ----
    - This method is intended for converting the most severely mislabeled examples into a format compatible with ``summary`` methods like :py:func:`segmentation.summary.display_issues <cleanlab.segmentation.summary.display_issues>`.
    - This method does not estimate the number of label errors since the `threshold` is arbitrary, for that instead use :py:func:`segmentation.filter.find_label_issues <cleanlab.segmentation.filter.find_label_issues>`, which estimates the label errors via Confident Learning rather than score thresholding.

    Parameters
    ----------
    image_scores:
      Array of shape `(N, )` of overall image scores, where `N` is the number of images in the dataset.
      Same format as the `image_scores` returned by `~cleanlab.segmentation.rank.get_label_quality_scores`.

    pixel_scores:
      Optional array of shape ``(N,H,W)`` of scores between 0 and 1, one per pixel in the dataset.
      Same format as the `pixel_scores` returned by `~cleanlab.segmentation.rank.get_label_quality_scores`.

    threshold:
        Optional quality scores threshold that determines which pixels are included in result. Pixels with with quality scores above the `threshold` are not
        included in the result. If not provided, all pixels are included in result.

    Returns
    ---------
    issues:
      Returns a boolean **mask** for the entire dataset
      where ``True`` represents a pixel label issue and ``False`` represents an example that is
      accurately labeled with using the threshold provided by the user.
      Use :py:func:`segmentation.summary.display_issues <cleanlab.segmentation.summary.display_issues>`
      to view these issues within the original images.

      If `pixel_scores` is not provided, returns array of integer indices (rather than boolean mask) of the images whose label quality score
      falls below the `threshold` (sorted by overall label quality score of each image).

    """

    if image_scores is None:
        raise ValueError("pixel_scores must be provided")
    if threshold < 0 or threshold > 1 or threshold is None:
        raise ValueError("threshold must be between 0 and 1")

    if pixel_scores is not None:
        issues = np.where(pixel_scores < threshold, True, False)
    else:
        ranking = np.argsort(image_scores)
        cutoff = np.searchsorted(image_scores[ranking], threshold)
        issues = ranking[: cutoff + 1]
    return issues


def _get_label_quality_per_image(pixel_scores, method=None, temperature=0.1):
    from cleanlab.internal.multilabel_scorer import softmin

    """
    Input pixel scores and get label quality score for that image, currently using the "softmin" method.

    Parameters
    ----------
    pixel_scores:
        Per-pixel label quality scores in flattened array of shape ``(N, )``, where N is the number of pixels in the image.

    method: default "softmin"
        Method to use to calculate the image's label quality score.
        Currently only supports "softmin".
    temperature: default 0.1
        Temperature of the softmax function. Too small values may cause numerical underflow and NaN scores.

        Lower values encourage this method to converge toward the label quality score of the pixel with the lowest quality label in the image.

        Higher values encourage this method to converge toward the average label quality score of all pixels in the image.

    Returns
    ---------
    image_score:
        Float of the image's label quality score from 0 to 1, 0 being the lowest quality and 1 being the highest quality.

    """
    if pixel_scores is None or pixel_scores.size == 0:
        raise Exception("Invalid Input: pixel_scores cannot be None or an empty list")

    if temperature == 0 or temperature is None:
        raise Exception("Invalid Input: temperature cannot be zero or None")

    pixel_scores_64 = pixel_scores.astype("float64")
    if method == "softmin":
        if len(pixel_scores_64) > 0:
            return softmin(
                np.expand_dims(pixel_scores_64, axis=0), axis=1, temperature=temperature
            )[0]
        else:
            raise Exception("Invalid Input: pixel_scores is empty")
    else:
        raise Exception("Invalid Method: Specify correct method. Currently only supports 'softmin'")

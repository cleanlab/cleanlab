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
Methods to find label issues in image semantic segmentation datasets, where each pixel in an image receives its own class label.

"""

from cleanlab.experimental.label_issues_batched import LabelInspector
import numpy as np
from typing import Tuple, Optional

from cleanlab.internal.segmentation_utils import _get_valid_optional_params, _check_input


def find_label_issues(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    batch_size: Optional[int] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Returns a boolean mask for the entire dataset, per pixel where ``True`` represents
    an example identified with a label issue and ``False`` represents an example of a pixel correctly labeled.

    * N - Number of images in the dataset
    * K - Number of classes in the dataset
    * H - Height of each image
    * W - Width of each image

    Tip
    ---
    If you encounter the error "pred_probs is not defined", try setting ``n_jobs=1``.

    Parameters
    ----------
    labels:
      A discrete array of shape ``(N,H,W,)`` of noisy labels for a semantic segmentation dataset, i.e. some labels may be erroneous.

      *Format requirements*: For a dataset with K classes, each pixel must be labeled using an integer in 0, 1, ..., K-1.

      Tip
      ---
      If your labels are one hot encoded you can do: ``labels = np.argmax(labels_one_hot, axis=1)`` assuming that `labels_one_hot` is of dimension ``(N,K,H,W)``, in order to get properly formatted `labels`.

    pred_probs:
      An array of shape ``(N,K,H,W,)`` of model-predicted class probabilities,
      ``P(label=k|x)`` for each pixel ``x``. The prediction for each pixel is an array corresponding to the estimated likelihood that this pixel belongs to each of the ``K`` classes. The 2nd dimension of `pred_probs` must be ordered such that these probabilities correspond to class 0, 1, ..., K-1.

    batch_size:
      Optional size of image mini-batches used for computing the label issues in a streaming fashion (does not affect results, just the runtime and memory requirements).
      To maximize efficiency, try to use the largest `batch_size` your memory allows. If not provided, a good default is used.

    n_jobs:
      Optional number of processes for multiprocessing (default value = 1). Only used on Linux.
      If `n_jobs=None`, will use either the number of: physical cores if psutil is installed, or logical cores otherwise.

    verbose:
      Set to ``False`` to suppress all print statements.

    **kwargs:
      * downsample: int,
        Optional factor to shrink labels and pred_probs by. Default ``1``
        Must be a factor divisible by both the labels and the pred_probs. Larger values of `downsample` produce faster runtimes but potentially less accurate results due to over-compression. Set to 1 to avoid any downsampling.

    Returns
    -------
    label_issues: np.ndarray
      Returns a boolean **mask** for the entire dataset of length `(N,H,W)`
      where ``True`` represents a pixel label issue and ``False`` represents an example that is correctly labeled.
    """
    batch_size, n_jobs = _get_valid_optional_params(batch_size, n_jobs)
    downsample = kwargs.get("downsample", 1)

    def downsample_arrays(
        labels: np.ndarray, pred_probs: np.ndarray, factor: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        if factor == 1:
            return labels, pred_probs

        num_image, num_classes, h, w = pred_probs.shape

        # Check if possible to downsample
        if h % downsample != 0 or w % downsample != 0:
            raise ValueError(
                f"Height {h} and width {w} not divisible by downsample value of {downsample}. Set kwarg downsample to 1 to avoid downsampling."
            )
        small_labels = np.round(
            labels.reshape((num_image, h // factor, factor, w // factor, factor)).mean(4).mean(2)
        )
        small_pred_probs = (
            pred_probs.reshape((num_image, num_classes, h // factor, factor, w // factor, factor))
            .mean(5)
            .mean(3)
        )

        # We want to make sure that pred_probs are renormalized
        row_sums = small_pred_probs.sum(axis=1)
        renorm_small_pred_probs = small_pred_probs / np.expand_dims(row_sums, 1)

        return small_labels, renorm_small_pred_probs

    def flatten_and_preprocess_masks(
        labels: np.ndarray, pred_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, num_classes, _, _ = pred_probs.shape
        labels_flat = labels.flatten().astype(int)
        pred_probs_flat = np.moveaxis(pred_probs, 0, 1).reshape(num_classes, -1)

        return labels_flat, pred_probs_flat.T

    ##
    _check_input(labels, pred_probs)

    # Added Downsampling
    pre_labels, pre_pred_probs = downsample_arrays(labels, pred_probs, downsample)

    num_image, _, h, w = pre_pred_probs.shape

    ### This section is a modified version of find_label_issues_batched(), old code is commented out
    # ranked_label_issues = find_label_issues_batched(
    #     pre_labels, pre_pred_probs, batch_size=batch_size, n_jobs=n_jobs, verbose=verbose
    # )
    lab = LabelInspector(
        num_class=pre_pred_probs.shape[1],
        verbose=verbose,
        n_jobs=n_jobs,
        quality_score_kwargs=None,
        num_issue_kwargs=None,
    )
    n = len(pre_labels)

    if verbose:
        from tqdm.auto import tqdm

        pbar = tqdm(desc="number of examples processed for estimating thresholds", total=n)

    # Precompute the size of each image in the batch
    image_size = np.prod(pre_pred_probs.shape[1:])
    images_per_batch = max(batch_size // image_size, 1)

    for start_index in range(0, n, images_per_batch):
        end_index = min(start_index + images_per_batch, n)
        labels_batch, pred_probs_batch = flatten_and_preprocess_masks(
            pre_labels[start_index:end_index], pre_pred_probs[start_index:end_index]
        )
        lab.update_confident_thresholds(labels_batch, pred_probs_batch)
        if verbose:
            pbar.update(end_index - start_index)

    if verbose:
        pbar.close()
        pbar = tqdm(desc="number of examples processed for checking labels", total=n)

    for start_index in range(0, n, images_per_batch):
        end_index = min(start_index + images_per_batch, n)
        labels_batch, pred_probs_batch = flatten_and_preprocess_masks(
            pre_labels[start_index:end_index], pre_pred_probs[start_index:end_index]
        )
        _ = lab.score_label_quality(labels_batch, pred_probs_batch)
        if verbose:
            pbar.update(end_index - start_index)

    if verbose:
        pbar.close()

    ranked_label_issues = lab.get_label_issues()
    ### End find_label_issues_batched() section

    # Finding the right indicies
    relative_index = ranked_label_issues % (h * w)
    pixel_coor_i, pixel_coor_j = np.unravel_index(relative_index, (h, w))
    image_number = ranked_label_issues // (h * w)

    # Upsample carefully maintaining indicies
    label_issues = np.full((num_image, h, w), False)

    for num, ii, jj in zip(image_number, pixel_coor_i, pixel_coor_j):
        # only want to call it an error if pred_probs doesnt match the label at that pixel
        label_issues[num, ii, jj] = True
        if downsample == 1:
            # check if pred_probs matches the label at that pixel
            if np.argmax(pred_probs[num, :, ii, jj]) == labels[num, ii, jj]:
                label_issues[num, ii, jj] = False

    if downsample != 1:
        label_issues = label_issues.repeat(downsample, axis=1).repeat(downsample, axis=2)

        for num, ii, jj in zip(image_number, pixel_coor_i, pixel_coor_j):
            # Upsample the coordinates
            upsampled_ii = ii * downsample
            upsampled_jj = jj * downsample
            # Iterate over the upsampled region
            for row in range(upsampled_ii, upsampled_ii + downsample):
                for col in range(upsampled_jj, upsampled_jj + downsample):
                    # Check if the predicted class (argmax) at the identified issue location matches the true label
                    if np.argmax(pred_probs[num, :, row, col]) == labels[num, row, col]:
                        # If they match, set the corresponding entry in the label_issues array to False
                        label_issues[num, row, col] = False

    return label_issues

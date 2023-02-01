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
Implementation of :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`
that does not need much memory by operating in mini-batches.
The recommended usage is two passes over your data: 
one pass to compute `confident_thresholds`,another to evaluate each label.
To maximize efficiency, try to use the largest batch_sizes your memory allows.
To reduce runtime ~50%, you can run the first pass on a subset of your dataset
as long as it contains enough data from each class to estimate `confident_thresholds` accurately.

To run this, either follow the examples script,
or use the ``find_label_issues_batched()`` convenience function defined in this module.

In the examples script below:
- `labels` is a (big) 1D ``np.ndarray`` of class labels represented as integers in ``0,1,...,K-1``.
- ``pred_probs`` = is a (big) 2D ``np.ndarray`` of predicted class probabilities,
where each row is an example, each column represents a class.

`labels` and `pred_probs` can be stored in a file instead where you load chunks of them at a time.
Methods to load arrays in chunks include: ``np.load(...,mmap_mode='r')``, ``numpy.memmap()``,
HDF5 or Zarr files, see: https://pythonspeed.com/articles/mmap-vs-zarr-hdf5/

Examples
--------
>>> n = len(labels)
>>> batch_size = 10000  # you can change this in between batches, set as big as your RAM allows
>>> lab = LabelInspector(num_class = pred_probs.shape[1])
>>> # First compute confident thresholds (for faster results, can also do this on a random subset of your data):
>>> i = 0
>>> while i < n:
>>>     end_index = i + batch_size
>>>     labels_batch = labels[i:end_index]
>>>     pred_probs_batch = pred_probs[i:end_index,:]
>>>     i = end_index
>>>     lab.update_confident_thresholds(labels_batch, pred_probs_batch)
>>> # See what we calculated:
>>> confident_thresholds = lab.get_confident_thresholds()
>>> # Evaluate the quality of the labels (run this on full dataset you want to evaluate):
>>> i = 0
>>> while i < n:
>>>     end_index = i + batch_size
>>>     labels_batch = labels[i:end_index]
>>>     pred_probs_batch = pred_probs[i:end_index,:]
>>>     i = end_index
>>>     batch_results = lab.score_label_quality(labels_batch, pred_probs_batch)
>>> # Indices of examples with label issues, sorted by label quality score (most severe to least severe):
>>> indices_of_examples_with_issues = lab.get_label_issues()
>>> # If your `pred_probs` and `labels` are arrays already in memory,
>>> # then you can use this shortcut for all of the above:
>>> indices_of_examples_with_issues = find_label_issues_batched(labels, pred_probs, batch_size=10000)
"""

import numpy as np
from typing import List, Optional

from cleanlab.count import get_confident_thresholds
from cleanlab.rank import get_label_quality_scores, find_top_issues
from cleanlab.typing import LabelLike
from cleanlab.internal.util import value_counts_fill_missing_classes


EPS = 1e-6  # small number


class LabelInspector:
    """
    Class for finding label issues in big datasets where memory becomes a problem for other cleanlab methods.
    Only create one such object per dataset and do not try to use the same ``LabelInspector`` across 2 datasets.
    For efficiency, this class does little input checking.
    You can first run :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`
    on a small subset of your data to verify your inputs are properly formatted.
    Do NOT modify any of the attributes of this class yourself!
    Multi-label classification is not supported by this class, it is only for multi-class classification.

    Parameters
    ----------
    num_class : int
      The number of classes in your multi-class classification task.

    store_results : bool, optional
      Whether this object will store all label quality scores.
      Set this to False if you encounter memory problems even for small batch sizes (~1000).
      If ``False``, you can still identify the label issues yourself by aggregating
      the label quality scores for each batch, sorting them across all batches, and returning the top ``T`` indices
      with ``T = self.get_num_issues()``.

    verbose : bool, optional
      Whether to suppress print statements or not.

    quality_score_kwargs : dict, optional
      Keyword arguments to pass into :py:func:`rank.get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.

    num_issue_kwargs : dict, optional
      Keyword arguments to :py:func:`count.num_label_issues()` <cleanlab.count.num_label_issues>`
      to control estimation of the number of label issues.
      The only supported kwarg here for now is: `estimation_method`.
    """

    def __init__(
        self,
        *,
        num_class: int,
        store_results: bool = True,
        verbose: bool = True,
        quality_score_kwargs: Optional[dict] = None,
        num_issue_kwargs: Optional[dict] = None,
    ):
        if quality_score_kwargs is None:
            quality_score_kwargs = {}
        if num_issue_kwargs is None:
            num_issue_kwargs = {}

        self.num_class = num_class
        self.store_results = store_results
        self.verbose = verbose
        self.quality_score_kwargs = quality_score_kwargs  # extra arguments for ``rank.get_label_quality_scores()`` to control label quality scoring
        self.num_issue_kwargs = num_issue_kwargs  # extra arguments for ``count.num_label_issues()`` to control estimation of the number of label issues (only supported argument for now is: `estimation_method`).
        self.off_diagonal_calibrated = False
        if num_issue_kwargs.get("estimation_method") == "off_diagonal_calibrated":
            # store extra attributes later needed for calibration:
            self.off_diagonal_calibrated = True
            self.prune_counts = np.zeros(self.num_class)
            self.class_counts = np.zeros(self.num_class)
            self.normalization = np.zeros(self.num_class)
        else:
            self.prune_count = 0  # number of label issues estimated based on data seen so far (only used when estimation_method is not calibrated)

        if self.store_results:
            self.label_quality_scores = []

        self.confident_thresholds = np.zeros(
            (num_class,)
        )  # current estimate of thresholds based on data seen so far
        self.examples_per_class = np.zeros(
            (num_class,)
        )  # current counts of examples with each given label seen so far
        self.examples_processed_thresh = (
            0  # number of examples seen so far for estimating thresholds
        )
        self.examples_processed_quality = 0  # number of examples seen so far for estimating label quality and number of label issues

    def get_confident_thresholds(self) -> np.ndarray:
        """
        Fetches already-computed confident thresholds from the data seen so far
        in same format as: :py:func:`count.get_confident_thresholds <cleanlab.count.get_confident_thresholds>`.

        Returns
        -------
        confident_thresholds : np.ndarray
          An array of shape ``(K, )`` where ``K`` is the number of classes.
        """
        if self.examples_processed_thresh < 1:
            raise ValueError(
                "Have not computed any confident_thresholds yet. Call `update_confident_thresholds()` first."
            )
        else:
            if self.verbose:
                print(
                    f"Total number of examples used to estimate confident thresholds: {self.examples_processed_thresh}"
                )
            return self.confident_thresholds

    def get_num_issues(self) -> int:
        """
        Fetches already-computed estimate of the number of label issues in the data seen so far
        in the same format as: :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>`.

        Note: The estimated number of issues may differ from :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>`
        by 1 due to rounding differences.

        Returns
        -------
        num_issues :
          The estimated number of examples with label issues in the data seen so far.
        """
        if self.examples_processed_quality < 1:
            raise ValueError(
                "Have not evaluated any labels yet. Call `score_label_quality()` first."
            )
        else:
            if self.verbose:
                print(
                    f"Total number of examples whose labels have been evaluated: {self.examples_processed_quality}"
                )
            if self.off_diagonal_calibrated:
                calibrated_prune_counts = (
                    self.prune_counts
                    * self.class_counts
                    / np.clip(self.normalization, a_min=EPS, a_max=None)
                )  # avoid division by 0
                return np.floor(np.sum(calibrated_prune_counts)).astype("int")
            else:  # not calibrated
                return self.prune_count

    def get_quality_scores(self) -> np.ndarray:
        """
        Fetches already-computed estimate of the label quality of each example seen so far
        in the same format as: :py:func:`rank.get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.

        Returns
        -------
        label_quality_scores : np.ndarray
          Contains one score (between 0 and 1) per example seen so far.
          Lower scores indicate more likely mislabeled examples.
        """
        if not self.store_results:
            raise ValueError(
                "Must initialize the LabelInspector with `store_results` == True. "
                "Otherwise you can assemble the label quality scores yourself based on "
                "the scores returned for each batch of data from `score_label_quality()`"
            )
        else:
            return np.asarray(self.label_quality_scores)

    def get_label_issues(self) -> np.ndarray:
        """
        Fetches already-computed estimate of indices of examples with label issues in the data seen so far,
        in the same format as: :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`
        with its `return_indices_ranked_by` argument specified.

        Note: this method corresponds to ``filter.find_label_issues(..., filter_by=METHOD1, return_indices_ranked_by=METHOD2)
        where by default: ``METHOD1="low_self_confidence"``, ``METHOD2="self_confidence"``
        or if this object was instantiated with ``quality_score_kwargs = {"method": "normalized_margin"}`` then we instead have:
        ``METHOD1="low_normalized_margin"``, ``METHOD2="normalized_margin"``.

        Note: The estimated number of issues may differ from :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`
        by 1 due to rounding differences.

        Returns
        -------
        issue_indices : np.ndarray
          Indices of examples with label issues, sorted by label quality score.
        """
        if not self.store_results:
            raise ValueError(
                "Must initialize the LabelInspector with `store_results` == True. "
                "Otherwise you can identify label issues yourself based on the scores from all "
                "the batches of data and the total number of issues returned by `get_num_issues()`"
            )
        if self.examples_processed_quality < 1:
            raise ValueError(
                "Have not evaluated any labels yet. Call `score_label_quality()` first."
            )
        if self.verbose:
            print(
                f"Total number of examples whose labels have been evaluated: {self.examples_processed_quality}"
            )
        return find_top_issues(self.get_quality_scores(), top=self.get_num_issues())

    def update_confident_thresholds(self, labels: LabelLike, pred_probs: np.ndarray):
        """
        Updates the estimate of confident_thresholds stored in this class using a new batch of data.
        Inputs should be in same format as for: :py:func:`count.get_confident_thresholds <cleanlab.count.get_confident_thresholds>`.

        Parameters
        ----------
        labels: np.ndarray or list
          Given class labels for each example in the batch.

        pred_probs: np.ndarray
          2D array of model-predicted class probabilities for each example in the batch.
        """
        labels = _batch_check(labels, pred_probs, self.num_class)
        batch_size = len(labels)
        batch_thresholds = get_confident_thresholds(
            labels, pred_probs
        )  # values for missing classes may exceed 1 but should not matter since we multiply by this class counts in the batch
        batch_class_counts = value_counts_fill_missing_classes(labels, num_classes=self.num_class)
        self.confident_thresholds = (
            self.examples_per_class * self.confident_thresholds
            + batch_class_counts * batch_thresholds
        ) / np.clip(
            self.examples_per_class + batch_class_counts, a_min=1, a_max=None
        )  # avoid division by 0
        self.examples_per_class += batch_class_counts
        self.examples_processed_thresh += batch_size

    def score_label_quality(self, labels: LabelLike, pred_probs: np.ndarray) -> np.ndarray:
        """
        Updates the estimate of confident_thresholds stored in this class using a new batch of data.
        Inputs should be in same format as for: :py:func:`rank.get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`..

        Parameters
        ----------
        labels: np.ndarray or list
          Given class labels for each example in the batch.

        pred_probs: np.ndarray
          2D array of model-predicted class probabilities for each example in the batch of data.

        Returns
        -------
        label_quality_scores : np.ndarray
          Contains one score (between 0 and 1) for each example in the batch of data.
        """
        labels = _batch_check(labels, pred_probs, self.num_class)
        batch_size = len(labels)
        scores = get_label_quality_scores(
            labels, pred_probs, **self.quality_score_kwargs
        )  # TODO: adjusted scores are messed up because threshold arent being passed.
        class_counts = value_counts_fill_missing_classes(labels, num_classes=self.num_class)
        self._update_num_label_issues(labels, pred_probs, **self.num_issue_kwargs)
        self.examples_processed_quality += batch_size
        if self.store_results:
            self.label_quality_scores += list(scores)

        return scores

    def _update_num_label_issues(self, labels: LabelLike, pred_probs: np.ndarray, **kwargs):
        """
        Update the estimate of num_label_issues stored in this class using a new batch of data.
        Kwargs are ignored here for now (included for forwards compatibility).
        Instead of being specified here, `estimation_method` should be declared when this class is initialized.
        """
        if self.examples_processed_thresh < 1:
            raise ValueError(
                "Have not computed any confident_thresholds yet. Call `update_confident_thresholds()` first."
            )

        pred_class = np.argmax(pred_probs, axis=1)
        batch_size = len(labels)
        pred_confidence = pred_probs[np.arange(batch_size), pred_class]
        # add margin for floating point comparison operations:
        adj_confident_thresholds = self.confident_thresholds - EPS
        pred_gt_thresholds = pred_probs >= adj_confident_thresholds
        max_ind = np.argmax(pred_probs * pred_gt_thresholds, axis=1)
        if not self.off_diagonal_calibrated:
            prune_count_batch = np.sum(
                (pred_probs[np.arange(batch_size), max_ind] >= adj_confident_thresholds[max_ind])
                & (max_ind != labels)
                & (pred_class != labels)
            )
            self.prune_count += prune_count_batch
        else:  # calibrated
            self.class_counts += value_counts_fill_missing_classes(
                labels, num_classes=self.num_class
            )
            to_increment = (
                pred_probs[np.arange(batch_size), max_ind] >= adj_confident_thresholds[max_ind]
            )
            for class_label in range(self.num_class):
                labels_equal_to_class = labels == class_label
                self.normalization[class_label] += np.sum(labels_equal_to_class & to_increment)
                self.prune_counts[class_label] += np.sum(
                    labels_equal_to_class
                    & to_increment
                    & (max_ind != labels)
                    # & (pred_class != labels)
                    # This is not applied in num_label_issues(..., estimation_method="off_diagonal_custom"). Do we want to add it?
                )


def _batch_check(labels: LabelLike, pred_probs: np.ndarray, num_class: int) -> np.ndarray:
    """
    Basic checks to ensure batch of data looks ok. For efficiency, this check is quite minimal.

    Returns
    -------
    labels : np.ndarray
      `labels` formatted as a 1D array.
    """
    batch_size = pred_probs.shape[0]
    labels = np.asarray(labels)
    if batch_size < 10:
        raise ValueError("Please run this with batches containing at least 10 examples.")
    if len(labels) != batch_size:
        raise ValueError("labels and pred_probs must have same length")
    if pred_probs.shape[1] != num_class:
        raise ValueError("num_class must equal pred_probs.shape[1]")

    return labels


def find_label_issues_batched(
    labels: LabelLike,
    pred_probs: np.ndarray,
    *,
    batch_size: int = 10000,
    verbose: bool = True,
    quality_score_kwargs: Optional[dict] = None,
    num_issue_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """
    Variant of :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`
    that requires less memory by reading in `pred_probs`, `labels` from (already loaded) arrays in mini-batches.
    If you want to read from file in mini-batches instead of from already loaded arrays,
    see the example usage of the ``LabelInspector`` class.
    This function basically implements the example ``LabelInspector`` usage script,
    but you can further customize that script by running it yourself.
    See the documentation of ``LabelInspector`` to learn more about how this method works.

    Parameters
    ----------
    labels: np.ndarray or list
      Given class labels for each example in the dataset.

    pred_probs: np.ndarray
      2D array of model-predicted class probabilities for each example in the dataset.

    batch_size : int, optional
      Size of mini-batches to use for estimating the label issues.
      To maximize efficiency, try to use the largest `batch_size` your memory allows.

    verbose : bool, optional
      Whether to suppress print statements or not.

    quality_score_kwargs : dict, optional
      Keyword arguments to pass into :py:func:`rank.get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.

    num_issue_kwargs : dict, optional
      Keyword arguments to :py:func:`count.num_label_issues()` <cleanlab.count.num_label_issues>`
      to control estimation of the number of label issues.
      The only supported kwarg here for now is: `estimation_method`.

    Returns
    -------
    issue_indices : np.ndarray
      Indices of examples with label issues, sorted by label quality score.
    """
    lab = LabelInspector(
        num_class=pred_probs.shape[1],
        verbose=verbose,
        quality_score_kwargs=quality_score_kwargs,
        num_issue_kwargs=num_issue_kwargs,
    )
    n = len(labels)
    i = 0
    while i < n:
        end_index = i + batch_size
        labels_batch = labels[i:end_index]
        pred_probs_batch = pred_probs[i:end_index, :]
        i = end_index
        lab.update_confident_thresholds(labels_batch, pred_probs_batch)

    confident_thresholds = lab.get_confident_thresholds()  # just to see what we calculated

    # Next evaluate the quality of the labels (run this on full dataset you want to evaluate):
    i = 0
    while i < n:
        end_index = i + batch_size
        labels_batch = labels[i:end_index]
        pred_probs_batch = pred_probs[i:end_index, :]
        i = end_index
        batch_results = lab.score_label_quality(labels_batch, pred_probs_batch)

    return lab.get_label_issues()

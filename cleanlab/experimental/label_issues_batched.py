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
You can also use this approach to estimate label quality scores or the number of label issues
for big datasets with limited memory.

With default settings, the results returned from this approach closely approximate those returned from:
``cleanlab.filter.find_label_issues(..., filter_by="low_self_confidence", return_indices_ranked_by="self_confidence")``

To run this approach, either use the ``find_label_issues_batched()`` convenience function defined in this module,
or follow the examples script for the ``LabelInspector`` class if you require greater customization.
"""

import numpy as np
from typing import Optional, List, Tuple, Any

from cleanlab.count import get_confident_thresholds, _reduce_issues
from cleanlab.rank import find_top_issues, _compute_label_quality_scores
from cleanlab.typing import LabelLike
from cleanlab.internal.util import value_counts_fill_missing_classes
from cleanlab.internal.constants import (
    CONFIDENT_THRESHOLDS_LOWER_BOUND,
    FLOATING_POINT_COMPARISON,
    CLIPPING_LOWER_BOUND,
)

import platform
import multiprocessing as mp

try:
    import psutil

    PSUTIL_EXISTS = True
except ImportError:  # pragma: no cover
    PSUTIL_EXISTS = False

# global variable for multiproc on linux
adj_confident_thresholds_shared: np.ndarray
labels_shared: LabelLike
pred_probs_shared: np.ndarray


def find_label_issues_batched(
    labels: Optional[LabelLike] = None,
    pred_probs: Optional[np.ndarray] = None,
    *,
    labels_file: Optional[str] = None,
    pred_probs_file: Optional[str] = None,
    batch_size: int = 10000,
    n_jobs: Optional[int] = 1,
    verbose: bool = True,
    quality_score_kwargs: Optional[dict] = None,
    num_issue_kwargs: Optional[dict] = None,
    return_mask: bool = False,
) -> np.ndarray:
    """
    Variant of :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`
    that requires less memory by reading from `pred_probs`, `labels` in mini-batches.
    To avoid loading big `pred_probs`, `labels` arrays into memory,
    provide these as memory-mapped objects like Zarr arrays or memmap arrays instead of regular numpy arrays.
    See: https://pythonspeed.com/articles/mmap-vs-zarr-hdf5/

    With default settings, the results returned from this method closely approximate those returned from:
    ``cleanlab.filter.find_label_issues(..., filter_by="low_self_confidence", return_indices_ranked_by="self_confidence")``

    This function internally implements the example usage script of the ``LabelInspector`` class,
    but you can further customize that script by running it yourself instead of this function.
    See the documentation of ``LabelInspector`` to learn more about how this method works internally.

    Parameters
    ----------
    labels: np.ndarray-like object, optional
      1D array of given class labels for each example in the dataset, (int) values in ``0,1,2,...,K-1``.
      To avoid loading big objects into memory, you should pass this as a memory-mapped object like:
      Zarr array loaded with ``zarr.convenience.open(YOURFILE.zarr, mode="r")``,
      or memmap array loaded with ``np.load(YOURFILE.npy, mmap_mode="r")``.

      Tip: You can save an existing numpy array to Zarr via: ``zarr.convenience.save_array(YOURFILE.zarr, your_array)``,
      or to .npy file that can be loaded with mmap via: ``np.save(YOURFILE.npy, your_array)``.

    pred_probs: np.ndarray-like object, optional
      2D array of model-predicted class probabilities (floats) for each example in the dataset.
      To avoid loading big objects into memory, you should pass this as a memory-mapped object like:
      Zarr array loaded with ``zarr.convenience.open(YOURFILE.zarr, mode="r")``
      or memmap array loaded with ``np.load(YOURFILE.npy, mmap_mode="r")``.

    labels_file: str, optional
      Specify this instead of `labels` if you want this method to load from file for you into a memmap array.
      Path to .npy file where the entire 1D `labels` numpy array is stored on disk (list format is not supported).
      This is loaded using: ``np.load(labels_file, mmap_mode="r")``
      so make sure this file was created via: ``np.save()`` or other compatible methods (.npz not supported).

    pred_probs_file: str, optional
      Specify this instead of `pred_probs` if you want this method to load from file for you into a memmap array.
      Path to .npy file where the entire `pred_probs` numpy array is stored on disk.
      This is loaded using: ``np.load(pred_probs_file, mmap_mode="r")``
      so make sure this file was created via: ``np.save()`` or other compatible methods (.npz not supported).

    batch_size : int, optional
      Size of mini-batches to use for estimating the label issues.
      To maximize efficiency, try to use the largest `batch_size` your memory allows.

    n_jobs: int, optional
      Number of processes for multiprocessing (default value = 1). Only used on Linux.
      If `n_jobs=None`, will use either the number of: physical cores if psutil is installed, or logical cores otherwise.

    verbose : bool, optional
      Whether to suppress print statements or not.

    quality_score_kwargs : dict, optional
      Keyword arguments to pass into :py:func:`rank.get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.

    num_issue_kwargs : dict, optional
      Keyword arguments to :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>`
      to control estimation of the number of label issues.
      The only supported kwarg here for now is: `estimation_method`.
    return_mask : bool, optional
       Determines what is returned by this method: If `return_mask=True`, return a boolean mask.
       If `False`, return a list of indices specifying examples with label issues, sorted by label quality score.

    Returns
    -------
    label_issues : np.ndarray
      If `return_mask` is `True`, returns a boolean **mask** for the entire dataset
      where ``True`` represents a label issue and ``False`` represents an example that is
      accurately labeled with high confidence.
      If `return_mask` is `False`, returns an array containing **indices** of examples identified to have
      label issues (i.e. those indices where the mask would be ``True``), sorted by likelihood that the corresponding label is correct.
    --------
    >>> batch_size = 10000  # for efficiency, set this to as large of a value as your memory can handle
    >>> # Just demonstrating how to save your existing numpy labels, pred_probs arrays to compatible .npy files:
    >>> np.save("LABELS.npy", labels_array)
    >>> np.save("PREDPROBS.npy", pred_probs_array)
    >>> # You can load these back into memmap arrays via: labels = np.load("LABELS.npy", mmap_mode="r")
    >>> # and then run this method on the memmap arrays, or just run it directly on the .npy files like this:
    >>> issues = find_label_issues_batched(labels_file="LABELS.npy", pred_probs_file="PREDPROBS.npy", batch_size=batch_size)
    >>> # This method also works with Zarr arrays:
    >>> import zarr
    >>> # Just demonstrating how to save your existing numpy labels, pred_probs arrays to compatible .zarr files:
    >>> zarr.convenience.save_array("LABELS.zarr", labels_array)
    >>> zarr.convenience.save_array("PREDPROBS.zarr", pred_probs_array)
    >>> # You can load from such files into Zarr arrays:
    >>> labels = zarr.convenience.open("LABELS.zarr", mode="r")
    >>> pred_probs = zarr.convenience.open("PREDPROBS.zarr", mode="r")
    >>> # This method can be directly run on Zarr arrays, memmap arrays, or regular numpy arrays:
    >>> issues = find_label_issues_batched(labels=labels, pred_probs=pred_probs, batch_size=batch_size)
    """
    if labels_file is not None:
        if labels is not None:
            raise ValueError("only specify one of: `labels` or `labels_file`")
        if not isinstance(labels_file, str):
            raise ValueError(
                "labels_file must be str specifying path to .npy file containing the array of labels"
            )
        labels = np.load(labels_file, mmap_mode="r")
        assert isinstance(labels, np.ndarray)

    if pred_probs_file is not None:
        if pred_probs is not None:
            raise ValueError("only specify one of: `pred_probs` or `pred_probs_file`")
        if not isinstance(pred_probs_file, str):
            raise ValueError(
                "pred_probs_file must be str specifying path to .npy file containing 2D array of pred_probs"
            )
        pred_probs = np.load(pred_probs_file, mmap_mode="r")
        assert isinstance(pred_probs, np.ndarray)
        if verbose:
            print(
                f"mmap-loaded numpy arrays have: {len(pred_probs)} examples, {pred_probs.shape[1]} classes"
            )
    if labels is None:
        raise ValueError("must provide one of: `labels` or `labels_file`")
    if pred_probs is None:
        raise ValueError("must provide one of: `pred_probs` or `pred_probs_file`")

    assert pred_probs is not None
    if len(labels) != len(pred_probs):
        raise ValueError(
            f"len(labels)={len(labels)} does not match len(pred_probs)={len(pred_probs)}. Perhaps an issue loading mmap numpy arrays from file."
        )
    lab = LabelInspector(
        num_class=pred_probs.shape[1],
        verbose=verbose,
        n_jobs=n_jobs,
        quality_score_kwargs=quality_score_kwargs,
        num_issue_kwargs=num_issue_kwargs,
    )
    n = len(labels)
    if verbose:
        from tqdm.auto import tqdm

        pbar = tqdm(desc="number of examples processed for estimating thresholds", total=n)
    i = 0
    while i < n:
        end_index = i + batch_size
        labels_batch = labels[i:end_index]
        pred_probs_batch = pred_probs[i:end_index, :]
        i = end_index
        lab.update_confident_thresholds(labels_batch, pred_probs_batch)
        if verbose:
            pbar.update(batch_size)

    # Next evaluate the quality of the labels (run this on full dataset you want to evaluate):
    if verbose:
        pbar.close()
        pbar = tqdm(desc="number of examples processed for checking labels", total=n)
    i = 0
    while i < n:
        end_index = i + batch_size
        labels_batch = labels[i:end_index]
        pred_probs_batch = pred_probs[i:end_index, :]
        i = end_index
        _ = lab.score_label_quality(labels_batch, pred_probs_batch)
        if verbose:
            pbar.update(batch_size)

    if verbose:
        pbar.close()

    label_issues_indices = lab.get_label_issues()
    label_issues_mask = np.zeros(len(labels), dtype=bool)
    label_issues_mask[label_issues_indices] = True
    mask = _reduce_issues(pred_probs=pred_probs, labels=labels)
    label_issues_mask[mask] = False
    if return_mask:
        return label_issues_mask
    return np.where(label_issues_mask)[0]


class LabelInspector:
    """
    Class for finding label issues in big datasets where memory becomes a problem for other cleanlab methods.
    Only create one such object per dataset and do not try to use the same ``LabelInspector`` across 2 datasets.
    For efficiency, this class does little input checking.
    You can first run :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`
    on a small subset of your data to verify your inputs are properly formatted.
    Do NOT modify any of the attributes of this class yourself!
    Multi-label classification is not supported by this class, it is only for multi-class classification.

    The recommended usage demonstrated in the examples script below involves two passes over your data:
    one pass to compute `confident_thresholds`, another to evaluate each label.
    To maximize efficiency, try to use the largest batch_size your memory allows.
    To reduce runtime further, you can run the first pass on a subset of your dataset
    as long as it contains enough data from each class to estimate `confident_thresholds` accurately.

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

    Parameters
    ----------
    num_class : int
      The number of classes in your multi-class classification task.

    store_results : bool, optional
      Whether this object will store all label quality scores, a 1D array of shape ``(N,)``
      where ``N`` is the total number of examples in your dataset.
      Set this to False if you encounter memory problems even for small batch sizes (~1000).
      If ``False``, you can still identify the label issues yourself by aggregating
      the label quality scores for each batch, sorting them across all batches, and returning the top ``T`` indices
      with ``T = self.get_num_issues()``.

    verbose : bool, optional
      Whether to suppress print statements or not.

    n_jobs: int, optional
      Number of processes for multiprocessing (default value = 1). Only used on Linux.
      If `n_jobs=None`, will use either the number of: physical cores if psutil is installed, or logical cores otherwise.

    quality_score_kwargs : dict, optional
      Keyword arguments to pass into :py:func:`rank.get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.

    num_issue_kwargs : dict, optional
      Keyword arguments to :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>`
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
        n_jobs: Optional[int] = 1,
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
            self.label_quality_scores: List[float] = []

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
        # Determine number of cores for multiprocessing:
        self.n_jobs: Optional[int] = None
        os_name = platform.system()
        if os_name != "Linux":
            self.n_jobs = 1
            if n_jobs is not None and n_jobs != 1 and self.verbose:
                print(
                    "n_jobs is overridden to 1 because multiprocessing is only supported for Linux."
                )
        elif n_jobs is not None:
            self.n_jobs = n_jobs
        else:
            if PSUTIL_EXISTS:
                self.n_jobs = psutil.cpu_count(logical=False)  # physical cores
            if not self.n_jobs:
                # switch to logical cores
                self.n_jobs = mp.cpu_count()
                if self.verbose:
                    print(
                        f"Multiprocessing will default to using the number of logical cores ({self.n_jobs}). To default to number of physical cores: pip install psutil"
                    )

    def get_confident_thresholds(self, silent: bool = False) -> np.ndarray:
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
            if self.verbose and not silent:
                print(
                    f"Total number of examples used to estimate confident thresholds: {self.examples_processed_thresh}"
                )
            return self.confident_thresholds

    def get_num_issues(self, silent: bool = False) -> int:
        """
        Fetches already-computed estimate of the number of label issues in the data seen so far
        in the same format as: :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>`.

        Note: The estimated number of issues may differ from :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>`
        by 1 due to rounding differences.

        Returns
        -------
        num_issues : int
          The estimated number of examples with label issues in the data seen so far.
        """
        if self.examples_processed_quality < 1:
            raise ValueError(
                "Have not evaluated any labels yet. Call `score_label_quality()` first."
            )
        else:
            if self.verbose and not silent:
                print(
                    f"Total number of examples whose labels have been evaluated: {self.examples_processed_quality}"
                )
            if self.off_diagonal_calibrated:
                calibrated_prune_counts = (
                    self.prune_counts
                    * self.class_counts
                    / np.clip(self.normalization, a_min=CLIPPING_LOWER_BOUND, a_max=None)
                )  # avoid division by 0
                return np.rint(np.sum(calibrated_prune_counts)).astype("int")
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

        Note: this method corresponds to ``filter.find_label_issues(..., filter_by=METHOD1, return_indices_ranked_by=METHOD2)``
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
        return find_top_issues(self.get_quality_scores(), top=self.get_num_issues(silent=True))

    def update_confident_thresholds(self, labels: LabelLike, pred_probs: np.ndarray):
        """
        Updates the estimate of confident_thresholds stored in this class using a new batch of data.
        Inputs should be in same format as for: :py:func:`count.get_confident_thresholds <cleanlab.count.get_confident_thresholds>`.

        Parameters
        ----------
        labels: np.ndarray or list
          Given class labels for each example in the batch, values in ``0,1,2,...,K-1``.

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
        self.confident_thresholds = np.clip(
            self.confident_thresholds, a_min=CONFIDENT_THRESHOLDS_LOWER_BOUND, a_max=None
        )
        self.examples_per_class += batch_class_counts
        self.examples_processed_thresh += batch_size

    def score_label_quality(
        self,
        labels: LabelLike,
        pred_probs: np.ndarray,
        *,
        update_num_issues: bool = True,
    ) -> np.ndarray:
        """
        Scores the label quality of each example in the provided batch of data,
        and also updates the number of label issues stored in this class.
        Inputs should be in same format as for: :py:func:`rank.get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.

        Parameters
        ----------
        labels: np.ndarray
          Given class labels for each example in the batch, values in ``0,1,2,...,K-1``.

        pred_probs: np.ndarray
          2D array of model-predicted class probabilities for each example in the batch of data.

        update_num_issues: bool, optional
          Whether or not to update the number of label issues or only compute label quality scores.
          For lower runtimes, set this to ``False`` if you only want to score label quality and not find label issues.

        Returns
        -------
        label_quality_scores : np.ndarray
          Contains one score (between 0 and 1) for each example in the batch of data.
        """
        labels = _batch_check(labels, pred_probs, self.num_class)
        batch_size = len(labels)
        scores = _compute_label_quality_scores(
            labels,
            pred_probs,
            confident_thresholds=self.get_confident_thresholds(silent=True),
            **self.quality_score_kwargs,
        )
        class_counts = value_counts_fill_missing_classes(labels, num_classes=self.num_class)
        if update_num_issues:
            self._update_num_label_issues(labels, pred_probs, **self.num_issue_kwargs)
        self.examples_processed_quality += batch_size
        if self.store_results:
            self.label_quality_scores += list(scores)

        return scores

    def _update_num_label_issues(
        self,
        labels: LabelLike,
        pred_probs: np.ndarray,
        **kwargs,
    ):
        """
        Update the estimate of num_label_issues stored in this class using a new batch of data.
        Kwargs are ignored here for now (included for forwards compatibility).
        Instead of being specified here, `estimation_method` should be declared when this class is initialized.
        """

        # whether to match the output of count.num_label_issues exactly
        # default is False, which gives significant speedup on large batches
        # and empirically matches num_label_issues even on input sizes of
        # 1M x 10k
        thorough = False
        if self.examples_processed_thresh < 1:
            raise ValueError(
                "Have not computed any confident_thresholds yet. Call `update_confident_thresholds()` first."
            )

        if self.n_jobs == 1:
            adj_confident_thresholds = self.confident_thresholds - FLOATING_POINT_COMPARISON
            pred_class = np.argmax(pred_probs, axis=1)
            batch_size = len(labels)
            if thorough:
                # add margin for floating point comparison operations:
                pred_gt_thresholds = pred_probs >= adj_confident_thresholds
                max_ind = np.argmax(pred_probs * pred_gt_thresholds, axis=1)
                if not self.off_diagonal_calibrated:
                    mask = (max_ind != labels) & (pred_class != labels)
                else:
                    # calibrated
                    # should we change to above?
                    mask = pred_class != labels
            else:
                max_ind = pred_class
                mask = pred_class != labels

            if not self.off_diagonal_calibrated:
                prune_count_batch = np.sum(
                    (
                        pred_probs[np.arange(batch_size), max_ind]
                        >= adj_confident_thresholds[max_ind]
                    )
                    & mask
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
        else:  # multiprocessing implementation
            global adj_confident_thresholds_shared
            adj_confident_thresholds_shared = self.confident_thresholds - FLOATING_POINT_COMPARISON

            global labels_shared, pred_probs_shared
            labels_shared = labels
            pred_probs_shared = pred_probs

            # good values for this are ~1000-10000 in benchmarks where pred_probs has 1B entries:
            processes = 5000
            if len(labels) <= processes:
                chunksize = 1
            else:
                chunksize = len(labels) // processes
            inds = split_arr(np.arange(len(labels)), chunksize)

            if thorough:
                use_thorough = np.ones(len(inds), dtype=bool)
            else:
                use_thorough = np.zeros(len(inds), dtype=bool)
            args = zip(inds, use_thorough)
            with mp.Pool(self.n_jobs) as pool:
                if not self.off_diagonal_calibrated:
                    prune_count_batch = np.sum(
                        np.asarray(list(pool.imap_unordered(_compute_num_issues, args)))
                    )
                    self.prune_count += prune_count_batch
                else:
                    results = list(pool.imap_unordered(_compute_num_issues_calibrated, args))
                    for result in results:
                        class_label = result[0]
                        self.class_counts[class_label] += 1
                        self.normalization[class_label] += result[1]
                        self.prune_counts[class_label] += result[2]


def split_arr(arr: np.ndarray, chunksize: int) -> List[np.ndarray]:
    """
    Helper function to split array into chunks for multiprocessing.
    """
    return np.split(arr, np.arange(chunksize, arr.shape[0], chunksize), axis=0)


def _compute_num_issues(arg: Tuple[np.ndarray, bool]) -> int:
    """
    Helper function for `_update_num_label_issues` multiprocessing without calibration.
    """
    ind = arg[0]
    thorough = arg[1]
    label = labels_shared[ind]
    pred_prob = pred_probs_shared[ind, :]
    pred_class = np.argmax(pred_prob, axis=-1)
    batch_size = len(label)

    if thorough:
        pred_gt_thresholds = pred_prob >= adj_confident_thresholds_shared
        max_ind = np.argmax(pred_prob * pred_gt_thresholds, axis=-1)
        prune_count_batch = np.sum(
            (pred_prob[np.arange(batch_size), max_ind] >= adj_confident_thresholds_shared[max_ind])
            & (max_ind != label)
            & (pred_class != label)
        )
    else:
        prune_count_batch = np.sum(
            (
                pred_prob[np.arange(batch_size), pred_class]
                >= adj_confident_thresholds_shared[pred_class]
            )
            & (pred_class != label)
        )
    return prune_count_batch


def _compute_num_issues_calibrated(arg: Tuple[np.ndarray, bool]) -> Tuple[Any, int, int]:
    """
    Helper function for `_update_num_label_issues` multiprocessing with calibration.
    """
    ind = arg[0]
    thorough = arg[1]
    label = labels_shared[ind]
    pred_prob = pred_probs_shared[ind, :]
    batch_size = len(label)

    pred_class = np.argmax(pred_prob, axis=-1)
    if thorough:
        pred_gt_thresholds = pred_prob >= adj_confident_thresholds_shared
        max_ind = np.argmax(pred_prob * pred_gt_thresholds, axis=-1)
        to_inc = (
            pred_prob[np.arange(batch_size), max_ind] >= adj_confident_thresholds_shared[max_ind]
        )

        prune_count_batch = to_inc & (max_ind != label)
        normalization_batch = to_inc
    else:
        to_inc = (
            pred_prob[np.arange(batch_size), pred_class]
            >= adj_confident_thresholds_shared[pred_class]
        )
        normalization_batch = to_inc
        prune_count_batch = to_inc & (pred_class != label)

    return (label, normalization_batch, prune_count_batch)


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
    if len(labels) != batch_size:
        raise ValueError("labels and pred_probs must have same length")
    if pred_probs.shape[1] != num_class:
        raise ValueError("num_class must equal pred_probs.shape[1]")

    return labels

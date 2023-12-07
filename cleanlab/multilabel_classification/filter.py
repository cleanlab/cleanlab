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
Methods to flag which examples have label issues in multi-label classification datasets.
Here each example can belong to one or more classes, or none of the classes at all.
Unlike in standard multi-class classification, model-predicted class probabilities need not sum to 1 for each row in multi-label classification.
"""

import warnings
import inspect
from typing import Optional, Union, Tuple, List, Any
import numpy as np


def find_label_issues(
    labels: list,
    pred_probs: np.ndarray,
    return_indices_ranked_by: Optional[str] = None,
    rank_by_kwargs={},
    filter_by: str = "prune_by_noise_rate",
    frac_noise: float = 1.0,
    num_to_remove_per_class: Optional[List[int]] = None,
    min_examples_per_class=1,
    confident_joint: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    low_memory: bool = False,
) -> np.ndarray:
    """
    Identifies potentially mislabeled examples in a multi-label classification dataset.
    An example is flagged as with a label issue if *any* of the classes appear to be incorrectly annotated for this example.

    Parameters
    ----------
    labels : List[List[int]]
      List of noisy labels for multi-label classification where each example can belong to multiple classes.
      This is an iterable of iterables where the i-th element of `labels` corresponds to a list of classes that the i-th example belongs to,
      according to the original data annotation (e.g. ``labels = [[1,2],[1],[0],..]``).
      This method will return the indices i where the inner list ``labels[i]`` is estimated to have some error.
      For a dataset with K classes, each class must be represented as an integer in 0, 1, ..., K-1 within the labels.

    pred_probs : np.ndarray
      An array of shape ``(N, K)`` of model-predicted class probabilities.
      Each row of this matrix corresponds to an example `x`
      and contains the predicted probability that `x` belongs to each possible class,
      for each of the K classes (along its columns).
      The columns need not sum to 1 but must be ordered such that
      these probabilities correspond to class 0, 1, ..., K-1.

      Note
      ----
      Estimated label quality scores are most accurate when they are computed based on out-of-sample ``pred_probs`` from your model.
      To obtain out-of-sample predicted probabilities for every example in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`.
      This is encouraged to get better results.

    return_indices_ranked_by : {None, 'self_confidence', 'normalized_margin', 'confidence_weighted_entropy'}, default = None
      This function can return a boolean mask (if None) or an array of the example-indices with issues sorted based on the specified ranking method.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    rank_by_kwargs : dict, optional
      Optional keyword arguments to pass into scoring functions for ranking by
      label quality score (see :py:func:`rank.get_label_quality_scores
      <cleanlab.rank.get_label_quality_scores>`).

    filter_by : {'prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning', 'predicted_neq_given', 'low_normalized_margin', 'low_self_confidence'}, default='prune_by_noise_rate'
      The specific Confident Learning method to determine precisely which examples have label issues in a dataset.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    frac_noise : float, default = 1.0
      This will return the "top" frac_noise * num_label_issues estimated label errors, dependent on the filtering method used,
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    num_to_remove_per_class : array_like
      An iterable that specifies the number of mislabeled examples to return from each class.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    min_examples_per_class : int, default = 1
      The minimum number of examples required per class below which examples from this class will not be flagged as label issues.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    confident_joint : np.ndarray, optional
      An array of shape ``(K, 2, 2)`` representing a one-vs-rest formatted confident joint, as is appropriate for multi-label classification tasks.
      Entry ``(c, i, j)`` in this array is the number of examples confidently counted into a ``(class c, noisy label=i, true label=j)`` bin,
      where `i, j` are either 0 or 1 to denote whether this example belongs to class `c` or not
      (recall examples can belong to multiple classes in multi-label classification).
      The `confident_joint` can be computed using :py:func:`count.compute_confident_joint <cleanlab.count.compute_confident_joint>` with ``multi_label=True``.
      If not provided, it is computed from the given (noisy) `labels` and `pred_probs`.

    n_jobs : optional
      Number of processing threads used by multiprocessing.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    verbose : optional
      If ``True``, prints when multiprocessing happens.

    low_memory: bool, default=False
      Set as ``True`` if you have a big dataset with limited memory.
      Uses :py:func:`experimental.label_issues_batched.find_label_issues_batched <cleanlab.experimental.label_issues_batched>`

    Returns
    -------
    label_issues : np.ndarray
      If `return_indices_ranked_by` left unspecified, returns a boolean **mask** for the entire dataset
      where ``True`` represents an example suffering from some label issue and
      ``False`` represents an example that appears accurately labeled.

      If `return_indices_ranked_by` is specified, this method instead returns a list of **indices** of examples identified with
      label issues (i.e. those indices where the mask would be ``True``).
      Indices are sorted by the likelihood that *all* classes are correctly annotated for the corresponding example.

      Note
      ----
      Obtain the *indices* of examples with label issues in your dataset by setting
      `return_indices_ranked_by`.

    """
    from cleanlab.filter import _find_label_issues_multilabel

    if low_memory:
        if rank_by_kwargs:
            warnings.warn(f"`rank_by_kwargs` is not used when `low_memory=True`.")

        func_signature = inspect.signature(find_label_issues)
        default_args = {
            k: v.default
            for k, v in func_signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        arg_values = {
            "filter_by": filter_by,
            "num_to_remove_per_class": num_to_remove_per_class,
            "confident_joint": confident_joint,
            "n_jobs": n_jobs,
            "num_to_remove_per_class": num_to_remove_per_class,
            "frac_noise": frac_noise,
            "min_examples_per_class": min_examples_per_class,
        }
        for arg_name, arg_val in arg_values.items():
            if arg_val != default_args[arg_name]:
                warnings.warn(f"`{arg_name}` is not used when `low_memory=True`.")

    return _find_label_issues_multilabel(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by=return_indices_ranked_by,
        rank_by_kwargs=rank_by_kwargs,
        filter_by=filter_by,
        frac_noise=frac_noise,
        num_to_remove_per_class=num_to_remove_per_class,
        min_examples_per_class=min_examples_per_class,
        confident_joint=confident_joint,
        n_jobs=n_jobs,
        verbose=verbose,
        low_memory=low_memory,
    )


def find_multilabel_issues_per_class(
    labels: list,
    pred_probs: np.ndarray,
    return_indices_ranked_by: Optional[str] = None,
    rank_by_kwargs={},
    filter_by: str = "prune_by_noise_rate",
    frac_noise: float = 1.0,
    num_to_remove_per_class: Optional[List[int]] = None,
    min_examples_per_class=1,
    confident_joint: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    low_memory: bool = False,
) -> Union[np.ndarray, Tuple[List[np.ndarray], List[Any], List[np.ndarray]]]:
    """
    Identifies potentially bad labels for each example and each class in a multi-label classification dataset.
    Whereas `~cleanlab.multilabel_classification.filter.find_label_issues`
    estimates which examples have an erroneous annotation for *any* class, this method estimates which specific classes are incorrectly annotated as well.
    This method returns a list of size K, the number of classes in the dataset.

    Parameters
    ----------
    labels : List[List[int]]
      List of noisy labels for multi-label classification where each example can belong to multiple classes.
      Refer to documentation for this argument in `~cleanlab.multilabel_classification.filter.find_label_issues` for further details.
      This method will identify whether ``labels[i][k]`` appears correct, for every example ``i`` and class ``k``.

    pred_probs : np.ndarray
      An array of shape ``(N, K)`` of model-predicted class probabilities.
      Refer to documentation for this argument in `~cleanlab.multilabel_classification.filter.find_label_issues` for further details.

    return_indices_ranked_by : {None, 'self_confidence', 'normalized_margin', 'confidence_weighted_entropy'}, default = None
      This function can return a boolean mask (if this argument is ``None``) or a sorted array of indices based on the specified ranking method (if not ``None``).
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    rank_by_kwargs : dict, optional
      Optional keyword arguments to pass into scoring functions for ranking by.
      label quality score (see :py:func:`rank.get_label_quality_scores
      <cleanlab.rank.get_label_quality_scores>`).

    filter_by : {'prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning', 'predicted_neq_given', 'low_normalized_margin', 'low_self_confidence'}, default = 'prune_by_noise_rate'
      The specific method that can be used to filter or prune examples with label issues from a dataset.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    frac_noise : float, default = 1.0
      This will return the "top" frac_noise * num_label_issues estimated label errors, dependent on the filtering method used,
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    num_to_remove_per_class : array_like
      This parameter is an iterable that specifies the number of mislabeled examples to return from each class.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    min_examples_per_class : int, default = 1
      The minimum number of examples required per class to avoid flagging as label issues.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    confident_joint : np.ndarray, optional
      An array of shape ``(K, 2, 2)`` representing a one-vs-rest formatted confident joint.
      Refer to documentation for this argument in `~cleanlab.multilabel_classification.filter.find_label_issues` for details.

    n_jobs : optional
      Number of processing threads used by multiprocessing.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    verbose : optional
      If ``True``, prints when multiprocessing happens.

    Returns
    -------
    per_class_label_issues : list(np.ndarray)
      By default, this is a list of length K containing the examples where each class appears incorrectly annotated.
      ``per_class_label_issues[k]`` is a Boolean mask of the same length as the dataset,
      where ``True`` values indicate examples where class ``k`` appears incorrectly annotated.

      For more details, refer to `~cleanlab.multilabel_classification.filter.find_label_issues`.

      Otherwise if `return_indices_ranked_by` is not ``None``, then this method returns 3 objects (each of length K, the number of classes): `label_issues_list`, `labels_list`, `pred_probs_list`.
        - *label_issues_list*: an ordered list of indices of examples where class k appears incorrectly annotated, sorted by the likelihood that class k is correctly annotated.
        - *labels_list*: a binary one-hot representation of the original labels, useful if you want to compute label quality scores.
        - *pred_probs_list*: a one-vs-rest representation of the original predicted probabilities of shape ``(N, 2)``, useful if you want to compute label quality scores.
          ``pred_probs_list[k][i][0]`` is the estimated probability that example ``i`` belongs to class ``k``, and is equal to: ``1 - pred_probs_list[k][i][1]``.
    """
    import cleanlab.filter
    from cleanlab.internal.multilabel_utils import get_onehot_num_classes, stack_complement
    from cleanlab.experimental.label_issues_batched import find_label_issues_batched

    y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
    if return_indices_ranked_by is None:
        bissues = np.zeros(y_one.shape).astype(bool)
    else:
        label_issues_list = []
    labels_list = []
    pred_probs_list = []
    if confident_joint is not None and not low_memory:
        confident_joint_shape = confident_joint.shape
        if confident_joint_shape == (num_classes, num_classes):
            warnings.warn(
                f"The new recommended format for `confident_joint` in multi_label settings is (num_classes,2,2) as output by compute_confident_joint(...,multi_label=True). Your K x K confident_joint in the old format is being ignored."
            )
            confident_joint = None
        elif confident_joint_shape != (num_classes, 2, 2):
            raise ValueError("confident_joint should be of shape (num_classes, 2, 2)")
    for class_num, (label, pred_prob_for_class) in enumerate(zip(y_one.T, pred_probs.T)):
        pred_probs_binary = stack_complement(pred_prob_for_class)
        if low_memory:
            quality_score_kwargs = (
                {"method": return_indices_ranked_by} if return_indices_ranked_by else None
            )
            binary_label_issues = find_label_issues_batched(
                labels=label,
                pred_probs=pred_probs_binary,
                verbose=verbose,
                quality_score_kwargs=quality_score_kwargs,
                return_mask=return_indices_ranked_by is None,
            )
        else:
            if confident_joint is None:
                conf = None
            else:
                conf = confident_joint[class_num]
            if num_to_remove_per_class is not None:
                ml_num_to_remove_per_class = [num_to_remove_per_class[class_num], 0]
            else:
                ml_num_to_remove_per_class = None
            binary_label_issues = cleanlab.filter.find_label_issues(
                labels=label,
                pred_probs=pred_probs_binary,
                return_indices_ranked_by=return_indices_ranked_by,
                frac_noise=frac_noise,
                rank_by_kwargs=rank_by_kwargs,
                filter_by=filter_by,
                num_to_remove_per_class=ml_num_to_remove_per_class,
                min_examples_per_class=min_examples_per_class,
                confident_joint=conf,
                n_jobs=n_jobs,
                verbose=verbose,
            )

        if return_indices_ranked_by is None:
            bissues[:, class_num] = binary_label_issues
        else:
            label_issues_list.append(binary_label_issues)
            labels_list.append(label)
            pred_probs_list.append(pred_probs_binary)
    if return_indices_ranked_by is None:
        return bissues
    else:
        return label_issues_list, labels_list, pred_probs_list

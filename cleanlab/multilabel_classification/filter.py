import warnings
from typing import Optional, Union, Tuple, List, Any
import numpy as np


def find_label_issues(
    labels: list,
    pred_probs: np.ndarray,
    return_indices_ranked_by: Optional[str] = None,
    rank_by_kwargs={},
    filter_by: str = "prune_by_noise_rate",
    frac_noise: float = 1.0,
    num_to_remove_per_class: Optional[int] = None,
    min_examples_per_class=1,
    confident_joint: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Identifies potentially bad labels in a multi-label classification dataset using confident learning.
    An example is flagged as with a label issue if *any* of the classes appear to be incorrectly annotated for this example.

    Parameters
    ----------
    labels : List[List[int]]
      List of noisy labels for multi-label classification where each example can belong to multiple classes.
      The i-th element of `labels` corresponds to list of classes that i-th example belongs to (e.g. ``labels = [[1,2],[1],[0],..]``).

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

    return_indices_ranked_by : {None, 'self_confidence', 'normalized_margin', 'confidence_weighted_entropy'}, default=None
      This function can return a boolean mask (if None) or a sorted array of indices based on the specified ranking method.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    rank_by_kwargs : dict, optional
      Optional keyword arguments to pass into scoring functions for ranking by
      label quality score (see :py:func:`rank.get_label_quality_scores
      <cleanlab.rank.get_label_quality_scores>`).

    filter_by : {'prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning', 'predicted_neq_given',
        'low_normalized_margin', 'low_self_confidence'}, default='prune_by_noise_rate'
      The specific method that can be used to filter or prune examples with label issues from a dataset.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    frac_noise : float, default=1.0
      This will return the "top" frac_noise * num_label_issues estimated label errors, dependent on the filtering method used,
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    num_to_remove_per_class : array_like
      This parameter is an iterable that specifies the number of mislabeled examples to return from each class.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    min_examples_per_class : int, default=1
      The minimum number of examples required per class to avoid flagging as label issues.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    confident_joint : np.ndarray, optional
      An array of shape ``(K, 2, 2)`` representing a one-vs-rest formatted confident joint, as is appropriate for multi-label classification tasks.
      Entry ``(c, i, j)`` in this array is the number of examples confidently counted into a ``(class c, noisy label=i, true label=j)`` bin,
      where `i, j` are either 0 or 1 to denote whether this example belongs to class `c` or not
      (recall examples can belong to multiple classes in multi-label classification).
      The `confident_joint` can be computed using :py:func:`count.compute_confident_joint <cleanlab.count.compute_confident_joint>` with multi_label=True.
      If not provided, it is computed from the given (noisy) `labels` and `pred_probs`.

    n_jobs : optional
      Number of processing threads used by multiprocessing.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    verbose : optional
      If ``True``, prints when multiprocessing happens.

    Returns
    -------
    label_issues : np.ndarray
      If `return_indices_ranked_by` left unspecified, returns a boolean **mask** for the entire dataset
      where ``True`` represents a label issue and ``False`` represents an example that is
      accurately labeled with high confidence.

      If `return_indices_ranked_by` is specified, returns a (shorter) list of **indices** of examples identified to have
      label issues (i.e. those indices where the mask would be ``True``). Indices are sorted by likelihood that *all* classes are correctly annotated for the corresponding example.

      Note
      ----
      Obtain the *indices* of examples with label issues in your dataset by setting
      `return_indices_ranked_by`.

    """
    from cleanlab.filter import _find_label_issues_multilabel

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
    )


def find_multilabel_issues_per_class(
    labels: list,
    pred_probs: np.ndarray,
    return_indices_ranked_by: Optional[str] = None,
    rank_by_kwargs={},
    filter_by: str = "prune_by_noise_rate",
    frac_noise: float = 1.0,
    num_to_remove_per_class: Optional[int] = None,
    min_examples_per_class=1,
    confident_joint: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
) -> Union[np.ndarray, Tuple[List[np.ndarray], List[Any], List[np.ndarray]]]:
    """
    Identifies potentially bad labels for each class in a multi-label classification dataset using confident learning.
    Refer to documentation in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for further details.
    This function returns a list of label issues of size K, where K = num_classes.

    Parameters
     ----------
     labels : List[List[int]]
       List of noisy labels for multi-label classification where each example can belong to multiple classes.
       Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.multilabel_classification.filter.find_label_issues>` for further details.

    pred_probs : np.ndarray
      An array of shape ``(N, K)`` of model-predicted class probabilities.
      Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.multilabel_classification.filter.find_label_issues>` for further details.

     return_indices_ranked_by : {None, 'self_confidence', 'normalized_margin', 'confidence_weighted_entropy'}, default=None
       This function can return a boolean mask (if None) or a sorted array of indices based on the specified ranking method.
       Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

     rank_by_kwargs : dict, optional
       Optional keyword arguments to pass into scoring functions for ranking by.
       label quality score (see :py:func:`rank.get_label_quality_scores
       <cleanlab.rank.get_label_quality_scores>`).

     filter_by : {'prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning', 'predicted_neq_given',
         'low_normalized_margin', 'low_self_confidence'}, default='prune_by_noise_rate'
       The specific method that can be used to filter or prune examples with label issues from a dataset.
       Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

     frac_noise : float, default=1.0
       This will return the "top" frac_noise * num_label_issues estimated label errors, dependent on the filtering method used,
       Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

     num_to_remove_per_class : array_like
       This parameter is an iterable that specifies the number of mislabeled examples to return from each class.
       Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

     min_examples_per_class : int, default=1
       The minimum number of examples required per class to avoid flagging as label issues.
       Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

     confident_joint : np.ndarray, optional
       An array of shape ``(K, 2, 2)`` representing a one-vs-rest formatted confident joint.
       Refer to documentation for this argument in :py:func:`cleanlab.multilabel_classification.filter.find_label_issues <cleanlab.multilabel_classification.filter.find_label_issues>` for details.

    n_jobs : optional
       Number of processing threads used by multiprocessing.
       Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

     verbose : optional
       If ``True``, prints when multiprocessing happens.

     Returns
     -------
     per_class_label_issues : list(np.ndarray)
        returns a list of per class label issues, refer to :py:func:`cleanlab.multilabel_classification.filter.find_label_issues <cleanlab.multilabel_classification.filter.find_label_issues>`


       Note
       ----
       Obtain the *indices* of label issues in your dataset by setting
       `return_indices_ranked_by`.

    """
    from cleanlab.filter import find_label_issues
    from cleanlab.internal.multilabel_utils import get_onehot_num_classes, stack_complement

    y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
    if return_indices_ranked_by is None:
        bissues = np.zeros(y_one.shape).astype(bool)
    else:
        label_issues_list = []
    labels_list = []
    pred_probs_list = []
    if confident_joint is not None:
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
        if confident_joint is None:
            conf = None
        else:
            conf = confident_joint[class_num]
        binary_label_issues = find_label_issues(
            labels=label,
            pred_probs=pred_probs_binary,
            return_indices_ranked_by=return_indices_ranked_by,
            frac_noise=frac_noise,
            rank_by_kwargs=rank_by_kwargs,
            filter_by=filter_by,
            multi_label=False,
            num_to_remove_per_class=num_to_remove_per_class,
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

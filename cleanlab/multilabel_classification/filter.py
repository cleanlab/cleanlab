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
    Finds label issues in multi-label classification data where each example can belong to more than one class.

    Parameters
    ----------
    labels : List[List[int]]
      List of noisy labels for multi-label classification where each example can belong to multiple classes (e.g. ``labels = [[1,2],[1],[0],[],...]`` indicates the first example in dataset belongs to both class 1 and class 2.


    pred_probs : np.ndarray
      An array of shape ``(N, K)`` of model-predicted probabilities,
      ``P(label=k|x)``. Each row of this matrix corresponds
      to an example `x` and contains the model-predicted probabilities that
      `x` belongs to each possible class, for each of the K classes. The
      columns must be ordered such that these probabilities correspond to
      class 0, 1, ..., K-1. They need not sum to 1.0


    return_indices_ranked_by : {None, 'self_confidence', 'normalized_margin', 'confidence_weighted_entropy'}, default=None
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    rank_by_kwargs : dict, optional
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    filter_by : {'prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning', 'predicted_neq_given',
        'low_normalized_margin', 'low_self_confidence'}, default='prune_by_noise_rate'
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    frac_noise : float, default=1.0
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    num_to_remove_per_class : array_like
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    min_examples_per_class : int, default=1
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    confident_joint : np.ndarray, optional
      An array of shape ``(K, 2, 2)`` representing a one-vs-rest formatted confident joint.
      Entry ``(c, i, j)`` in this array is the number of examples confidently counted into a ``(class c, noisy label=i, true label=j)`` bin,
      where `i, j` are either 0 or 1 to denote whether this example belongs to class `c` or not
      (recall examples can belong to multiple classes in multi-label classification).
      The `confident_joint` can be computed using :py:func:`count.compute_confident_joint <cleanlab.count.compute_confident_joint>`.
      If not provided, it is computed from the given (noisy) `labels` and `pred_probs`.

    n_jobs : optional
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    verbose : optional
      If ``True``, prints when multiprocessing happens.

    Returns
    -------
    per_class_label_issues : list(np.ndarray)
      If `return_indices_ranked_by` left unspecified, returns a list of boolean **masks** for the entire dataset
      where ``True`` represents a label issue and ``False`` represents an example that is
      accurately labeled with high confidence.
      If `return_indices_ranked_by` is specified, returns a list of shorter arrays of **indices** of examples identified to have
      label issues (i.e. those indices where the mask would be ``True``), sorting by likelihood that the corresponding label is correct is not supported yet.

      Note
      ----
      Obtain the *indices* of label issues in your dataset by setting
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
    Parameters
    ----------
    labels : List[List[int]]
      List of noisy labels for multi-label classification where each example can belong to multiple classes (e.g. ``labels = [[1,2],[1],[0],[],...]`` indicates the first example in dataset belongs to both class 1 and class 2.


    pred_probs : np.ndarray
      An array of shape ``(N, K)`` of model-predicted probabilities,
      ``P(label=k|x)``. Each row of this matrix corresponds
      to an example `x` and contains the model-predicted probabilities that
      `x` belongs to each possible class, for each of the K classes. The
      columns must be ordered such that these probabilities correspond to
      class 0, 1, ..., K-1. They need not sum to 1.0


    return_indices_ranked_by : {None, 'self_confidence', 'normalized_margin', 'confidence_weighted_entropy'}, default=None
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    rank_by_kwargs : dict, optional
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    filter_by : {'prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning', 'predicted_neq_given',
        'low_normalized_margin', 'low_self_confidence'}, default='prune_by_noise_rate'
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    frac_noise : float, default=1.0
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    num_to_remove_per_class : array_like
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    min_examples_per_class : int, default=1
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    confident_joint : np.ndarray, optional
      An array of shape ``(K, 2, 2)`` representing a one-vs-rest formatted confident joint.
      Entry ``(c, i, j)`` in this array is the number of examples confidently counted into a ``(class c, noisy label=i, true label=j)`` bin,
      where `i, j` are either 0 or 1 to denote whether this example belongs to class `c` or not
      (recall examples can belong to multiple classes in multi-label classification).
      The `confident_joint` can be computed using :py:func:`count.compute_confident_joint <cleanlab.count.compute_confident_joint>`.
      If not provided, it is computed from the given (noisy) `labels` and `pred_probs`.

    n_jobs : optional
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    verbose : optional
      If ``True``, prints when multiprocessing happens.

    Returns
    -------
    per_class_label_issues : list(np.ndarray)
      If `return_indices_ranked_by` left unspecified, returns a list of boolean **masks** for the entire dataset
      where ``True`` represents a label issue and ``False`` represents an example that is
      accurately labeled with high confidence.
      If `return_indices_ranked_by` is specified, returns a list of shorter arrays of **indices** of examples identified to have
      label issues (i.e. those indices where the mask would be ``True``), sorting by likelihood that the corresponding label is correct is not supported yet.

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

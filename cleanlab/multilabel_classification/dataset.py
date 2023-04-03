import pandas as pd
import numpy as np
from cleanlab.filter import _find_multilabel_issues_per_class
from cleanlab.internal.multilabel_utils import get_onehot_num_classes
from collections import defaultdict


def common_multilabel_issues(
    labels=list,
    pred_probs=None,
    *,
    class_names=None,
    confident_joint=None,
) -> pd.DataFrame:
    """Summarizes which classes in a multi-label dataset appear most often mislabeled overall.

    This method works by providing any one (and only one) of the following inputs:

    1. ``labels`` and ``pred_probs``, or
    2. ``confident_joint``

    Only provide **exactly one of the above input options**, do not provide a combination.

    Parameters
    ----------
    labels : List[List[int]]
        Refer to documentation for this argument in :py:func:`filter._find_multilabel_issues_per_class <cleanlab.filter._find_multilabel_issues_per_class>` for further details.

    pred_probs : np.ndarray, optional
      Refer to documentation for this argument in :py:func:`filter._find_multilabel_issues_per_class <cleanlab.filter._find_multilabel_issues_per_class>` for further details.


    class_names : Iterable[str], optional
        A list or other iterable of the string class names. The list should be in the order that
        matches the label indices. So if class 0 is 'dog' and class 1 is 'cat', then
        ``class_names = ['dog', 'cat']``.

    confident_joint : np.ndarray, optional
      An array of shape ``(K, 2, 2)`` representing a one-vs-rest formatted confident joint for multi-label data,
      as returned by :py:func:`count.compute_confident_joint <cleanlab.count.compute_confident_joint>`.
      Entry ``(c, i, j)`` in this array is the number of examples confidently counted into a ``(class c, noisy label=i, true label=j)`` bin,
      where `i, j` are either 0 or 1 to denote whether this example belongs to class `c` or not
      (recall examples can belong to multiple classes in multi-label classification).


    Returns
    -------
    common_multilabel_issues : pd.DataFrame
        DataFrame where each row corresponds to a Class (specified as the row-index) with columns "In Given Label", "In Suggested Label", "Num Examples", "Issue Probability".

        * *In Given Label*: specifies whether the Class is True/False in the given label
        * *Class*: If class_names is provided, the 'Class' column of the DataFrame will have the class name,
            otherwise, the values will represent the class index.
        * *In Suggested Label*: specifies whether the Class is  True/False in the suggested label (based on model prediction)
        * *Num Examples*: Estimated number of examples with a label issue where this Class is True/False as specified "In Given Label" but cleanlab suggests it should be as specified In Suggested Label. I.e. the number of examples in your dataset where the Class was labeled as True but likely should have been False (or vice versa).
        * *Issue Probability*: This is the  *Num Examples* column divided by the total number of examples in the dataset. It corresponds to the relative overall frequency of each type of label issue in your dataset.

        By default, the rows in this DataFrame are ordered by "Issue Probability" (descending).
    """

    y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
    if class_names is None:
        class_names = list(range(num_classes))
    label_issues_list, labels_list, pred_probs_list = _find_multilabel_issues_per_class(
        labels=labels,
        pred_probs=pred_probs,
        confident_joint=confident_joint,
        return_indices_ranked_by="self_confidence",
    )

    summary_issue_counts = defaultdict(list)
    for class_num, (label, issues_for_class) in enumerate(zip(y_one.T, label_issues_list)):
        binary_label_issues = np.zeros(len(label)).astype(bool)
        binary_label_issues[issues_for_class] = True
        class_name = class_names[class_num]
        true_but_false_count = sum(np.logical_and(label == 1, binary_label_issues))
        false_but_true_count = sum(np.logical_and(label == 0, binary_label_issues))

        summary_issue_counts["Class"].append(class_name)
        summary_issue_counts["Class Index"].append(class_num)
        summary_issue_counts["In Given Label"].append(True)
        summary_issue_counts["In Suggested Label"].append(False)
        summary_issue_counts["Num Examples"].append(true_but_false_count)
        summary_issue_counts["Issue Probability"].append(true_but_false_count / len(y_one))

        summary_issue_counts["Class"].append(class_name)
        summary_issue_counts["Class Index"].append(class_num)
        summary_issue_counts["In Given Label"].append(False)
        summary_issue_counts["In Suggested Label"].append(True)
        summary_issue_counts["Num Examples"].append(false_but_true_count)
        summary_issue_counts["Issue Probability"].append(false_but_true_count / len(y_one))

    return (
        pd.DataFrame.from_dict(summary_issue_counts)
        .set_index("Class Index")
        .sort_values(by=["Issue Probability"], ascending=False)
    )

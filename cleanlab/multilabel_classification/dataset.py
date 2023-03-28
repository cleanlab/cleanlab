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
    """Returns the Tags that are often mis-labeled.


    This method provides two scores in the Pandas DataFrame that is returned:

    * **num_examples**: The number of examples where the given/predicted label differ.
    * **Issue_probability**: `(num_examples / total number of examples in the dataset`).

    This method works by providing any one (and only one) of the following inputs:

    1. ``labels`` and ``pred_probs``, or
    2. ``confident_joint``

    Only provide **exactly one of the above input options**, do not provide a combination.

    Parameters
    ----------
    labels : List[List[int]]
        Refer to documentation for this argument in filter._find_multilabel_issues_per_class() for further details.

    pred_probs : np.ndarray, optional
      Refer to documentation for this argument in filter._find_multilabel_issues_per_class() for further details.


    class_names : Iterable[str]
        A list or other iterable of the string class names. The list should be in the order that
        matches the label indices. So if class 0 is 'dog' and class 1 is 'cat', then
        ``class_names = ['dog', 'cat']``.

    Returns
    -------
    common_multilabel_issues : pd.DataFrame
        Pandas DataFrame with indexed on Tag with columns "In_Given_Label", "In_Suggested_Label", "num_examples",
        "Issue_probability"and a description of each below.
        Each row corresponds to a pair of classes.

        * *In_Given_Label*: Bool True/False
        * *In_Suggested_Label*: Bool True/False
        * *num_examples*: estimated number of issues for a different label other than the given label
        * *Issue_probability*: the *num_examples* divided by the number of examples in the dataset.

        By default, the DataFrame is ordered by "Issue_probability" descending.
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

        summary_issue_counts["Tag"].append(class_name)
        summary_issue_counts["In_Given_Label"].append(True)
        summary_issue_counts["In_Suggested_Label"].append(False)
        summary_issue_counts["num_examples"].append(true_but_false_count)
        summary_issue_counts["Issue_probability"].append(true_but_false_count / len(y_one))

        summary_issue_counts["Tag"].append(class_name)
        summary_issue_counts["In_Given_Label"].append(False)
        summary_issue_counts["In_Suggested_Label"].append(True)
        summary_issue_counts["num_examples"].append(false_but_true_count)
        summary_issue_counts["Issue_probability"].append(false_but_true_count / len(y_one))

    return (
        pd.DataFrame.from_dict(summary_issue_counts)
        .set_index("Tag")
        .sort_values(by=["Issue_probability"], ascending=False)
    )

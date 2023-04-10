import pandas as pd
import numpy as np
from typing import Optional, cast, Dict, Any  # noqa: F401
from cleanlab.multilabel_classification.filter import find_multilabel_issues_per_class
from cleanlab.internal.multilabel_utils import get_onehot_num_classes
from collections import defaultdict

from cleanlab.internal.util import get_num_classes


def common_multilabel_issues(
    labels=list,
    pred_probs=None,
    *,
    class_names=None,
    confident_joint=None,
) -> pd.DataFrame:
    """Summarizes which classes in a multi-label dataset appear most often mislabeled overall.

    Since classes are not mutually exclusive in multi-label classification, this method summarizes the label issues for each class independently of the others.
    This method works by providing ``labels``, ``pred_probs`` and an optional Confident Joint.


    Parameters
    ----------
    labels : List[List[int]]
        Refer to documentation for this argument in :py:func:`filter._find_multilabel_issues_per_class <cleanlab.filter._find_multilabel_issues_per_class>` for further details.

    pred_probs : np.ndarray
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
        DataFrame where each row corresponds to a Class with the following columns:

        * *Class Index*: The index of the Class.
        * *Class*: If `class_names` is provided, the "Class" column of the DataFrame will indicate the name of the class, otherwise this column contains integers representing the class index.
        * *In Given Label*: whether the Class is originally annotated True or False in the given label.
        * *In Suggested Label*: whether the Class should be True or False in the suggested label (based on model's prediction).
        * *Num Examples*: Number of examples flagged as a label issue where this Class is True/False "In Given Label" but cleanlab estimates the annotation should actually be as specified "In Suggested Label". I.e. the number of examples in your dataset where this Class was labeled as True but likely should have been False (or vice versa).
        * *Issue Probability*: The  *Num Examples* column divided by the total number of examples in the dataset; i.e. the relative overall frequency of each type of label issue in your dataset.

        By default, the rows in this DataFrame are ordered by "Issue Probability" (descending).
    """

    num_examples = _get_num_examples_multilabel(labels=labels, confident_joint=confident_joint)
    summary_issue_counts = defaultdict(list)
    if class_names is None:
        num_classes = get_num_classes(labels=labels, pred_probs=pred_probs, multi_label=True)
        class_names = list(range(num_classes))
    y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
    label_issues_list, labels_list, pred_probs_list = find_multilabel_issues_per_class(
        labels=labels,
        pred_probs=pred_probs,
        confident_joint=confident_joint,
        return_indices_ranked_by="self_confidence",
    )

    for class_num, (label, issues_for_class) in enumerate(zip(y_one.T, label_issues_list)):
        binary_label_issues = np.zeros(len(label)).astype(bool)
        binary_label_issues[issues_for_class] = True
        class_name = class_names[class_num]
        true_but_false_count = sum(np.logical_and(label == 1, binary_label_issues))
        false_but_true_count = sum(np.logical_and(label == 0, binary_label_issues))

        summary_issue_counts["Class Index"].append(class_num)
        summary_issue_counts["Class"].append(class_name)
        summary_issue_counts["In Given Label"].append(True)
        summary_issue_counts["In Suggested Label"].append(False)
        summary_issue_counts["Num Examples"].append(true_but_false_count)
        summary_issue_counts["Issue Probability"].append(true_but_false_count / num_examples)

        summary_issue_counts["Class"].append(class_name)
        summary_issue_counts["Class Index"].append(class_num)
        summary_issue_counts["In Given Label"].append(False)
        summary_issue_counts["In Suggested Label"].append(True)
        summary_issue_counts["Num Examples"].append(false_but_true_count)
        summary_issue_counts["Issue Probability"].append(false_but_true_count / num_examples)
    return (
        pd.DataFrame.from_dict(summary_issue_counts)
        .sort_values(by=["Issue Probability"], ascending=False)
        .reset_index(drop=True)
    )


def rank_classes_by_multilabel_quality(
    labels=None,
    pred_probs=None,
    *,
    class_names=None,
    joint=None,
    confident_joint=None,
) -> pd.DataFrame:
    """
    Returns a Pandas DataFrame with three overall class label quality scores summarizing all examples annotated with each  class in a multi-label classification dataset.
    Details about each score are listed below under the Returns parameter. By default, classes are ordered
    by "Label Quality Score", ascending, so the most problematic classes are reported first in the returned DataFrame.

    Score values are unnormalized and may tend to be very small. What matters is their relative
    ranking across the classes.

    This method works by providing ``labels``, ``pred_probs`` and an optional Confident Joint.



    **Parameters**: For parameter info, see the docstring of :py:func:`find_overlapping_classes <cleanlab.dataset.find_overlapping_classes>`.

    Returns
    -------
    overall_label_quality : pd.DataFrame
        Pandas DataFrame with cols "Class Index", "Label Issues", "Inverse Label Issues",
        "Label Issues", "Inverse Label Noise", "Label Quality Score",
        with a description of each of these columns below.
        The length of the DataFrame is ``num_classes`` (one row per class).
        Noise scores are between 0 and 1, where 0 implies no label issues
        in the class. The "Label Quality Score" is also between 0 and 1 where 1 implies
        perfect quality. Columns:

        * *Class Index*: The index of the class in 0, 1, ..., K-1.
        * *Class*: If `class_names` is provided, the "Class" column of the DataFrame will indicate the name of the class, otherwise this column contains integers representing the class index.
        * *Label Issues*: ``count(given_label = k, true_label != k)``, estimated number of examples in the dataset that are labeled as class k but should have a different label.
        * *Label Noise*: ``prob(true_label != k | given_label = k)``, estimated proportion of examples in the dataset that are labeled as class k but should have a different label. For each class k: this is computed by dividing the number of examples with "Label Issues" that were labeled as class k by the total number of examples labeled as class k.
        * *Label Quality Score*: ``p(true_label = k | given_label = k)``. This is the proportion of examples with given label k that have been labeled correctly, i.e. ``1 - label_noise``.
        * *Inverse Label Issues*: ``count(given_label != k, true_label = k)``, estimated number of examples in the dataset that should actually be labeled as class k but have been given another label.
        * *Inverse Label Noise*: ``prob(given_label != k | true_label = k)``, estimated proportion of examples in the dataset that should actually be labeled as class k but have been given another label.

        By default, the DataFrame is ordered by "Label Quality Score", ascending.
    """

    issues_df = common_multilabel_issues(
        labels=labels, pred_probs=pred_probs, class_names=class_names, confident_joint=joint
    )
    issues_dict = defaultdict(defaultdict)  # type: Dict[str, Any]
    num_examples = _get_num_examples_multilabel(labels=labels, confident_joint=confident_joint)
    for class_num, row in issues_df.iterrows():
        if row["In Given Label"]:
            issues_dict[row["Class Index"]]["Label Issues"] = int(
                row["Issue Probability"] * num_examples
            )
            issues_dict[row["Class Index"]]["Label Noise"] = row["Issue Probability"]
            issues_dict[row["Class Index"]]["Label Quality Score"] = (
                1 - issues_dict[row["Class Index"]]["Label Noise"]
            )
        else:
            issues_dict[row["Class Index"]]["Inverse Label Issues"] = int(
                row["Issue Probability"] * num_examples
            )
            issues_dict[row["Class Index"]]["Inverse Label Noise"] = row["Issue Probability"]

    issues_df_dict = defaultdict(list)
    for i in issues_dict:
        issues_df_dict["Class Index"].append(i)
        for j in issues_dict[i]:
            issues_df_dict[j].append(issues_dict[i][j])
    return (
        pd.DataFrame.from_dict(issues_df_dict)
        .sort_values(by="Label Quality Score", ascending=True)
        .reset_index(drop=True)
    )


def _get_num_examples_multilabel(labels=None, confident_joint: Optional[np.ndarray] = None) -> int:
    """Helper method that finds the number of examples from the parameters or throws an error
    if neither parameter is provided.

    Parameters
    ----------
    For parameter info, see the docstring of :py:func:`common_multilabel_issues <cleanlab.multilabel_classification.dataset.common_multilabel_issues>`

    Returns
    -------
    num_examples : int
        The number of examples in the dataset.

    Raises
    ------
    ValueError
        If `labels` is None."""

    if labels is None and confident_joint is None:
        raise ValueError(
            "Error: num_examples is None. You must either provide confident_joint, "
            "or provide both num_example and joint as input parameters."
        )
    _confident_joint = cast(np.ndarray, confident_joint)
    num_examples = len(labels) if labels is not None else cast(int, np.sum(_confident_joint[0]))
    return num_examples


def overall_multilabel_health_score(
    labels=None,
    pred_probs=None,
    *,
    joint=None,
    confident_joint=None,
) -> float:
    """Returns a single score between 0 and 1 measuring the overall quality of all labels in a multi-label classification dataset.
    Intuitively, the score is the average correctness of the given labels across all examples in the
    dataset. So a score of 1 suggests your data is perfectly labeled and a score of 0.5 suggests
    half of the examples in the dataset may be incorrectly labeled. Thus, a higher
    score implies a higher quality dataset.

    This method works by providing ``labels``, ``pred_probs`` and an optional Confident Joint.


    Only provide **exactly one of the above input options**, do not provide a combination.

    **Parameters**: For parameter info, see the docstring of :py:func:`find_overlapping_classes <cleanlab.dataset.find_overlapping_classes>`.

    Returns
    -------
    health_score : float
        A score between 0 and 1, where 1 implies all labels in the dataset are estimated to be correct.
        A score of 0.5 implies that half of the dataset's labels are estimated to have issues.
    """
    num_classes = get_num_classes(labels=labels, pred_probs=pred_probs, multi_label=True)
    class_names = list(range(num_classes))
    num_examples = _get_num_examples_multilabel(labels=labels, confident_joint=confident_joint)
    issues_df = common_multilabel_issues(
        labels=labels, pred_probs=pred_probs, class_names=class_names, confident_joint=joint
    )
    return sum(issues_df["Num Examples"]) / num_examples


def multilabel_health_summary(
    labels=None,
    pred_probs=None,
    *,
    class_names=None,
    num_examples=None,
    confident_joint=None,
    verbose=True,
) -> Dict:
    """Prints a health summary of your datasets including useful statistics like:

    * The classes with the most and least label issues
    * Overall data label quality health score statistics for your dataset

    This method works by providing ``labels``, ``pred_probs`` and an optional Confident Joint.

    **Parameters**: For parameter info, see the docstring of :py:func:`common_multilabel_issues <cleanlab.multilabel_classificaiton.dataset.common_multilabel_issues>`.

    Returns
    -------
    summary : dict
        A dictionary containing keys (see the corresponding functions' documentation to understand the values):

        - ``"overall_label_health_score"``, corresponding to :py:func:`overall_multilabel_health_score <cleanlab.multilabel_classification.dataset.overall_multilabel_health_score>`
        - ``"classes_by_multilabel_quality"``, corresponding to :py:func:`rank_classes_by_multilabel_quality <cleanlab.multilabel_classification.dataset.rank_classes_by_multilabel_quality>`
        - ``"common_multilabel_issues"``, corresponding to :py:func:`common_multilabel_issues <cleanlab.multilabel_classification.dataset.common_multilabel_issues>`
    """
    from cleanlab.internal.util import smart_display_dataframe

    if num_examples is None:
        num_examples = _get_num_examples_multilabel(labels=labels)

    if verbose:
        longest_line = f"|   for your dataset with {num_examples:,} examples "
        print(
            "-" * (len(longest_line) - 1)
            + "\n"
            + f"|  Generating a Cleanlab Dataset Health Summary{' ' * (len(longest_line) - 49)}|\n"
            + longest_line
            + f"|  Note, Cleanlab is not a medical doctor... yet.{' ' * (len(longest_line) - 51)}|\n"
            + "-" * (len(longest_line) - 1)
            + "\n",
        )

    df_class_label_quality = rank_classes_by_multilabel_quality(
        labels=labels,
        pred_probs=pred_probs,
        class_names=class_names,
        confident_joint=confident_joint,
    )
    if verbose:
        print("Overall Class Quality and Noise across your dataset (below)")
        print("-" * 60, "\n", flush=True)
        smart_display_dataframe(df_class_label_quality)

    df_common_issues = common_multilabel_issues(
        labels=labels,
        pred_probs=pred_probs,
        class_names=class_names,
        confident_joint=confident_joint,
    )
    if verbose:
        print(
            "\nCommon multilabel issues are" + "\n" + "-" * 83 + "\n",
            flush=True,
        )
        smart_display_dataframe(df_common_issues)
        print()

    health_score = overall_multilabel_health_score(
        labels=labels,
        pred_probs=pred_probs,
        confident_joint=confident_joint,
    )
    if verbose:
        print("\nGenerated with <3 from Cleanlab.\n")
    return {
        "overall_multilabel_health_score": health_score,
        "classes_by_multilabel_quality": df_class_label_quality,
        "common_multilabel_issues": df_common_issues,
    }

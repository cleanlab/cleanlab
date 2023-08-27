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
Methods to summarize overall labeling issues across a multi-label classification dataset.
Here each example can belong to one or more classes, or none of the classes at all.
Unlike in standard multi-class classification, model-predicted class probabilities need not sum to 1 for each row in multi-label classification.
"""

import pandas as pd
import numpy as np
from typing import Optional, cast, Dict, Any  # noqa: F401
from cleanlab.multilabel_classification.filter import (
    find_multilabel_issues_per_class,
    find_label_issues,
)
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

    Since classes are not mutually exclusive in multi-label classification, this method summarizes the label issues for each class independently of the others.

    Parameters
    ----------
    labels : List[List[int]]
       List of noisy labels for multi-label classification where each example can belong to multiple classes.
       Refer to documentation for this argument in :py:func:`multilabel_classification.filter.find_label_issues <cleanlab.multilabel_classification.filter.find_label_issues>` for further details.

    pred_probs : np.ndarray
      An array of shape ``(N, K)`` of model-predicted class probabilities.
      Refer to documentation for this argument in :py:func:`multilabel_classification.filter.find_label_issues <cleanlab.multilabel_classification.filter.find_label_issues>` for further details.

    class_names : Iterable[str], optional
        A list or other iterable of the string class names. Its order must match the label indices.
        If class 0 is 'dog' and class 1 is 'cat', then ``class_names = ['dog', 'cat']``.
        If provided, the returned DataFrame will have an extra *Class Name* column with this info.

    confident_joint : np.ndarray, optional
       An array of shape ``(K, 2, 2)`` representing a one-vs-rest formatted confident joint.
       Refer to documentation for this argument in :py:func:`multilabel_classification.filter.find_label_issues <cleanlab.multilabel_classification.filter.find_label_issues>` for details.

    Returns
    -------
    common_multilabel_issues : pd.DataFrame
        DataFrame where each row corresponds to a class summarized by the following columns:
            - *Class Name*: The name of the class if class_names is provided.
            - *Class Index*: The index of the class.
            - *In Given Label*: Whether the Class is originally annotated True or False in the given label.
            - *In Suggested Label*: Whether the Class should be True or False in the suggested label (based on model's prediction).
            - *Num Examples*: Number of examples flagged as a label issue where this Class is True/False "In Given Label" but cleanlab estimates the annotation should actually be as specified "In Suggested Label". I.e. the number of examples in your dataset where this Class was labeled as True but likely should have been False (or vice versa).
            - *Issue Probability*: The  *Num Examples* column divided by the total number of examples in the dataset; i.e. the relative overall frequency of each type of label issue in your dataset.

        By default, the rows in this DataFrame are ordered by "Issue Probability" (descending).
    """

    num_examples = _get_num_examples_multilabel(labels=labels, confident_joint=confident_joint)
    summary_issue_counts = defaultdict(list)
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
        true_but_false_count = sum(np.logical_and(label == 1, binary_label_issues))
        false_but_true_count = sum(np.logical_and(label == 0, binary_label_issues))

        if class_names is not None:
            summary_issue_counts["Class Name"].append(class_names[class_num])
        summary_issue_counts["Class Index"].append(class_num)
        summary_issue_counts["In Given Label"].append(True)
        summary_issue_counts["In Suggested Label"].append(False)
        summary_issue_counts["Num Examples"].append(true_but_false_count)
        summary_issue_counts["Issue Probability"].append(true_but_false_count / num_examples)

        if class_names is not None:
            summary_issue_counts["Class Name"].append(class_names[class_num])
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
    Returns a DataFrame with three overall label quality scores per class for a multi-label dataset.

    These numbers summarize all examples annotated with the class (details listed below under the Returns parameter).
    By default, classes are ordered by "Label Quality Score", so the most problematic classes are reported first in the DataFrame.

    Score values are unnormalized and may be very small. What matters is their relative ranking across the classes.

    **Parameters**:

    For information about the arguments to this method, see the documentation of
    `~cleanlab.multilabel_classification.dataset.common_multilabel_issues`.

    Returns
    -------
    overall_label_quality : pd.DataFrame
        Pandas DataFrame with one row per class and columns: "Class Index", "Label Issues",
        "Inverse Label Issues", "Label Issues", "Inverse Label Noise", "Label Quality Score".
        Some entries are overall quality scores between 0 and 1, summarizing how good overall the labels
        appear to be for that class (lower values indicate more erroneous labels).
        Other entries are estimated counts of annotation errors related to this class.

        Here is what each column represents:
            - *Class Name*: The name of the class if class_names is provided.
            - *Class Index*: The index of the class in 0, 1, ..., K-1.
            - *Label Issues*: Estimated number of examples in the dataset that are labeled as belonging to class k but actually should not belong to this class.
            - *Inverse Label Issues*: Estimated number of examples in the dataset that should actually be labeled as class k but did not receive this label.
            - *Label Noise*: Estimated proportion of examples in the dataset that are labeled as class k but should not be. For each class k: this is computed by dividing the number of examples with "Label Issues" that were labeled as class k by the total number of examples labeled as class k.
            - *Inverse Label Noise*: Estimated proportion of examples in the dataset that should actually be labeled as class k but did not receive this label.
            - *Label Quality Score*: Estimated proportion of examples labeled as class k that have been labeled correctly, i.e. ``1 - label_noise``.

        By default, the DataFrame is ordered by "Label Quality Score" (in ascending order), so the classes with the most label issues appear first.
    """

    issues_df = common_multilabel_issues(
        labels=labels, pred_probs=pred_probs, class_names=class_names, confident_joint=joint
    )
    issues_dict = defaultdict(defaultdict)  # type: Dict[str, Any]
    num_examples = _get_num_examples_multilabel(labels=labels, confident_joint=confident_joint)
    return_columns = [
        "Class Name",
        "Class Index",
        "Label Issues",
        "Inverse Label Issues",
        "Label Noise",
        "Inverse Label Noise",
        "Label Quality Score",
    ]
    if class_names is None:
        return_columns = return_columns[1:]
    for class_num, row in issues_df.iterrows():
        if row["In Given Label"]:
            if class_names is not None:
                issues_dict[row["Class Index"]]["Class Name"] = row["Class Name"]
            issues_dict[row["Class Index"]]["Label Issues"] = int(
                row["Issue Probability"] * num_examples
            )
            issues_dict[row["Class Index"]]["Label Noise"] = row["Issue Probability"]
            issues_dict[row["Class Index"]]["Label Quality Score"] = (
                1 - issues_dict[row["Class Index"]]["Label Noise"]
            )
        else:
            if class_names is not None:
                issues_dict[row["Class Index"]]["Class Name"] = row["Class Name"]
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
    )[return_columns]


def _get_num_examples_multilabel(labels=None, confident_joint: Optional[np.ndarray] = None) -> int:
    """Helper method that finds the number of examples from the parameters or throws an error
    if neither parameter is provided.

    Parameters
    ----------
    For parameter info, see the docstring of `~cleanlab.multilabel_classification.dataset.common_multilabel_issues`.

    Returns
    -------
    num_examples : int
        The number of examples in the dataset.

    Raises
    ------
    ValueError
        If `labels` is None.
    """

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
    confident_joint=None,
) -> float:
    """Returns a single score between 0 and 1 measuring the overall quality of all labels in a multi-label classification dataset.
    Intuitively, the score is the average correctness of the given labels across all examples in the
    dataset. So a score of 1 suggests your data is perfectly labeled and a score of 0.5 suggests
    half of the examples in the dataset may be incorrectly labeled. Thus, a higher
    score implies a higher quality dataset.

    **Parameters**: For information about the arguments to this method, see the documentation of
    `~cleanlab.multilabel_classification.dataset.common_multilabel_issues`.

    Returns
    -------
    health_score : float
        A overall score between 0 and 1, where 1 implies all labels in the dataset are estimated to be correct.
        A score of 0.5 implies that half of the dataset's labels are estimated to have issues.
    """
    num_examples = _get_num_examples_multilabel(labels=labels)
    issues = find_label_issues(
        labels=labels, pred_probs=pred_probs, confident_joint=confident_joint
    )
    return 1.0 - sum(issues) / num_examples


def multilabel_health_summary(
    labels=None,
    pred_probs=None,
    *,
    class_names=None,
    num_examples=None,
    confident_joint=None,
    verbose=True,
) -> Dict:
    """Prints a health summary of your multi-label dataset.

    This summary includes useful statistics like:

    * The classes with the most and least label issues.
    * Overall label quality scores, summarizing how accurate the labels appear across the entire dataset.

    **Parameters**: For information about the arguments to this method, see the documentation of
    `~cleanlab.multilabel_classification.dataset.common_multilabel_issues`.

    Returns
    -------
    summary : dict
        A dictionary containing keys (see the corresponding functions' documentation to understand the values):
            - ``"overall_label_health_score"``, corresponding to output of `~cleanlab.multilabel_classification.dataset.overall_multilabel_health_score`
            - ``"classes_by_multilabel_quality"``, corresponding to output of `~cleanlab.multilabel_classification.dataset.rank_classes_by_multilabel_quality`
            - ``"common_multilabel_issues"``, corresponding to output of `~cleanlab.multilabel_classification.dataset.common_multilabel_issues`
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

# Copyright (C) 2017-2022  Cleanlab Inc.
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


"""Dataset Module
Supports dataset-level and class-level automated quality, including finding which classes should
be merged with other classes in your dataset, which classes should be removed,
and which classes tend to be annotated most/least correctly overall.
"""

import numpy as np
import pandas as pd
from cleanlab.count import estimate_joint


def rank_classes_by_label_quality(
    labels=None,
    pred_probs=None,
    *,
    class_names=None,
    num_examples=None,
    joint=None,
    confident_joint=None,
    multi_label=False,
):
    """Returns a Pandas DataFrame with all classes and three overall class label quality scores
     (details about each score are listed in the Returns parameter). By default, classes are ordered
     by "Overall Label Noise Score" descending.

    Score values are unnormalized and may tend to be very small. What matters is their relative
    ranking across the classes.

    Parameters
    ----------
    For parameter info, see the docstring of `dataset.find_overlapping_classes`

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with cols "Class Index", "Overall Label Issues", "Inverse Label Issues",
        "Overall Label Issues", "Inverse Label Noise Score", "Overall Label Quality Score" and a
        description of each below. Noise scores are between 0 and 1, where 0 implies no label issues
        in the class. The "Overall Label Quality Score" is also between 0 and 1 where 1 implies
        perfect quality.
        Columns:
        * "Class Index"
            The index of the class in 0, 1, ..., num_classes - 1.
        * Overall Label Issues
            count(given_label = k, true_label != k)
            Estimated number of label issues in the class (usually the most accurate method).
        * Inverse Label Issues
            count(given_label != k, true_label = k)
            Estimated number of times this class occurs in other classes in the dataset.
        * Overall Label Noise Score
            prob(true_label != k | given_label = k)
            Estimated proportion of label issues in the class.
        * Inverse Label Noise Score
            prob(given_label != k | true_label = k)
            Estimated proportion of times this class label occurs in other classes in the dataset.
        * Overall Label Quality Score
            p(given_label = k | true_label = k)
            How accurate the labels are in each class as a conditional prob p(given=k | true=k).
        By default, the DataFrame is ordered by "Overall Label Noise Score" descending.
    """

    if joint is None:
        joint = estimate_joint(
            labels=labels,
            pred_probs=pred_probs,
            confident_joint=confident_joint,
            multi_label=multi_label,
        )
    if num_examples is None:
        num_examples = _get_num_examples(labels=labels)
    given_label_noise = joint.sum(axis=1) - joint.diagonal()  # p(s=k) - p(s=k,y=k) = p(y!=k, s=k)
    true_label_noise = joint.sum(axis=0) - joint.diagonal()  # p(y=k) - p(s=k,y=k) = p(s!=k,y=k)
    given_conditional_noise = given_label_noise / joint.sum(axis=1)  # p(y!=k, s=k) / p(s=k)
    true_conditional_noise = true_label_noise / joint.sum(axis=0)  # p(s!=k, y=k) / p(y=k)
    df = pd.DataFrame(
        {
            "Class Index": np.arange(len(joint)),
            "Overall Label Issues": (given_label_noise * num_examples).round().astype(int),
            "Inverse Label Issues": (true_label_noise * num_examples).round().astype(int),
            "Overall Label Noise Score": given_conditional_noise,  # p(y!=k | s=k)
            "Inverse Label Noise Score": true_conditional_noise,  # p(s!=k | y=k)
            # Below could equivalently be computed as: joint.diagonal() / joint.sum(axis=1)
            "Overall Label Quality Score": 1 - given_conditional_noise,  # p(y=k | s=k)
        }
    )
    if class_names is not None:
        df.insert(loc=0, column="Class Name", value=class_names)
    return df.sort_values(by="Overall Label Noise Score", ascending=False)


def find_overlapping_classes(
    labels=None,
    pred_probs=None,
    *,
    asymmetric=False,
    class_names=None,
    num_examples=None,
    joint=None,
    confident_joint=None,
    multi_label=False,
):
    """Returns the classes that are often confused by machine learning model or data labelers.
    Consider merging the top pairs of classes returned by this method each into a single class.
    If the dataset is labeled by human annotators, consider clearly defining the
    difference between the classes prior to have annotators label the data.

    This method provides two scores in the pandas Data Frame that is returned:
    * "Num Overlapping Examples" - The number of examples where the two classes overlap
    * "Joint Probability" - "Num Overlapping Examples" / total number of examples in the dataset.
    interchangeable and returns a dataframe with the classe and the joint probability score
    between 0 and 1. Use this function to determine which classes in a dataset should be merged.

    This method works by providing any one (and only one) of the following inputs:
    1. labels and pred_probs, or
    2. joint, or
    3. confident_joint
    but only provide **one of the above input options**, do not provide a combination.

    The method uses the joint distribution of noisy and true labels to compute ontological issues
    via the approach published in (Northcutt, Jiang, Chuang (2021) JAIR).
    paper: https://arxiv.org/abs/1911.00068

    This method measures how often the annotators confuse two classes.
    This method differs from just using a similarity matrix or confusion matrix. Instead, it works
    even if the model that generated pred_probs in more confident in some classes than others
    and has heterogeneity in average confidence across classes.

    Parameters
    ----------
    labels : np.array (shape (num_examples, 1))
        A discrete vector of noisy labels, i.e. some labels may be erroneous.
        *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    pred_probs : np.array (shape (num_examples, num_classes))
        P(label=k|x) is a matrix with K model-predicted probabilities.
        Each row of this matrix corresponds to an example `x` and contains the model-predicted
        probabilities that `x` belongs to each possible class.
        The columns must be ordered such that these probabilities correspond to class 0,1,2,...
        `pred_probs` should have been computed using 3 (or higher) fold cross-validation.

    asymmetric : bool (default: True)
        If `asymmetric==True`, includes both pairs (class1, class2) and (class2, class1). Use this
        for finding "is a" relationships where for example "class1 is a class2".
        If `asymmetric==False`, the pair (class1, class2) will only be returned once and order is
        arbitrary (internally this is just summing score(class1, class2) + score(class2, class1)).

    class_names : iterable<str>
        A list or other iterable of the string class names. The list should be in the order that
        matches the class indices. So if class 0 is 'dog' and class 1 is 'cat', then
        `class_names = ['dog', 'cat'].

    num_examples : int (default: None)
        The number of examples in the datasets, i.e. len(labels). You only need to provide this if
        you use this function with the joint, e.g. `find_overlapping_classes(joint=joint)` otherwise
        this is automatically inferred via `sum(confident_joint)` or `len(labels)`.

    joint : np.array<float> (shape (num_classes, num_classes))
        Estimated joint distribution of the noisy labels and true labels. This takes the form of a
        2-D square matrix (num_classes, num_classes) with all entries summing to 1 (valid
        probability distribution). Each entry in the matrix captures the co-occurence joint
        probability of a true label and a noisy label, i.e. p(noisy_label=i, true_label=j).
        **Important**. If you input the joint, you must also input `num_examples`.

    confident_joint : np.array<int> (shape (num_examples, num_classes))
        A K,K integer matrix of count(label=k, true_label=k). Estimates a confident subset of
        the joint distribution of the noisy and true labels P_{labels,y}.
        Each entry in the matrix contains the number of examples confidently
        counted into every pair (label=j, true_label=k) classes.

    multi_label : bool
        If true, labels should be an iterable (e.g. list) of iterables, containing a
        list of labels for each example, instead of just a single label.
        The MAJOR DIFFERENCE in how this is calibrated versus single_label, is the total number of
        errors considered is based on the number of labels, not the number of examples. So, the
        calibrated confident_joint will sum to the number of total labels.
        The multi-label setting supports classification tasks where an example has 1 or more labels.
        Example of a multi-labeled `labels` input: [[0,1], [1], [0,2], [0,1,2], [0], [1], ...]

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with columns "Class Index A", "Class Index B",
        "Num Overlapping Examples", "Joint Probability" and a description of each below.
        * "Class Index A"
            The index of a class in 0, 1, ..., num_classes - 1.
        * "Class Index B"
            The index of a different class (from Class A) in 0, 1, ..., num_classes - 1.
        * Num Overlapping Examples
            Estimated number of labels overlapping between the two classes.
        * Joint Probability
            The "Num Overlapping Examples" divided by the number of examples in the dataset.
        By default, the DataFrame is ordered by "Joint Probability" descending.
    """

    def _2d_matrix_to_row_column_value_list(matrix):
        """Create a list<tuple> [(row_index, col_index, value)] representation of matrix.

        Parameters
        ----------
        matrix : np.array<float>
            Any valid np.array 2-d dimensional matrix.

        Returns
        -------
        list<tuple>
            A [(row_index, col_index, value)] representation of matrix.
        """

        return [(*i, v) for i, v in np.ndenumerate(matrix)]

    if joint is None:
        joint = estimate_joint(
            labels=labels,
            pred_probs=pred_probs,
            confident_joint=confident_joint,
            multi_label=multi_label,
        )
    if num_examples is None:
        num_examples = _get_num_examples(labels=labels)
    if asymmetric:
        rcv_list = _2d_matrix_to_row_column_value_list(joint)
        # Remove diagonal elements
        rcv_list = [tup for tup in rcv_list if tup[0] != tup[1]]
    else:  # symmetric
        # Sum the upper and lower triangles and remove the lower triangle and the diagonal
        sym_joint = np.triu(joint) + np.tril(joint).T
        rcv_list = _2d_matrix_to_row_column_value_list(sym_joint)
        # Provide values only in (the upper triangle) of the matrix.
        rcv_list = [tup for tup in rcv_list if tup[0] < tup[1]]
    df = pd.DataFrame(rcv_list, columns=["Class Index A", "Class Index B", "Joint Probability"])
    num_overlapping = (df["Joint Probability"] * num_examples).round().astype(int)
    df.insert(loc=2, column="Num Overlapping Examples", value=num_overlapping)
    if class_names is not None:
        df.insert(
            loc=0, column="Class Name A", value=df["Class Index A"].apply(lambda x: class_names[x])
        )
        df.insert(
            loc=1, column="Class Name B", value=df["Class Index B"].apply(lambda x: class_names[x])
        )
    return df.sort_values(by="Joint Probability", ascending=False)


def overall_label_health_score(
    labels=None,
    pred_probs=None,
    *,
    num_examples=None,
    joint=None,
    confident_joint=None,
    multi_label=False,
    verbose=True,
):
    """Returns a single score/metric between 0 and 1 for the quality of all labels in a dataset.
    Intuitively, the score is the average label consistency across all classes in the
    dataset. So a score of 1 suggests your data is perfectly labeled and a score of 0.5 suggests
    that, on average across all classes, about half of the label may have issues. Thus, a higher
    score implies higher quality labels, with 1 implying labels that have no issues.

    Parameters
    ----------
    For parameter info, see the docstring of `dataset.find_overlapping_classes`

    Returns
    -------
    health_score : float
        A score between 0 and 1 where 1 implies the dataset has all estimated perfect labels.
        A score of 0.5 implies that, on average, half of the dataset's label have estimated issues.
    """

    if joint is None:
        joint = estimate_joint(
            labels=labels,
            pred_probs=pred_probs,
            confident_joint=confident_joint,
            multi_label=multi_label,
        )
    if num_examples is None:
        num_examples = _get_num_examples(labels=labels)
    joint_trace = joint.trace()
    if verbose:
        num_issues = (num_examples * (1 - joint_trace)).round().astype(int)
        print(
            f" * Overall, about {1 - joint_trace:.0%} ({num_issues:,} of the {num_examples:,}) "
            f"labels in your dataset have issues.\n"
            f" ** The overall label health score for this dataset is: {joint_trace:.2f}."
        )
    return joint_trace


def health_summary(
    labels=None,
    pred_probs=None,
    *,
    asymmetric=False,
    class_names=None,
    num_examples=None,
    joint=None,
    confident_joint=None,
    multi_label=False,
):
    """Prints a healthy summary of your datasets including results for powerful statistics like:
    * the classes with the most and least label issues
    * classes that overlap and could potentially be merged
    * overall data label quality health score statistics for your dataset

    Parameters
    ----------
    For parameter info, see the docstring of `dataset.find_overlapping_classes`

    Returns
    -------
    health_score : float
        A score between 0 and 1 where 1 implies the dataset has all estimated perfect labels.
        A score of 0.5 implies that, on average, half of the dataset's label have estimated issues.
    """

    if joint is None:
        joint = estimate_joint(
            labels=labels,
            pred_probs=pred_probs,
            confident_joint=confident_joint,
            multi_label=multi_label,
        )
    if num_examples is None:
        num_examples = _get_num_examples(labels=labels)

    print(
        "*" * 56
        + "\n"
        + "Generating a Cleanlab Dataset Health Summary\n"
        + f" for your dataset with {num_examples:,} examples and {len(joint):,} classes.\n"
        + "Note, Cleanlab is not a medical doctor... yet.\n"
        + "*" * 56
        + "\n",
    )

    df_class_label_quality = rank_classes_by_label_quality(
        labels=labels,
        pred_probs=pred_probs,
        class_names=class_names,
        num_examples=num_examples,
        joint=joint,
        confident_joint=confident_joint,
        multi_label=multi_label,
    )
    print("Overall Class Quality and Noise across your dataset (below)")
    print("-" * 60, "\n", flush=True)
    print(df_class_label_quality)

    df_class_label_quality = find_overlapping_classes(
        labels=labels,
        pred_probs=pred_probs,
        asymmetric=asymmetric,
        class_names=class_names,
        num_examples=num_examples,
        joint=joint,
        confident_joint=confident_joint,
        multi_label=multi_label,
    )
    print(
        "\nClass Overlap. In some cases, you may want to merge classes in the top rows (below)"
        + "\n"
        + "-" * 83
        + "\n",
        flush=True,
    )
    print(df_class_label_quality)
    print()
    health_score = overall_label_health_score(labels=labels, pred_probs=pred_probs, verbose=True)
    print("\nGenerated with <3 from Cleanlab.")
    return health_score


def _get_num_examples(labels=None):
    """Helper method that finds the number of examples from the parameters or throws an error
    if neither parameter is provided.

    Parameters
    ----------
    For parameter info, see the docstring of `dataset.find_overlapping_classes`

    Returns
    -------
    num_examples : int
        The number of examples in the dataset.

    Raises
    ------
    ValueError
        If `labels` is None."""

    if labels is not None:
        num_examples = len(labels)
    else:
        raise ValueError(
            "Error: num_examples is None. You must provide a value for num_examples "
            "when calling this method using the joint as an input parameter."
        )
    return num_examples

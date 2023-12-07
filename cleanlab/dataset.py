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
Provides dataset-level and class-level overviews of issues in your classification dataset.
If your task allows you to modify the classes in your dataset, this module can help you determine
which classes to remove (see `~cleanlab.dataset.rank_classes_by_label_quality`)
and which classes to merge (see `~cleanlab.dataset.find_overlapping_classes`).
"""

from typing import Optional, cast
import numpy as np
import pandas as pd
from cleanlab.count import estimate_joint, num_label_issues


def rank_classes_by_label_quality(
    labels=None,
    pred_probs=None,
    *,
    class_names=None,
    num_examples=None,
    joint=None,
    confident_joint=None,
    multi_label=False,
) -> pd.DataFrame:
    """
    Returns a Pandas DataFrame with all classes and three overall class label quality scores
    (details about each score are listed in the Returns parameter). By default, classes are ordered
    by "Label Quality Score", ascending, so the most problematic classes are reported first.

    Score values are unnormalized and may tend to be very small. What matters is their relative
    ranking across the classes.

    This method works by providing any one (and only one) of the following inputs:

    1. ``labels`` and ``pred_probs``, or
    2. ``joint`` and ``num_examples``, or
    3. ``confident_joint``

    Only provide **exactly one of the above input options**, do not provide a combination.

    Examples
    --------
    >>> from cleanlab.dataset import rank_classes_by_label_quality
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import cross_val_predict
    >>> data, labels = get_data_labels_from_dataset()
    >>> yourFavoriteModel = LogisticRegression()
    >>> pred_probs = cross_val_predict(yourFavoriteModel, data, labels, cv=3, method="predict_proba")
    >>> df = rank_classes_by_label_quality(labels=labels, pred_probs=pred_probs)

    **Parameters**: For parameter info, see the docstring of `~cleanlab.dataset.find_overlapping_classes`.

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
        * *Label Issues*: ``count(given_label = k, true_label != k)``, estimated number of examples in the dataset that are labeled as class k but should have a different label.
        * *Inverse Label Issues*: ``count(given_label != k, true_label = k)``, estimated number of examples in the dataset that should actually be labeled as class k but have been given another label.
        * *Label Noise*: ``prob(true_label != k | given_label = k)``, estimated proportion of examples in the dataset that are labeled as class k but should have a different label. For each class k: this is computed by dividing the number of examples with "Label Issues" that were labeled as class k by the total number of examples labeled as class k.
        * *Inverse Label Noise*: ``prob(given_label != k | true_label = k)``, estimated proportion of examples in the dataset that should actually be labeled as class k but have been given another label.
        * *Label Quality Score*: ``p(true_label = k | given_label = k)``. This is the proportion of examples with given label k that have been labeled correctly, i.e. ``1 - label_noise``.

        By default, the DataFrame is ordered by "Label Quality Score", ascending.
    """
    if multi_label:
        raise ValueError(
            "For multilabel data, please instead call:  multilabel_classification.dataset.overall_multilabel_health_score()"
        )

    if joint is None:
        joint = estimate_joint(
            labels=labels,
            pred_probs=pred_probs,
            confident_joint=confident_joint,
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
            "Label Issues": (given_label_noise * num_examples).round().astype(int),
            "Inverse Label Issues": (true_label_noise * num_examples).round().astype(int),
            "Label Noise": given_conditional_noise,  # p(y!=k | s=k)
            "Inverse Label Noise": true_conditional_noise,  # p(s!=k | y=k)
            # Below could equivalently be computed as: joint.diagonal() / joint.sum(axis=1)
            "Label Quality Score": 1 - given_conditional_noise,  # p(y=k | s=k)
        }
    )
    if class_names is not None:
        df.insert(loc=0, column="Class Name", value=class_names)
    return df.sort_values(by="Label Quality Score", ascending=True).reset_index(drop=True)


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
) -> pd.DataFrame:
    """Returns the pairs of classes that are often mislabeled as one another.
    Consider merging the top pairs of classes returned by this method each into a single class.
    If the dataset is labeled by human annotators, consider clearly defining the
    difference between the classes prior to having annotators label the data.

    This method provides two scores in the Pandas DataFrame that is returned:

    * **Num Overlapping Examples**: The number of examples where the two classes overlap
    * **Joint Probability**: `(num overlapping examples / total number of examples in the dataset`).

    This method works by providing any one (and only one) of the following inputs:

    1. ``labels`` and ``pred_probs``, or
    2. ``joint`` and ``num_examples``, or
    3. ``confident_joint``

    Only provide **exactly one of the above input options**, do not provide a combination.

    This method uses the joint distribution of noisy and true labels to compute ontological
    issues via the approach published in `Northcutt et al.,
    2021 <https://jair.org/index.php/jair/article/view/12125>`_.

    Examples
    --------
    >>> from cleanlab.dataset import find_overlapping_classes
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import cross_val_predict
    >>> data, labels = get_data_labels_from_dataset()
    >>> yourFavoriteModel = LogisticRegression()
    >>> pred_probs = cross_val_predict(yourFavoriteModel, data, labels, cv=3, method="predict_proba")
    >>> df = find_overlapping_classes(labels=labels, pred_probs=pred_probs)

    Note
    ----
    The joint distribution of noisy and true labels is asymmetric, and therefore the joint
    probability ``p(given="vehicle", true="truck") != p(true="truck", given="vehicle")``.
    This is intuitive. Images of trucks (true label) are much more likely to be labeled as a car
    (given label) than images of cars (true label) being frequently mislabeled as truck (given
    label). cleanlab takes these differences into account for you automatically via the joint
    distribution. If you do not want this behavior, simply set ``asymmetric=False``.

    This method estimates how often the annotators confuse two classes.
    This differs from just using a similarity matrix or confusion matrix,
    as these summarize characteristics of the predictive model rather than the data labelers (i.e. annotators).
    Instead, this method works even if the model that generated `pred_probs` tends to be more confident in some classes than others.

    Parameters
    ----------
    labels : np.ndarray or list, optional
      An array_like (of length N) of noisy labels for the classification dataset, i.e. some labels may be erroneous.
      Elements must be integers in the set 0, 1, ..., K-1, where K is the number of classes.
      All the classes (0, 1, ..., and K-1) should be present in ``labels``, such that
      ``len(set(labels)) == pred_probs.shape[1]`` for standard multi-class classification with single-labeled data (e.g. ``labels =  [1,0,2,1,1,0...]``).
      For multi-label classification where each example can belong to multiple classes (e.g. ``labels = [[1,2],[1],[0],[],...]``),
      your labels should instead satisfy: ``len(set(k for l in labels for k in l)) == pred_probs.shape[1])``.

    pred_probs : np.ndarray, optional
      An array of shape ``(N, K)`` of model-predicted probabilities,
      ``P(label=k|x)``. Each row of this matrix corresponds
      to an example `x` and contains the model-predicted probabilities that
      `x` belongs to each possible class, for each of the K classes. The
      columns must be ordered such that these probabilities correspond to
      class 0, 1, ..., K-1. `pred_probs` should have been computed using 3 (or
      higher) fold cross-validation.

    asymmetric : bool, optional
      If ``asymmetric=True``, returns separate estimates for both pairs (class1, class2) and (class2, class1). Use this
      for finding "is a" relationships where for example "class1 is a class2".
      In this case, num overlapping examples counts the number of examples that have been labeled as class1 which should actually have been labeled as class2.
      If ``asymmetric=False``, the pair (class1, class2) will only be returned once with an arbitrary order.
      In this case, their estimated score is the sum: ``score(class1, class2) + score(class2, class1))``.

    class_names : Iterable[str]
        A list or other iterable of the string class names. The list should be in the order that
        matches the class indices. So if class 0 is 'dog' and class 1 is 'cat', then
        ``class_names = ['dog', 'cat']``.

    num_examples : int or None, optional
        The number of examples in the dataset, i.e. ``len(labels)``. You only need to provide this if
        you use this function with the joint, e.g. ``find_overlapping_classes(joint=joint)``, otherwise
        this is automatically computed via ``sum(confident_joint)`` or ``len(labels)``.

    joint : np.ndarray, optional
        An array of shape ``(K, K)``, where K is the number of classes,
        representing the estimated joint distribution of the noisy labels and
        true labels. The sum of all entries in this matrix must be 1 (valid
        probability distribution). Each entry in the matrix captures the co-occurence joint
        probability of a true label and a noisy label, i.e. ``p(noisy_label=i, true_label=j)``.
        **Important**. If you input the joint, you must also input `num_examples`.

    confident_joint : np.ndarray, optional
      An array of shape ``(K, K)`` representing the confident joint, the matrix used for identifying label issues, which
      estimates a confident subset of the joint distribution of the noisy and true labels, ``P_{noisy label, true label}``.
      Entry ``(j, k)`` in the matrix is the number of examples confidently counted into the pair of ``(noisy label=j, true label=k)`` classes.
      The `confident_joint` can be computed using :py:func:`count.compute_confident_joint <cleanlab.count.compute_confident_joint>`.
      If not provided, it is computed from the given (noisy) `labels` and `pred_probs`.

    Returns
    -------
    overlapping_classes : pd.DataFrame
        Pandas DataFrame with columns "Class Index A", "Class Index B",
        "Num Overlapping Examples", "Joint Probability" and a description of each below.
        Each row corresponds to a pair of classes.

        * *Class Index A*: the index of a class in 0, 1, ..., K-1.
        * *Class Index B*: the index of a different class (from Class A) in 0, 1, ..., K-1.
        * *Num Overlapping Examples*: estimated number of labels overlapping between the two classes.
        * *Joint Probability*: the *Num Overlapping Examples* divided by the number of examples in the dataset.

        By default, the DataFrame is ordered by "Joint Probability" descending.
    """

    def _2d_matrix_to_row_column_value_list(matrix):
        """Create a list<tuple> [(row_index, col_index, value)] representation of matrix.

        Parameters
        ----------
        matrix : np.ndarray<float>
            Any valid np.ndarray 2-d dimensional matrix.

        Returns
        -------
        list<tuple>
            A [(row_index, col_index, value)] representation of matrix.
        """

        return [(*i, v) for i, v in np.ndenumerate(matrix)]

    if multi_label:
        raise ValueError(
            "For multilabel data, please instead call: multilabel_classification.dataset.common_multilabel_issues()"
        )

    if joint is None:
        joint = estimate_joint(
            labels=labels,
            pred_probs=pred_probs,
            confident_joint=confident_joint,
        )
    if num_examples is None:
        num_examples = _get_num_examples(labels=labels, confident_joint=confident_joint)
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
    return df.sort_values(by="Joint Probability", ascending=False).reset_index(drop=True)


def overall_label_health_score(
    labels=None,
    pred_probs=None,
    *,
    num_examples=None,
    confident_joint=None,
    joint=None,
    multi_label=False,
    verbose=True,
) -> float:
    """Returns a single score between 0 and 1 measuring the overall quality of all labels in a dataset.
    Intuitively, the score is the average correctness of the given labels across all examples in the
    dataset. So a score of 1 suggests your data is perfectly labeled and a score of 0.5 suggests
    half of the examples in the dataset may be incorrectly labeled. Thus, a higher
    score implies a higher quality dataset.

    This method works by providing any one (and only one) of the following inputs:

    1. ``labels`` and ``pred_probs``, or
    2. ``joint`` and ``num_examples``, or
    3. ``confident_joint``

    Only provide **exactly one of the above input options**, do not provide a combination.

    Examples
    --------
    >>> from cleanlab.dataset import overall_label_health_score
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import cross_val_predict
    >>> data, labels = get_data_labels_from_dataset()
    >>> yourFavoriteModel = LogisticRegression()
    >>> pred_probs = cross_val_predict(yourFavoriteModel, data, labels, cv=3, method="predict_proba")
    >>> score = overall_label_health_score(labels=labels, pred_probs=pred_probs)  # doctest: +SKIP

    **Parameters**: For parameter info, see the docstring of `~cleanlab.dataset.find_overlapping_classes`.


    Returns
    -------
    health_score : float
        A score between 0 and 1, where 1 implies all labels in the dataset are estimated to be correct.
        A score of 0.5 implies that half of the dataset's labels are estimated to have issues.
    """
    if multi_label:
        raise ValueError(
            "For multilabel data, please instead call: multilabel_classification.dataset.overall_multilabel_health_score()"
        )
    if num_examples is None:
        num_examples = _get_num_examples(labels=labels, confident_joint=confident_joint)

    if pred_probs is None or labels is None:
        if joint is None:
            joint = estimate_joint(
                labels=labels,
                pred_probs=pred_probs,
                confident_joint=confident_joint,
            )
        joint_trace = joint.trace()
        num_issues = (num_examples * (1 - joint_trace)).round().astype(int)
        health_score = joint_trace
    else:
        num_issues = num_label_issues(
            labels=labels, pred_probs=pred_probs, confident_joint=confident_joint
        )
        health_score = 1 - num_issues / num_examples

    if verbose:
        print(
            f" * Overall, about {(1 - health_score):.0%} ({num_issues:,} of the {num_examples:,}) "
            f"labels in your dataset have potential issues.\n"
            f" ** The overall label health score for this dataset is: {health_score:.2f}."
        )
    return health_score


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
    verbose=True,
) -> dict:
    """Prints a health summary of your dataset.

    This summary includes useful statistics like:

    * The classes with the most and least label issues.
    * Classes that overlap and could potentially be merged.
    * Overall label quality scores, summarizing how accurate the labels appear overall.

    This method works by providing any one (and only one) of the following inputs:

    1. ``labels`` and ``pred_probs``, or
    2. ``joint`` and ``num_examples``, or
    3. ``confident_joint``

    Only provide **exactly one of the above input options**, do not provide a combination.

    Examples
    --------
    >>> from cleanlab.dataset import health_summary
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import cross_val_predict
    >>> data, labels = get_data_labels_from_dataset()
    >>> yourFavoriteModel = LogisticRegression()
    >>> pred_probs = cross_val_predict(yourFavoriteModel, data, labels, cv=3, method="predict_proba")
    >>> summary = health_summary(labels=labels, pred_probs=pred_probs)  # doctest: +SKIP

    **Parameters**: For parameter info, see the docstring of `~cleanlab.dataset.find_overlapping_classes`.

    Returns
    -------
    summary : dict
        A dictionary containing keys (see the corresponding functions' documentation to understand the values):

        - ``"overall_label_health_score"``, corresponding to `~cleanlab.dataset.overall_label_health_score`
        - ``"joint"``, corresponding to :py:func:`count.estimate_joint <cleanlab.count.estimate_joint>`
        - ``"classes_by_label_quality"``, corresponding to `~cleanlab.dataset.rank_classes_by_label_quality`
        - ``"overlapping_classes"``, corresponding to `~cleanlab.dataset.find_overlapping_classes`
    """
    from cleanlab.internal.util import smart_display_dataframe

    if multi_label:
        raise ValueError(
            "For multilabel data, please call multilabel_classification.dataset.health_summary"
        )
    if joint is None:
        joint = estimate_joint(
            labels=labels,
            pred_probs=pred_probs,
            confident_joint=confident_joint,
        )
    if num_examples is None:
        num_examples = _get_num_examples(labels=labels)

    if verbose:
        longest_line = (
            f"|   for your dataset with {num_examples:,} examples "
            f"and {len(joint):,} classes.  |\n"
        )
        print(
            "-" * (len(longest_line) - 1)
            + "\n"
            + f"|  Generating a Cleanlab Dataset Health Summary{' ' * (len(longest_line) - 49)}|\n"
            + longest_line
            + f"|  Note, Cleanlab is not a medical doctor... yet.{' ' * (len(longest_line) - 51)}|\n"
            + "-" * (len(longest_line) - 1)
            + "\n",
        )

    df_class_label_quality = rank_classes_by_label_quality(
        labels=labels,
        pred_probs=pred_probs,
        class_names=class_names,
        num_examples=num_examples,
        joint=joint,
        confident_joint=confident_joint,
    )
    if verbose:
        print("Overall Class Quality and Noise across your dataset (below)")
        print("-" * 60, "\n", flush=True)
        smart_display_dataframe(df_class_label_quality)

    df_overlapping_classes = find_overlapping_classes(
        labels=labels,
        pred_probs=pred_probs,
        asymmetric=asymmetric,
        class_names=class_names,
        num_examples=num_examples,
        joint=joint,
        confident_joint=confident_joint,
    )
    if verbose:
        print(
            "\nClass Overlap. In some cases, you may want to merge classes in the top rows (below)"
            + "\n"
            + "-" * 83
            + "\n",
            flush=True,
        )
        smart_display_dataframe(df_overlapping_classes)
        print()

    health_score = overall_label_health_score(
        labels=labels,
        pred_probs=pred_probs,
        num_examples=num_examples,
        confident_joint=confident_joint,
        verbose=verbose,
    )
    if verbose:
        print("\nGenerated with <3 from Cleanlab.\n")
    return {
        "overall_label_health_score": health_score,
        "joint": joint,
        "classes_by_label_quality": df_class_label_quality,
        "overlapping_classes": df_overlapping_classes,
    }


def _get_num_examples(labels=None, confident_joint: Optional[np.ndarray] = None) -> int:
    """Helper method that finds the number of examples from the parameters or throws an error
    if neither parameter is provided.

    **Parameters:** For information about the arguments to this method, see the documentation of `dataset.find_overlapping_classes`

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
    num_examples = len(labels) if labels is not None else cast(int, np.sum(_confident_joint))
    return num_examples

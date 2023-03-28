import pandas as pd
import numpy as np
from cleanlab.count import estimate_joint
from cleanlab.filter import _find_multilabel_issues_per_class
from cleanlab.internal.multilabel_utils import get_onehot_num_classes
from collections import defaultdict

def common_multilabel_issues(
    labels=list,
    pred_probs=None,
    *,
    class_names=None,
    num_examples=None,
    joint=None,
    confident_joint=None,
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

    multi_label : bool, optional
      If ``True``, labels should be an iterable (e.g. list) of iterables, containing a
      list of labels for each example, instead of just a single label.
      The multi-label setting supports classification tasks where an example has 1 or more labels.
      Example of a multi-labeled `labels` input: ``[[0,1], [1], [0,2], [0,1,2], [0], [1], ...]``.

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
    y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
    if class_names is None:
        class_names = list(range(len(num_classes)))
    label_issues_list, labels_list, pred_probs_list = _find_multilabel_issues_per_class(
        labels,
        pred_probs,
        return_indices_ranked_by='self_confidence'
    )
    dcnt = defaultdict(defaultdict)
    for class_num, (label, issues_for_class) in enumerate(zip(y_one.T, label_issues_list)):
        binary_label_issues = np.zeros(len(label)).astype(bool)
        binary_label_issues[issues_for_class] = True
        dcnt[class_names[class_num]]['TruebutFalse'] = sum(np.logical_and(label == 1, binary_label_issues))
        dcnt[class_names[class_num]]['FalsebutTrue'] = sum(np.logical_and(label == 0, binary_label_issues))

    dct2 = defaultdict(list)
    for i in dcnt:
        for j in dcnt[i]:
            dct2['class_name'].append(i)
            if j == 'TruebutFalse':
                dct2['In_Given_Label'].append(True)
                dct2['In_Suggested_Label'].append("False")
            else:
                dct2['In_Given_Label'].append(False)
                dct2['In_Suggested_Label'].append("True")
            dct2['num_examples'].append(dcnt[i][j])
            dct2['Issue_probability'].append((dcnt[i][j]) / len(y_one))

    return pd.DataFrame.from_dict(dct2).set_index("class_name")

import numpy as np
from typing import List, Union, Tuple
from cleanlab.rank import get_label_quality_scores
from cleanlab.dataset import overall_label_health_score, _get_worst_class
import pandas as pd

# TODO: unsure where to place this function? should it go to cleanlab.rank or stay here
def _get_label_quality_scores_with_NA(
    labels: pd.Series,
    pred_probs: np.array,
    kwargs: dict = {},
) -> pd.Series:
    label_quality_scores_subset = get_label_quality_scores(
        labels=labels[pd.notna(labels)],
        pred_probs=pred_probs[pd.notna(labels)],
        **kwargs,
    )
    label_quality_scores = pd.Series(np.nan, index=np.arange(len(labels)))
    label_quality_scores[pd.notna(labels)] = label_quality_scores_subset
    return label_quality_scores


def get_consensus_labels(
    labels_multiannotator: pd.DataFrame,
    pred_probs: np.array,
    method: str,
) -> np.array:
    """Returns consensus labels aggregated from multiple annotators, determined by method specified.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape (N, M),
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_scores_multiannotator <cleanlab.multiannotator.get_label_quality_scores_multiannotator>`.

    pred_probs : np.array
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores_multiannotator <cleanlab.multiannotator.get_label_quality_scores_multiannotator>`.

    method : str (Options: ["majority"])
        The method used to aggregate labels from multiple annotators into a single consensus label. Default is by using the majority vote.
        Options include:
        - ``"majority"``: means consensus labels are reached via a simple majority vote among annotators, with ties broken via ``pred_probs``.

    Returns
    -------
    consensus_labels: np.array
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    """

    valid_methods = ["majority"]
    consensus_labels = np.empty(len(labels_multiannotator), dtype="int32")

    if method == "majority":
        mode_labels_multiannotator = labels_multiannotator.mode(axis=1)
        for i in range(len(mode_labels_multiannotator)):
            consensus_labels[i] = mode_labels_multiannotator.iloc[i][
                pred_probs[
                    i, mode_labels_multiannotator.iloc[i].dropna().astype(int).to_numpy()
                ].argmax()
            ]
    else:
        raise ValueError(
            f"""
            {method} is not a valid consensus method!
            Please choose a valid consensus_method: {valid_methods}
            """
        )

    return consensus_labels


def get_annotator_agreement(
    labels_multiannotator: pd.DataFrame,
    consensus_labels: np.array,
) -> np.array:
    """Returns the fractions of annotators that agree with the consensus labels. Note that the
    fraction for each example only considers the annotators that labeled that particular example.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape (N, M),
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_scores_multiannotator <cleanlab.multiannotator.get_label_quality_scores_multiannotator>`.

    consensus_labels : np.array
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.

    Returns
    -------
    annotator_agreement : np.array
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.
    """
    return (
        labels_multiannotator.assign(consensus_label=consensus_labels)
        .apply(
            lambda s: np.mean(s.drop("consensus_label") == s["consensus_label"]),
            axis=1,
        )
        .to_numpy()
    )


def get_quality_of_consensus(
    consensus_labels: np.array,
    pred_probs: np.array,
    num_annotations: np.array,
    annotator_agreement: np.array,
    kwargs: dict = {},
) -> np.array:
    """Return scores representing quality of the consensus label for each example,
    calculated using weighted product of `label_quality_score` of `consensus_labels` and `annotator_agreement`

    Parameters
    ----------
    consensus_labels : np.array
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.

    pred_probs : np.array
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores_multiannotator <cleanlab.multiannotator.get_label_quality_scores_multiannotator>`.

    num_annotations : np.array
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.

    annotator_agreement : np.array
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.

    Returns
    -------
    quality_of_consensus : np.array
        An array of shape ``(N,)`` with the quality score of the consensus.
        TODO: better explanation of this
    """
    lqs_consensus_labels = get_label_quality_scores(consensus_labels, pred_probs, **kwargs)
    frac = pred_probs[range(len(consensus_labels)), (consensus_labels)] / num_annotations
    quality_of_consensus = frac * lqs_consensus_labels + (1 - frac) * annotator_agreement
    return quality_of_consensus


def compute_consensus_stats(
    labels_multiannotator: pd.DataFrame,
    pred_probs: np.array,
    num_annotations: np.array,
    consensus_method: str,
    kwargs: dict = {},
) -> tuple:
    """Returns a tuple containing the consensus labels, annotator agreement scores, and quality of consensus

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape (N, M),
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_scores_multiannotator <cleanlab.multiannotator.get_label_quality_scores_multiannotator>`.

    pred_probs : np.array
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores_multiannotator <cleanlab.multiannotator.get_label_quality_scores_multiannotator>`.

    num_annotations : np.array
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.

    method : str (Options: ["majority"])
        The method used to aggregate labels from multiple annotators into a single consensus label. Default is by using the majority vote.
        For valid consensus methods, view :py:func:`get_consensus_labels <cleanlab.multiannotator.get_consensus_labels>`

    kwargs : dict
        Keyword arguments to pass into ``get_label_quality_scores()``.

    Returns
    ------
    tuple
        A tuple of (consensus_labels, annotator_agreement, quality_of_consensus).
    """
    # Compute consensus label using method specified
    consensus_labels = get_consensus_labels(
        labels_multiannotator=labels_multiannotator,
        pred_probs=pred_probs,
        method=consensus_method,
    )

    # Compute the fraction of annotator agreeing with the consensus labels
    annotator_agreement = get_annotator_agreement(
        labels_multiannotator=labels_multiannotator,
        consensus_labels=consensus_labels,
    )

    # Compute the label quality scores of the consensus labels
    quality_of_consensus = get_quality_of_consensus(
        consensus_labels=consensus_labels,
        pred_probs=pred_probs,
        num_annotations=num_annotations,
        annotator_agreement=annotator_agreement,
        **kwargs,
    )

    return (consensus_labels, annotator_agreement, quality_of_consensus)


def compute_multiannotator_stats(
    labels_multiannotator: pd.DataFrame,
    pred_probs: np.array,
    # label_quality_scores_multiannotator=None, # is this still needed? it is never called in the functioned
) -> pd.DataFrame:
    """Returns overall statistics about each annotator.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape (N, M),
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_scores_multiannotator <cleanlab.multiannotator.get_label_quality_scores_multiannotator>`.

    pred_probs : np.array
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores_multiannotator <cleanlab.multiannotator.get_label_quality_scores_multiannotator>`.

    Returns
    -------
    annotator_stats : pd.DataFrame
        pandas DataFrame in which each row corresponds to one annotator (the row IDs correspond to annotator IDs), with columns:
        - ``overall_quality``: overall quality of a given annotator's labels
        - ``num_labeled``: number of examples annotated by a given annotator
        - ``worst_class``: the class that is most frequently mislabeled by a given annotator
    """

    # Compute overall quality of each annotator's labels
    overall_label_health_score_df = labels_multiannotator.apply(
        lambda s: overall_label_health_score(
            s[pd.notna(s)].astype("int32"), pred_probs[pd.notna(s)], verbose=False
        )
    )

    # Compute the number of labels labeled/annotated by each annotator
    num_labeled = labels_multiannotator.count()

    # Find the worst labeled class for each annotator
    worst_class = labels_multiannotator.apply(
        lambda s: _get_worst_class(s[pd.notna(s)].astype("int32"), pred_probs[pd.notna(s)])
    )

    # Create multi-annotator stats DataFrame from its columns
    return pd.DataFrame(
        {
            "overall_quality": overall_label_health_score_df,
            "num_labeled": num_labeled,
            "worst_class": worst_class,
        }
    )


def get_label_quality_scores_multiannotator(
    labels_multiannotator: pd.DataFrame,
    pred_probs: np.array,
    *,
    consensus_method: Union[str, List[str]] = "majority",
    return_annotator_stats: bool = False,
    verbose: bool = True,
    kwargs: dict = {},
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Returns label quality scores for each example for each annotator.
    This function is for multiclass classification datasets where examples have been labeled by
    multiple annotators (not necessarily the same number of annotators per example).
    It computes quality scores for each annotator's labels and other useful values like how
    confident we are that the current consensus label is actually correct.
    The score is between 0 and 1; lower scores
    indicate labels less likely to be correct. For example:
    1 - clean label (the given label is likely correct).
    0 - dirty label (the given label is unlikely correct).

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape (N, M),
        where N is the number of examples and M is the number of annotators.
        labels_multiannotator[n][m] = given label for n-th example by m-th annotator.
        For a dataset with K classes, the given labels must be an integer in 0, 1, ..., K-1 or
        np.nan if this annotator did not label a particular example. Column names should correspond to the annotators' ID.

    pred_probs : np.array
        An array of shape ``(N, K)`` of model-predicted probabilities,
        ``P(label=k|x)``. Each row of this matrix corresponds
        to an example `x` and contains the model-predicted probabilities that
        `x` belongs to each possible class, for each of the K classes. The
        columns must be ordered such that these probabilities correspond to
        class 0, 1, ..., K-1.
        **Caution**: `pred_probs` from your model must be out-of-sample!
        You should never provide predictions on the same examples used to train the model,
        as these will be overfit and unsuitable for finding label-errors.
        To obtain out-of-sample predicted probabilities for every datapoint in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`.
        Alternatively it is ok if your model was trained on a separate dataset and you are only evaluating
        data that was previously held-out.

    consensus_method : str or List[str]
        For each example, which method should be used to aggregate labels from multiple annotators into a single consensus label.
        Options include:
        - ``"majority"``: means consensus labels are reached via a simple majority vote among annotators, with ties broken via ``pred_probs``.
        A List may be passed if you want to consider multiple methods for producing consensus labels.
        If a List is passed, then the 0th element of List is the method used to produce columns "consensus_label", "quality_of_consensus", "annotator_agreement" in the returned DataFrame.
        The 1st, 2nd, 3rd, etc. elements of this List are output as extra columns in the returned ``pandas DataFrame`` with names formatted as:
        consensus_label_SUFFIX, quality_of_consensus_SUFFIX
        where SUFFIX = the str element of this list, which must correspond to a valid method for computing consensus labels.

    quality_method : str or List[str]
        TODO (not added to arg list yet - potential addition based on use case)

    return_annotator_stats : bool = False
        TODO

    verbose : bool = True
        If ``verbose`` is set to ``True``, the full ``annotator_stats`` DataFrame is printed out during the execution of this function.

    kwargs : dict
        Keyword arguments to pass into ``get_label_quality_scores()``.

    Returns
    -------
    label_quality_score_multiannotator : pandas.DataFrame
        pandas DataFrame in which each row corresponds to one example, with columns:
        - ``quality_of_annotator_1``, ``quality_of_annotator_2``, ..., ``quality_of_annotator_M``: the label quality score for the labels provided by annotator M (is ``NaN`` for examples which this annotator did not label).
        - ``consensus_labels``: the single label that is best for each example (you can control how it is derived from all annotators' labels via the argument: ``consensus_method``)
        - ``num_annotations``: the number of annotators that have labeled each example.
        - ``annotator_agreement``: the fraction of annotators that agree with the consensus label (only consider the annotators that labeled that particular example).
        - ``quality_of_consensus``: label quality score for consensus label, calculated using weighted product of `label_quality_score` of `consensus_labels` and `annotator_agreement`
        Here annotator_1, annotator_2, ..., annotator_M suffixes may be replaced column names in ``labels_multiannotator`` DataFrame used to ID the annotators.

    annotator_stats : pandas.DataFrame, optional (returned if `return_annotator_stats=True`)
        Returns overall statistics about each annotator.
        For details, see the documentation of :py:func:`compute_multiannotator_stats<cleanlab.multiannotator.compute_multiannotator_stats>`
    """

    # Raise error if labels_multiannotator has np.NaN rows
    if labels_multiannotator.isna().all(axis=1).any():
        raise ValueError(
            f"""
                labels_multiannotator cannot have rows with all np.NaN.
            """
        )

    # Raise error if labels_multiannotator has <= 1 column
    if len(labels_multiannotator.columns) <= 1:
        raise ValueError(
            """
                labels_multiannotator must have more than one column.
                If there is only one annotator, use cleanlab.rank.get_label_quality_scores instead
            """
        )

    # Raise error if labels_multiannotator only has 1 set of labels

    # Raise warning if no examples with 2 or more annotators agree

    # Compute the label quality scores for each annotators' labels
    label_quality_scores_multiannotator = labels_multiannotator.apply(
        _get_label_quality_scores_with_NA, args=[pred_probs], **kwargs
    )
    label_quality_scores_multiannotator = label_quality_scores_multiannotator.add_prefix(
        "quality_annotator_"
    )

    # Count number of non-NaN values for each example
    num_annotations = labels_multiannotator.count(axis=1).to_numpy()
    label_quality_scores_multiannotator["num_annotations"] = num_annotations

    # Compute consesnsus labels, annotator agreement and quality of consensus
    consensus_labels, annotator_agreement, quality_of_consensus = compute_consensus_stats(
        labels_multiannotator=labels_multiannotator,
        pred_probs=pred_probs,
        num_annotations=num_annotations,
        consensus_method=consensus_method[0]
        if isinstance(consensus_method, list)
        else consensus_method,
        **kwargs,
    )

    (
        label_quality_scores_multiannotator["annotator_agreement"],
        label_quality_scores_multiannotator["consensus_label"],
        label_quality_scores_multiannotator["quality_of_consensus"],
    ) = (
        annotator_agreement,
        consensus_labels,
        quality_of_consensus,
    )

    # compute consensus stats for alternative consensus methods (if provided)
    if isinstance(consensus_method, list) and len(consensus_method) > 1:
        for alt_method in consensus_method[1:]:
            consensus_labels_alt, _, quality_of_consensus_alt, = compute_consensus_stats(
                labels_multiannotator=labels_multiannotator,
                pred_probs=pred_probs,
                num_annotations=num_annotations,
                consensus_method=alt_method,
                **kwargs,
            )

            (
                label_quality_scores_multiannotator[f"consensus_label_{alt_method}"],
                label_quality_scores_multiannotator[f"quality_of_consensus_{alt_method}"],
            ) = (
                consensus_labels_alt,
                quality_of_consensus_alt,
            )

    annotator_stats = compute_multiannotator_stats(labels_multiannotator, pred_probs)

    if verbose:
        print(
            "Here are various overall statistics about the annotators (column names are defined in documentation):"
        )
        print(annotator_stats.to_string())

    return (
        (label_quality_scores_multiannotator, annotator_stats)
        if return_annotator_stats
        else label_quality_scores_multiannotator
    )

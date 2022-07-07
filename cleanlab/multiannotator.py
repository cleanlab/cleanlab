import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from cleanlab.rank import get_label_quality_scores
from cleanlab.internal.label_quality_utils import _get_label_quality_scores_with_NA
from cleanlab.dataset import overall_label_health_score, _get_worst_class
import warnings


def _get_consensus_label(
    labels_multiannotator: pd.DataFrame,
    pred_probs: np.array,
    consensus_method: str,
) -> np.array:
    """Returns consensus labels aggregated from multiple annotators, determined by method specified.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape (N, M),
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.

    pred_probs : np.array
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.

    consensus_method : str (Options: ["majority"])
        The method used to aggregate labels from multiple annotators into a single consensus label. Default is by using the majority vote.
        For details of consensus methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.

    Returns
    -------
    consensus_label: np.array
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    """

    valid_methods = ["majority"]
    consensus_label = np.empty(len(labels_multiannotator), dtype="int32")

    # TODO: is there a more efficient/elegant way to do this?
    if consensus_method == "majority":
        mode_labels_multiannotator = labels_multiannotator.mode(axis=1)
        for i in range(len(mode_labels_multiannotator)):
            consensus_label[i] = mode_labels_multiannotator.iloc[i][
                pred_probs[
                    i, mode_labels_multiannotator.iloc[i].dropna().astype(int).to_numpy()
                ].argmax()
            ]
    else:
        raise ValueError(
            f"""
            {consensus_method} is not a valid consensus method!
            Please choose a valid consensus_method: {valid_methods}
            """
        )

    return consensus_label


def _get_annotator_agreement(
    labels_multiannotator: pd.DataFrame,
    consensus_label: np.array,
) -> np.array:
    """Returns the fractions of annotators that agree with the consensus labels. Note that the
    fraction for each example only considers the annotators that labeled that particular example.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape (N, M),
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.

    consensus_label : np.array
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.

    Returns
    -------
    annotator_agreement : np.array
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.
    """
    return (
        labels_multiannotator.assign(consensus_label=consensus_label)
        .apply(
            lambda s: np.mean(s.drop("consensus_label").dropna() == s["consensus_label"]),
            axis=1,
        )
        .to_numpy()
    )


# TODO: add other ways to the get quality score
def _get_quality_of_consensus(
    consensus_label: np.array,
    pred_probs: np.array,
    num_annotations: np.array,
    annotator_agreement: np.array,
    quality_method: str,
    kwargs: dict = {},
) -> np.array:
    """Return scores representing quality of the consensus label for each example,
    calculated using weighted product of `label_quality_score` of `consensus_label` and `annotator_agreement`

    Parameters
    ----------
    consensus_label : np.array
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.

    pred_probs : np.array
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.

    num_annotations : np.array
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.

    annotator_agreement : np.array
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.

    quality_method : str
        Specifies the method used to calculate the quality of the consensus label.
        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`

    Returns
    -------
    quality_of_consensus : np.array
        An array of shape ``(N,)`` with the quality score of the consensus.
        TODO: better explanation of this
    """
    valid_methods = ["auto", "lqs", "agreement"]

    if quality_method == "auto":
        lqs_consensus_label = get_label_quality_scores(consensus_label, pred_probs, **kwargs)
        frac = pred_probs[range(len(consensus_label)), (consensus_label)] / num_annotations
        quality_of_consensus = frac * lqs_consensus_label + (1 - frac) * annotator_agreement
    elif quality_method == "lqs":
        quality_of_consensus = get_label_quality_scores(consensus_label, pred_probs, **kwargs)
    elif quality_method == "agreement":
        quality_of_consensus = annotator_agreement
    else:
        raise ValueError(
            f"""
            {quality_method} is not a valid consensus method!
            Please choose a valid consensus_method: {valid_methods}
            """
        )

    return quality_of_consensus


def _get_consensus_stats(
    labels_multiannotator: pd.DataFrame,
    pred_probs: np.array,
    num_annotations: np.array,
    consensus_method: str,
    quality_method: str,
    kwargs: dict = {},
) -> tuple:
    """Returns a tuple containing the consensus labels, annotator agreement scores, and quality of consensus

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape (N, M),
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.

    pred_probs : np.array
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.

    num_annotations : np.array
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.

    consensus_method : str (Options: ["majority"])
        The method used to aggregate labels from multiple annotators into a single consensus label. Default is by using the majority vote.
        For valid consensus methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`

    quality_method : str
        Specifies the method used to calculate the quality of the consensus label.
        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`

    kwargs : dict
        Keyword arguments to pass into ``get_label_quality_scores()``.

    Returns
    ------
    tuple
        A tuple of (consensus_label, annotator_agreement, quality_of_consensus).
    """
    # Compute consensus label using method specified
    consensus_label = _get_consensus_label(
        labels_multiannotator=labels_multiannotator,
        pred_probs=pred_probs,
        consensus_method=consensus_method,
    )

    # Compute the fraction of annotator agreeing with the consensus labels
    annotator_agreement = _get_annotator_agreement(
        labels_multiannotator=labels_multiannotator,
        consensus_label=consensus_label,
    )

    # Compute the label quality scores of the consensus labels
    quality_of_consensus = _get_quality_of_consensus(
        consensus_label=consensus_label,
        pred_probs=pred_probs,
        num_annotations=num_annotations,
        annotator_agreement=annotator_agreement,
        quality_method=quality_method,
        **kwargs,
    )

    return (consensus_label, annotator_agreement, quality_of_consensus)


# TODO: update this method once multi-annotator functionality is finalized
def get_multiannotator_stats(
    labels_multiannotator: pd.DataFrame,
    pred_probs: np.array,
    # label_quality_scores_multiannotator=None, # is this still needed? it is never called in the function
) -> pd.DataFrame:
    """Returns overall statistics about each annotator.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape (N, M),
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.

    pred_probs : np.array
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.

    Returns
    -------
    annotator_stats : pd.DataFrame
        pandas DataFrame in which each row corresponds to one annotator (the row IDs correspond to annotator IDs), with columns:
        - ``overall_quality``: overall quality of a given annotator's labels
        - ``num_labeled``: number of examples annotated by a given annotator
        - ``worst_class``: the class that is most frequently mislabeled by a given annotator
    """

    def try_overall_label_health_score(labels, pred_probs):
        try:
            return overall_label_health_score(labels, pred_probs, verbose=False)
        except:
            return np.NaN

    def try_get_worst_class(labels, pred_probs):
        try:
            return _get_worst_class(labels, pred_probs)
        except:
            return np.NaN

    # Compute overall quality of each annotator's labels
    overall_label_health_score_df = labels_multiannotator.apply(
        lambda s: try_overall_label_health_score(
            s[pd.notna(s)].astype("int32"), pred_probs[pd.notna(s)]
        )
    )

    # Compute the number of labels labeled/annotated by each annotator
    num_labeled = labels_multiannotator.count()

    # Find the worst labeled class for each annotator
    worst_class = labels_multiannotator.apply(
        lambda s: try_get_worst_class(s[pd.notna(s)].astype("int32"), pred_probs[pd.notna(s)])
    )

    # Create multi-annotator stats DataFrame from its columns
    return pd.DataFrame(
        {
            "overall_quality": overall_label_health_score_df,
            "num_labeled": num_labeled,
            "worst_class": worst_class,
        }
    )


def get_label_quality_multiannotator(
    labels_multiannotator: pd.DataFrame,
    pred_probs: np.array,
    *,
    consensus_method: Union[str, List[str]] = "majority",
    quality_method: str = "auto",
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
        For a dataset with K classes, each given label must be an integer in 0, 1, ..., K-1 or
        NaN if this annotator did not label a particular example. Column names should correspond to the annotators' ID.

    pred_probs : np.array
        An array of shape ``(N, K)`` of model-predicted probabilities.
        Predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.

    consensus_method : str or List[str]
        Specifies the method used to aggregate labels from multiple annotators into a single consensus label.
        Options include:
        - ``majority``: means consensus labels are reached via a simple majority vote among annotators, with ties broken via ``pred_probs``.
        A List may be passed if you want to consider multiple methods for producing consensus labels.
        If a List is passed, then the 0th element of List is the method used to produce columns "consensus_label", "quality_of_consensus", "annotator_agreement" in the returned DataFrame.
        The 1st, 2nd, 3rd, etc. elements of this List are output as extra columns in the returned ``pandas DataFrame`` with names formatted as:
        consensus_label_SUFFIX, quality_of_consensus_SUFFIX
        where SUFFIX = the str element of this list, which must correspond to a valid method for computing consensus labels.

    quality_method : str
        Specifies the method used to calculate the quality of the consensus label.
        Options include:
        - ``auto``: TODO: CL computed score
        - ``lqs``: the label quality score of the consensus label as computed using `get_label_quality_score`
        - ``agreement``: the fraction of annotators that agree with the consensus label

    return_annotator_stats : bool = False
        Boolean to specify if `annotator_stats` is returned.

    verbose : bool = True
        If ``verbose`` is set to ``True``, the full ``annotator_stats`` DataFrame is printed out during the execution of this function.

    kwargs : dict
        Keyword arguments to pass into ``get_label_quality_scores()``.

    Returns
    -------
    label_quality_score_multiannotator : pandas.DataFrame
        pandas DataFrame in which each row corresponds to one example, with columns:
        - ``quality_of_annotator_1``, ``quality_of_annotator_2``, ..., ``quality_of_annotator_M``: the label quality score for the labels provided by annotator M (is ``NaN`` for examples which this annotator did not label).
        - ``consensus_label``: the single label that is best for each example (you can control how it is derived from all annotators' labels via the argument: ``consensus_method``)
        - ``num_annotations``: the number of annotators that have labeled each example.
        - ``annotator_agreement``: the fraction of annotators that agree with the consensus label (only consider the annotators that labeled that particular example).
        - ``quality_of_consensus``: label quality score for consensus label, calculated using weighted product of `label_quality_score` of `consensus_label` and `annotator_agreement`
        Here annotator_1, annotator_2, ..., annotator_M suffixes may be replaced column names in ``labels_multiannotator`` DataFrame used to ID the annotators.

    annotator_stats : pandas.DataFrame, optional (returned if `return_annotator_stats=True`)
        Returns overall statistics about each annotator.
        For details, see the documentation of :py:func:`get_multiannotator_stats<cleanlab.multiannotator.get_multiannotator_stats>`
    """

    # Raise error if labels_multiannotator has NaN rows
    if labels_multiannotator.isna().all(axis=1).any():
        raise ValueError("labels_multiannotator cannot have rows with all NaN.")

    # Raise error if labels_multiannotator has NaN columns
    if labels_multiannotator.isna().all().any():
        nan_columns = list(
            labels_multiannotator.columns[labels_multiannotator.isna().all() == True]
        )
        raise ValueError(
            f"""labels_multiannotator cannot have columns with all NaN.
            Annotators {nan_columns} did not label any examples."""
        )

    # Raise error if labels_multiannotator has <= 1 column
    if len(labels_multiannotator.columns) <= 1:
        raise ValueError(
            """labels_multiannotator must have more than one column. 
            If there is only one annotator, use cleanlab.rank.get_label_quality_scores instead"""
        )

    # Raise error if labels_multiannotator only has 1 label per example
    if labels_multiannotator.apply(lambda s: len(s.dropna()) == 1, axis=1).all():
        raise ValueError(
            """Each example only has one label, collapse the labels into a 1-D array and use
            cleanlab.rank.get_label_quality_scores instead"""
        )

    # Raise warning if no examples with 2 or more annotators agree
    if labels_multiannotator.apply(
        lambda s: np.array_equal(s.dropna().unique(), s.dropna()), axis=1
    ).all():
        warnings.warn("Annotators do not agree on any example. Check input data.")

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

    # Compute consensus labels, annotator agreement and quality of consensus
    consensus_label, annotator_agreement, quality_of_consensus = _get_consensus_stats(
        labels_multiannotator=labels_multiannotator,
        pred_probs=pred_probs,
        num_annotations=num_annotations,
        consensus_method=consensus_method[0]
        if isinstance(consensus_method, list)
        else consensus_method,
        quality_method=quality_method,
        **kwargs,
    )

    (
        label_quality_scores_multiannotator["annotator_agreement"],
        label_quality_scores_multiannotator["consensus_label"],
        label_quality_scores_multiannotator["quality_of_consensus"],
    ) = (
        annotator_agreement,
        consensus_label,
        quality_of_consensus,
    )

    # compute consensus stats for alternative consensus methods (if provided)
    if isinstance(consensus_method, list) and len(consensus_method) > 1:
        for alt_consensus_method in consensus_method[1:]:
            consensus_label_alt, _, quality_of_consensus_alt, = _get_consensus_stats(
                labels_multiannotator=labels_multiannotator,
                pred_probs=pred_probs,
                num_annotations=num_annotations,
                consensus_method=alt_consensus_method,
                quality_method=quality_method,
                **kwargs,
            )

            (
                label_quality_scores_multiannotator[f"consensus_label_{alt_consensus_method}"],
                label_quality_scores_multiannotator[f"quality_of_consensus_{alt_consensus_method}"],
            ) = (
                consensus_label_alt,
                quality_of_consensus_alt,
            )

    # reordering the columns (return aggregated statistics first)
    lqsm_columns = label_quality_scores_multiannotator.columns.tolist()
    new_index = (
        lqsm_columns[labels_multiannotator.shape[1] :]
        + lqsm_columns[: labels_multiannotator.shape[1]]
    )
    label_quality_scores_multiannotator = label_quality_scores_multiannotator.reindex(
        columns=new_index
    )

    if verbose or return_annotator_stats:
        annotator_stats = get_multiannotator_stats(labels_multiannotator, pred_probs)

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

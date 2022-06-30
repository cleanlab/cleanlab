import numpy as np
from typing import List, Union, Tuple
from cleanlab.rank import get_label_quality_scores
from cleanlab.dataset import overall_label_health_score, _get_worst_class
import pandas as pd

# new functions - unsure where to place them yet
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


def compute_multiannotator_stats(
    labels_multiannotator,
    pred_probs,
    label_quality_scores_multiannotator=None,
):
    # TODO: compute agreement_with_consensus
    # Compute the overall label health score for each annotator
    overall_label_health_score_df = labels_multiannotator.apply(
        overall_label_health_score, args=[pred_probs], verbose=False
    )

    # Compute the number of labels labeled/annotated by each annotator
    num_labeled = labels_multiannotator.count()

    # Find the worst labeled class for each annotator
    worst_class = labels_multiannotator.apply(_get_worst_class, args=[pred_probs])

    #

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
        - ``annotator_agreement``: the fraction of annotators that agree with the label chosen by the majority of the annotators (only considering those annotators that labeled a particular example).
        - ``quality_of_consensus``: label quality score that quantifies how likely the consensus_label is correct,
          calculated using weighted product of `label_quality_score` of `consensus_labels` and `annotator_agreement`
        Here annotator_1, annotator_2, ..., annotator_M suffixes may be replaced column names in ``labels_multiannotator`` DataFrame used to ID the annotators.
    annotator_stats : pandas.DataFrame
        TODO
    """

    consensus_methods = ["majority"]

    # Raise error if consensus_method is a not valid method
    if not consensus_method in consensus_methods:
        raise ValueError(
            f"""
                {consensus_method} is not a valid consensus method!
                Please choose a valid consensus_method: {consensus_methods}
            """
        )

    # Raise error if labels_multiannotator has np.NaN rows
    if labels_multiannotator.isna().all(axis=1).any():
        raise ValueError(
            f"""
                labels_multiannotator cannot have rows with all np.NaN.
            """
        )

    # Raise error if labels_multiannotator has <= 1 column

    # Raise error if labels_multiannotator only has 1 set of labels

    # Raise warning if no examples with 2 or more annotators agree

    # Count number of non-NaN values for each example
    num_annotations = labels_multiannotator.count(axis=1)

    # Compute the label quality scores for each annotators' labels
    label_quality_scores_multiannotator = labels_multiannotator.apply(
        _get_label_quality_scores_with_NA, args=[pred_probs], **kwargs
    )

    # Prefix column name referring to the annotators' label quality scores
    label_quality_scores_multiannotator = label_quality_scores_multiannotator.add_prefix(
        "quality_annotator_"
    )

    # Compute the consensus_labels
    # TODO: conditional based on consensus_method, consensus_method can be a List[str], add dawid-skene
    mode_labels_multiannotator = labels_multiannotator.mode(axis=1)
    consensus_labels = []
    for i in range(len(mode_labels_multiannotator)):
        consensus_labels.append(
            int(
                mode_labels_multiannotator.iloc[i][
                    pred_probs[i][
                        mode_labels_multiannotator.iloc[i].dropna().astype(int).to_numpy()
                    ].argmax()
                ]
            )
        )

    # Compute the fraction of annotator agreeing with the consensus labels
    annotator_agreement = labels_multiannotator.assign(consensus_label=consensus_labels).apply(
        lambda s: np.mean(s.drop("consensus_label") == s["consensus_label"]),
        axis=1,
    )

    # Compute the label quality scores of the consensus labels
    lqs_of_consensus = get_label_quality_scores(consensus_labels, pred_probs, **kwargs)
    frac = pred_probs[range(len(consensus_labels)), (consensus_labels)] / num_annotations
    # quality_of_consensus = frac * lqs_of_consensus + (1 - frac) * (1 - annotator_disagreement)
    quality_of_consensus = frac * lqs_of_consensus + (1 - frac) * annotator_agreement

    # Concatenate additional columns to the label_quality_scores_multiannotator DataFrame
    (
        label_quality_scores_multiannotator["consensus_label"],
        label_quality_scores_multiannotator["num_annotations"],
        label_quality_scores_multiannotator["annotator_agreement"],
        label_quality_scores_multiannotator["quality_of_consensus"],
    ) = (consensus_labels, num_annotations, annotator_agreement, quality_of_consensus)

    # annotator_stats = compute_multiannotator_stats(labels_multiannotator, pred_probs)

    # if verbose:
    #     print(
    #         "Here are various overall statistics about the annotators (column names are defined in documentation):"
    #     )
    #     print(annotator_stats.to_string())

    # return (
    #     (label_quality_scores_multiannotator, annotator_stats)
    #     if return_annotator_stats
    #     else label_quality_scores_multiannotator
    # )

    return label_quality_scores_multiannotator

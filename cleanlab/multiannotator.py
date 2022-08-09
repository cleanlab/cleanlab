import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from cleanlab.rank import get_label_quality_scores
from cleanlab.internal.util import get_num_classes
from cleanlab.internal.label_quality_utils import _get_label_quality_scores_with_NA
from cleanlab.dataset import _get_worst_class
import warnings


def convert_long_to_wide_dataset(
    labels_multiannotator_long: pd.DataFrame,
) -> pd.DataFrame:
    """Converts a long format dataset to wide format which is suitable for passing into
    :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    Dataframe must contain three columns named:
    1. ``task`` representing each example labeled by the annotators
    2. ``annotator`` representing each annotator
    3. ``label`` representing the label given by an annotator for the corresponding task

    Parameters
    ----------
    labels_multiannotator_long : pd.DataFrame
        pandas DataFrame in long format with three columns named ``task``, ``annotator`` and ``label``

    Returns
    -------
    labels_multiannotator_wide : pd.DataFrame
        pandas DataFrame of the proper format to be passed as ``labels_multiannotator`` for the other ``cleanlab.multiannotator`` functions.
    """
    labels_multiannotator_wide = labels_multiannotator_long.pivot(
        index="task", columns="annotator", values="label"
    )
    labels_multiannotator_wide.index.name = None
    labels_multiannotator_wide.columns.name = None
    return labels_multiannotator_wide


def get_consensus_label(
    labels_multiannotator: pd.DataFrame or np.ndarray,
    pred_probs: np.ndarray = None,
    consensus_method: str = "majority",
) -> np.ndarray:
    """Returns a single consensus label for each example, aggregated from the labels given by multiple annotators.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame or np.ndarray
        2D pandas DataFrame of multiple given labels for each example with shape (N, M),
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    pred_probs : np.ndarray, optional
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    consensus_method : str (Options: ["majority"])
        The method used to aggregate labels from multiple annotators into a single consensus label. Default is by using the majority vote.
        For details of consensus methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.

    Returns
    -------
    consensus_label: np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    """

    if isinstance(labels_multiannotator, np.ndarray):
        labels_multiannotator = pd.DataFrame(labels_multiannotator)

    valid_methods = ["majority"]

    if consensus_method == "majority":
        consensus_label = np.full(len(labels_multiannotator), np.nan)
        mode_labels_multiannotator = labels_multiannotator.mode(axis=1)

        nontied_idx = []
        tied_idx = dict()

        # obtaining consensus using annotator majority vote
        for idx in range(len(mode_labels_multiannotator)):
            label_mode = mode_labels_multiannotator.iloc[idx].dropna().astype(int).to_numpy()
            if len(label_mode) == 1:
                consensus_label[idx] = label_mode[0]
                nontied_idx.append(idx)
            else:
                tied_idx[idx] = label_mode

        # tiebreak 1: using pred_probs (if provided)
        if len(tied_idx) > 0 and pred_probs is not None:
            for idx, label_mode in tied_idx.copy().items():
                max_pred_probs = np.where(
                    pred_probs[idx, label_mode] == pred_probs[idx, label_mode].max()
                )[0]
                if len(max_pred_probs) == 1:
                    consensus_label[idx] = label_mode[max_pred_probs[0]]
                    del tied_idx[idx]
                else:
                    tied_idx[idx] = label_mode[max_pred_probs]

        # tiebreak 2: using empirical class frequencies
        if len(tied_idx) > 0:
            class_frequencies = (
                labels_multiannotator.apply(lambda s: s.value_counts(), axis=1).sum().values
            )
            for idx, label_mode in tied_idx.copy().items():
                max_frequency = np.where(
                    class_frequencies[label_mode] == class_frequencies[label_mode].max()
                )[0]
                if len(max_frequency) == 1:
                    consensus_label[idx] = label_mode[max_frequency[0]]
                    del tied_idx[idx]
                else:
                    tied_idx[idx] = label_mode[max_frequency]

        # tiebreak 3: using initial annotator quality scores
        if len(tied_idx) > 0:
            nontied_consensus_label = consensus_label[nontied_idx]
            nontied_labels_multiannotator = labels_multiannotator.iloc[nontied_idx]
            annotator_agreement_with_consensus = nontied_labels_multiannotator.apply(
                lambda s: np.mean(s[pd.notna(s)] == nontied_consensus_label[pd.notna(s)]),
                axis=0,
            ).to_numpy()
            for idx, label_mode in tied_idx.copy().items():
                label_quality_score = np.array(
                    [
                        np.mean(
                            annotator_agreement_with_consensus[
                                labels_multiannotator.iloc[idx][
                                    labels_multiannotator.iloc[idx] == label
                                ].index.values
                            ]
                        )
                        for label in label_mode
                    ]
                )
                max_score = np.where(label_quality_score == label_quality_score.max())[0]
                if len(max_score) == 1:
                    consensus_label[idx] = label_mode[max_score[0]]
                    del tied_idx[idx]
                else:
                    tied_idx[idx] = label_mode[max_score]

        # if still tied, break by random selection
        if len(tied_idx) > 0:
            warnings.warn(
                f"breaking ties of examples {list(tied_idx.keys())} by random selection, you may want to set seed for reproducability"
            )
            for idx, label_mode in tied_idx.items():
                consensus_label[idx] = np.random.choice(label_mode)

    else:
        raise ValueError(
            f"""
            {consensus_method} is not a valid consensus method!
            Please choose a valid consensus_method: {valid_methods}
            """
        )

    return consensus_label.astype("int64")


def _get_annotator_agreement(
    labels_multiannotator: pd.DataFrame,
    consensus_label: np.ndarray,
) -> np.ndarray:
    """Returns the fractions of annotators that agree with the consensus labels. Note that the
    fraction for each example only considers the annotators that labeled that particular example.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape (N, M),
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.

    Returns
    -------
    annotator_agreement : np.ndarray
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.
    """
    annotator_agreement = labels_multiannotator.assign(consensus_label=consensus_label).apply(
        lambda s: np.mean(s.drop("consensus_label").dropna() == s["consensus_label"]),
        axis=1,
    )
    return annotator_agreement.to_numpy()


def _get_annotator_agreement_with_annotators(
    labels_multiannotator: pd.DataFrame,
    num_annotations: np.ndarray,
) -> np.ndarray:
    """Returns the average agreement of each annotators with other annotatos that label the same examples.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape (N, M),
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.

    Returns
    -------
    annotator_agreement : np.ndarray
        An array of shape ``(M,)`` where M is the number of annotators, with the agreement of each annotator with other
        annotators that labeled the same examples.
    """

    def get_single_annotator_agreement(
        labels_multiannotator: pd.DataFrame,
        num_annotations: np.ndarray,
        annotator_id: int or str,  # TODO: unknown type, index?
    ):
        annotator_agreement_per_example = labels_multiannotator.apply(
            lambda s: np.mean(s[pd.notna(s)].drop(annotator_id) == s[annotator_id]), axis=1
        )
        np.nan_to_num(annotator_agreement_per_example, copy=False, nan=0)
        annotator_agreement = np.average(
            annotator_agreement_per_example, weights=num_annotations - 1
        )
        return annotator_agreement

    annotator_agreement_with_annotators = labels_multiannotator.apply(
        lambda s: get_single_annotator_agreement(
            labels_multiannotator[pd.notna(s)], num_annotations[pd.notna(s)], s.name
        )
    )

    return annotator_agreement_with_annotators.to_numpy()


def _get_post_pred_probs(
    labels_multiannotator: pd.DataFrame,
    consensus_label: np.ndarray,
    prior_pred_probs: np.ndarray,
    num_annotations: np.ndarray,
    annotator_agreement: np.ndarray,
    quality_method: str,
) -> np.ndarray:
    """Return the posterior predicted probabilites of each example given a specified quality method.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape (N, M),
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    prior_pred_probs : np.ndarray
        TODO: more detailed explanation that this is the model pred_probs
        An array of shape ``(N, K)`` of posterior predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    num_annotations : np.ndarray
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.
    annotator_agreement : np.ndarray
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.
    quality_method : str
        Specifies the method used to calculate the quality of the consensus label.
        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`

    Returns
    -------
    post_pred_probs : np.ndarray
        An array of shape ``(N, K)`` with the posterior predicted probabilities.
    """
    valid_methods = [
        "auto",
        "agreement",
    ]

    if quality_method == "auto":
        num_classes = get_num_classes(pred_probs=prior_pred_probs)
        likelihood = np.mean(
            annotator_agreement[num_annotations != 1]
        )  # likelihood that any annotator will annotate the consensus label for any example

        # compute adjusted annotator agreement (used as annotator weights)
        annotator_agreement_with_annotators = _get_annotator_agreement_with_annotators(
            labels_multiannotator, num_annotations
        )
        annotator_error = 1 - annotator_agreement_with_annotators
        most_likely_class_error = np.mean(
            consensus_label != np.argmax(np.bincount(consensus_label, minlength=num_classes))
        )
        adjusted_annotator_agreement = np.clip(
            1 - (annotator_error / most_likely_class_error), a_min=0, a_max=None
        )

        # compute model weight
        mask = num_annotations != 1
        consensus_label_subset = consensus_label[mask]
        prior_pred_probs_subset = prior_pred_probs[mask]

        model_error = np.mean(np.argmax(prior_pred_probs_subset, axis=1) != consensus_label_subset)
        model_weight = (1 - (model_error / most_likely_class_error)) * np.sqrt(
            np.mean(num_annotations)
        )

        # compute weighted average
        post_pred_probs = np.full(prior_pred_probs.shape, np.nan)
        for i in range(len(labels_multiannotator)):
            example = labels_multiannotator.iloc[i].dropna()
            post_pred_probs[i] = [
                np.average(
                    [prior_pred_probs[i, true_label]]
                    + [
                        likelihood
                        if annotator_label == true_label
                        else (1 - likelihood) / (num_classes - 1)
                        for annotator_label in example
                    ],
                    weights=np.concatenate(
                        ([model_weight], adjusted_annotator_agreement[example.index])
                    ),
                )
                for true_label in range(num_classes)
            ]

    elif quality_method == "agreement":
        label_counts = labels_multiannotator.apply(lambda s: s.value_counts(), axis=1)
        label_counts = np.nan_to_num(label_counts, nan=0)
        post_pred_probs = label_counts / num_annotations.reshape(-1, 1)

    else:
        raise ValueError(
            f"""
            {quality_method} is not a valid quality method!
            Please choose a valid quality_method: {valid_methods}
            """
        )

    return post_pred_probs


def _get_quality_of_consensus(
    labels_multiannotator: pd.DataFrame,
    consensus_label: np.ndarray,
    pred_probs: np.ndarray,
    num_annotations: np.ndarray,
    annotator_agreement: np.ndarray,
    quality_method: str,
    kwargs: dict = {},
) -> np.ndarray:
    """Return scores representing quality of the consensus label for each example,
    calculated using weighted product of `label_quality_score` of `consensus_label` and `annotator_agreement`

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape (N, M),
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    pred_probs : np.ndarray
        An array of shape ``(N, K)`` of posterior predicted probabilities, ``P(label=k|x)``.
        TODO: specify that we are using post_pred_probs here
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    num_annotations : np.ndarray
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.
    annotator_agreement : np.ndarray
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.
    quality_method : str
        Specifies the method used to calculate the quality of the consensus label.
        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`

    Returns
    -------
    quality_of_consensus : np.ndarray
        An array of shape ``(N,)`` with the quality score of the consensus.
    """

    valid_methods = [
        "auto",
        "agreement",
    ]

    if quality_method == "auto":
        quality_of_consensus = get_label_quality_scores(consensus_label, pred_probs, **kwargs)

    elif quality_method == "agreement":
        quality_of_consensus = annotator_agreement

    else:
        raise ValueError(
            f"""
            {quality_method} is not a valid consensus quality method!
            Please choose a valid quality_method: {valid_methods}
            """
        )

    return quality_of_consensus


def _get_consensus_stats(
    labels_multiannotator: pd.DataFrame,
    pred_probs: np.ndarray,
    num_annotations: np.ndarray,
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
    pred_probs : np.ndarray
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    num_annotations : np.ndarray
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
        A tuple of (consensus_label, annotator_agreement, quality_of_consensus, post_pred_probs).
    """

    # get consensus labels
    consensus_label = get_consensus_label(
        labels_multiannotator=labels_multiannotator,
        pred_probs=pred_probs,
        consensus_method=consensus_method,
    )

    # compute the fraction of annotator agreeing with the consensus labels
    annotator_agreement = _get_annotator_agreement(
        labels_multiannotator=labels_multiannotator,
        consensus_label=consensus_label,
    )

    # compute posterior predicted probabilites
    post_pred_probs = _get_post_pred_probs(
        labels_multiannotator=labels_multiannotator,
        consensus_label=consensus_label,
        prior_pred_probs=pred_probs,
        num_annotations=num_annotations,
        annotator_agreement=annotator_agreement,
        quality_method=quality_method,
    )

    # compute quality of the consensus labels
    quality_of_consensus = _get_quality_of_consensus(
        labels_multiannotator=labels_multiannotator,
        consensus_label=consensus_label,
        pred_probs=post_pred_probs,
        num_annotations=num_annotations,
        annotator_agreement=annotator_agreement,
        quality_method=quality_method,
        **kwargs,
    )

    return (
        consensus_label,
        annotator_agreement,
        quality_of_consensus,
        post_pred_probs,
    )


def _get_annotator_quality(
    labels_multiannotator: pd.DataFrame,
    pred_probs: np.ndarray,
    consensus_label: np.ndarray,
    num_annotations: np.ndarray,
    annotator_agreement: np.ndarray,
    quality_of_consensus: np.ndarray,
    quality_method: str = "auto",
) -> pd.DataFrame:
    """Returns annotator quality score for each annotator.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape (N, M),
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    pred_probs : np.ndarray
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    num_annotations : np.ndarray
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.
    annotator_agreement : np.ndarray
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.
    quality_method : str
        Specifies the method used to calculate the quality of the annotators.
        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`

    Returns
    -------
    overall_annotator_quality : np.ndarray
        Overall quality of a given annotator's labels
    """

    valid_methods = [
        "auto",
        "agreement",
    ]

    if quality_method == "auto":
        mask = num_annotations != 1
        labels_multiannotator_subset = labels_multiannotator[mask]
        consensus_label_subset = consensus_label[mask]
        quality_of_consensus_subset = quality_of_consensus[mask]

        annotator_quality = labels_multiannotator_subset.apply(
            lambda s: np.average(
                s[pd.notna(s)] == consensus_label_subset[pd.notna(s)],
                weights=quality_of_consensus_subset[pd.notna(s)],
            ),
            axis=0,
        )

    elif quality_method == "agreement":
        mask = num_annotations != 1
        labels_multiannotator_subset = labels_multiannotator[mask]
        consensus_label_subset = consensus_label[mask]

        annotator_quality = labels_multiannotator_subset.apply(
            lambda s: np.mean(s[pd.notna(s)] == consensus_label_subset[pd.notna(s)]),
            axis=0,
        )

    else:
        raise ValueError(
            f"""
            {quality_method} is not a valid annotator quality method!
            Please choose a valid quality_method: {valid_methods}
            """
        )

    return annotator_quality


def get_annotator_stats(
    labels_multiannotator: pd.DataFrame,
    pred_probs: np.ndarray,
    consensus_label: np.ndarray,
    num_annotations: np.ndarray,
    annotator_agreement: np.ndarray,
    quality_of_consensus: np.ndarray,
    quality_method: str = "auto",
) -> pd.DataFrame:
    """Returns overall statistics about each annotator.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape (N, M),
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    pred_probs : np.ndarray
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    num_annotations : np.ndarray
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.
    annotator_agreement : np.ndarray
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.
    quality_method : str
        Specifies the method used to calculate the quality of the consensus label.
        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`

    Returns
    -------
    annotator_stats : pd.DataFrame
        pandas DataFrame in which each row corresponds to one annotator (the row IDs correspond to annotator IDs), with columns:
        - ``overall_quality``: overall quality of a given annotator's labels
        - ``num_labeled``: number of examples annotated by a given annotator
        - ``agreement_with_consensus``: fraction of examples where a given annotator agrees with the consensus label
        - ``worst_class``: the class that is most frequently mislabeled by a given annotator
    """

    def try_get_worst_class(labels, pred_probs):
        try:
            return _get_worst_class(labels, pred_probs)
        except:
            warnings.warn(
                "worst_class labels for some annotators are NaN due to missing class labels"
            )
            return np.NaN

    annotator_quality = _get_annotator_quality(
        labels_multiannotator=labels_multiannotator,
        pred_probs=pred_probs,
        consensus_label=consensus_label,
        num_annotations=num_annotations,
        annotator_agreement=annotator_agreement,
        quality_of_consensus=quality_of_consensus,
        quality_method=quality_method,
    )

    # Compute the number of labels labeled/ by each annotator
    num_labeled = labels_multiannotator.count()

    # Compute the fraction of labels annotated by each annotator that agrees with the consensus label
    # TODO: check if we should drop singleton labels here
    agreement_with_consensus = labels_multiannotator.apply(
        lambda s: np.mean(s[pd.notna(s)] == consensus_label[pd.notna(s)]),
        axis=0,
    ).to_numpy()

    # Find the worst labeled class for each annotator
    worst_class = labels_multiannotator.apply(
        lambda s: try_get_worst_class(s[pd.notna(s)], pred_probs[pd.notna(s)])
    )

    # Create multi-annotator stats DataFrame from its columns
    annotator_stats = pd.DataFrame(
        {
            "annotator_quality": annotator_quality,
            "num_labeled": num_labeled,
            "agreement_with_consensus": agreement_with_consensus,
            "worst_class": worst_class,
        }
    )

    return annotator_stats.sort_values(by=["annotator_quality", "agreement_with_consensus"])


def get_label_quality_multiannotator(
    labels_multiannotator: pd.DataFrame or np.ndarray,
    pred_probs: np.ndarray,
    *,
    consensus_method: Union[str, List[str]] = "majority",
    quality_method: str = "auto",
    return_annotator_stats: bool = False,  # sort by lowest overall_quality first
    return_post_pred_probs: bool = False,  # TODO: remove after benchmarking
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
    pred_probs : np.ndarray
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

    if isinstance(labels_multiannotator, np.ndarray):
        labels_multiannotator = pd.DataFrame(labels_multiannotator)

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
    # TODO: might shift this later in the code to avoid extra compute
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
    # TODO: are we still allowing list of consensus methods?
    (
        consensus_label,
        annotator_agreement,
        quality_of_consensus,
        post_pred_probs,
    ) = _get_consensus_stats(
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
            consensus_label_alt, _, quality_of_consensus_alt, _ = _get_consensus_stats(
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

    lqsm_columns = label_quality_scores_multiannotator.columns.tolist()
    new_index = (
        lqsm_columns[labels_multiannotator.shape[1] :]
        + lqsm_columns[: labels_multiannotator.shape[1]]
    )
    label_quality_scores_multiannotator = label_quality_scores_multiannotator.reindex(
        columns=new_index
    )

    if verbose or return_annotator_stats:
        annotator_stats = get_annotator_stats(
            labels_multiannotator=labels_multiannotator,
            pred_probs=post_pred_probs,
            consensus_label=consensus_label,
            num_annotations=num_annotations,
            annotator_agreement=annotator_agreement,
            quality_of_consensus=quality_of_consensus,
            quality_method=quality_method,
        )

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

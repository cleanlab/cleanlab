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
Methods for analysis of classification data labeled by multiple annotators.

To analyze a fixed dataset labeled by multiple annotators, use the
:py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>` function which estimates:

* A consensus label for each example that aggregates the individual annotations more accurately than alternative aggregation via majority-vote or other algorithms used in crowdsourcing like Dawid-Skene.
* A quality score for each consensus label which measures our confidence that this label is correct.
* An analogous label quality score for each individual label chosen by one annotator for a particular example.
* An overall quality score for each annotator which measures our confidence in the overall correctness of labels obtained from this annotator.

The underlying algorithms used to compute the statistics are described in `the CROWDLAB paper <https://arxiv.org/abs/2210.06812>`_.

If you have some labeled and unlabeled data (with multiple annotators for some labeled examples) and want to decide what data to collect additional labels for,
use the :py:func:`get_active_learning_scores <cleanlab.multiannotator.get_active_learning_scores>` function, which is intended for active learning. 
This function estimates an active learning quality score for each example,
which can be used to prioritize which examples are most informative to collect additional labels for.
This function is effective for settings where some examples have been labeled by one or more annotators and other examples can have no labels at all so far,
as well as settings where new labels are collected either in batches of examples or one at a time. 
Here is an `example notebook <https://github.com/cleanlab/examples/blob/master/active_learning_multiannotator/active_learning.ipynb>`_ showcasing the use of this function in multiple active learning rounds.

Each of the main functions in this module utilizes any trained classifier model.
Variants of these functions are provided for settings where you have trained an ensemble of multiple models.
"""

import warnings
import numpy as np
import pandas as pd

from typing import List, Dict, Any, Union, Tuple, Optional

from cleanlab.rank import get_label_quality_scores
from cleanlab.internal.util import get_num_classes, value_counts
from cleanlab.internal.multiannotator_utils import (
    assert_valid_inputs_multiannotator,
    assert_valid_pred_probs,
    check_consensus_label_classes,
    find_best_temp_scaler,
    temp_scale_pred_probs,
)


def get_label_quality_multiannotator(
    labels_multiannotator: Union[pd.DataFrame, np.ndarray],
    pred_probs: np.ndarray,
    *,
    consensus_method: Union[str, List[str]] = "best_quality",
    quality_method: str = "crowdlab",
    calibrate_probs: bool = False,
    return_detailed_quality: bool = True,
    return_annotator_stats: bool = True,
    return_weights: bool = False,
    verbose: bool = True,
    label_quality_score_kwargs: dict = {},
) -> Dict[str, Any]:
    """Returns label quality scores for each example and for each annotator.

    This function is for multiclass classification datasets where examples have been labeled by
    multiple annotators (not necessarily the same number of annotators per example).

    It computes one consensus label for each example that best accounts for the labels chosen by each
    annotator (and their quality), as well as a consensus quality score for how confident we are that this consensus label is actually correct.
    It also computes similar quality scores for each annotator's individual labels, and the quality of each annotator.
    Scores are between 0 and 1; lower scores indicate labels/annotators less likely to be correct.

    To decide what data to collect additional labels for, try the :py:func:`get_active_learning_scores <cleanlab.multiannotator.get_active_learning_scores>`
    function, which is intended for active learning with multiple annotators.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame of np.ndarray
        2D pandas DataFrame or array of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        ``labels_multiannotator[n][m]`` = label for n-th example given by m-th annotator.

        For a dataset with K classes, each given label must be an integer in 0, 1, ..., K-1 or ``NaN`` if this annotator did not label a particular example.
        If you have string or other differently formatted labels, you can convert them to the proper format using :py:func:`format_multiannotator_labels <cleanlab.internal.multiannotator_utils.format_multiannotator_labels>`.
        If pd.DataFrame, column names should correspond to each annotator's ID.
    pred_probs : np.ndarray
        An array of shape ``(N, K)`` of predicted class probabilities from a trained classifier model.
        Predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
    consensus_method : str or List[str], default = "majority_vote"
        Specifies the method used to aggregate labels from multiple annotators into a single consensus label.
        Options include:

        * ``majority_vote``: consensus obtained using a simple majority vote among annotators, with ties broken via ``pred_probs``.
        * ``best_quality``: consensus obtained by selecting the label with highest label quality (quality determined by method specified in ``quality_method``).

        A List may be passed if you want to consider multiple methods for producing consensus labels.
        If a List is passed, then the 0th element of the list is the method used to produce columns `consensus_label`, `consensus_quality_score`, `annotator_agreement` in the returned DataFrame.
        The remaning (1st, 2nd, 3rd, etc.) elements of this list are output as extra columns in the returned pandas DataFrame with names formatted as:
        `consensus_label_SUFFIX`, `consensus_quality_score_SUFFIX` where `SUFFIX` = each element of this
        list, which must correspond to a valid method for computing consensus labels.
    quality_method : str, default = "crowdlab"
        Specifies the method used to calculate the quality of the consensus label.
        Options include:

        * ``crowdlab``: an emsemble method that weighs both the annotators' labels as well as the model's prediction.
        * ``agreement``: the fraction of annotators that agree with the consensus label.
    calibrate_probs : bool, default = False
        Boolean value that specifies whether the provided `pred_probs` should be re-calibrated to better match the annotators' empirical label distribution.
        We recommend setting this to True in active learning applications, in order to prevent overconfident models from suggesting the wrong examples to collect labels for.
    return_detailed_quality: bool, default = True
        Boolean to specify if `detailed_label_quality` is returned.
    return_annotator_stats : bool, default = True
        Boolean to specify if `annotator_stats` is returned.
    return_weights : bool, default = False
        Boolean to specify if `model_weight` and `annotator_weight` is returned.
        Model and annotator weights are applicable for ``quality_method == crowdlab``, will return ``None`` for any other quality methods.
    verbose : bool, default = True
        Important warnings and other printed statements may be suppressed if ``verbose`` is set to ``False``.
    label_quality_score_kwargs : dict, optional
        Keyword arguments to pass into :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.

    Returns
    -------
    labels_info : dict
        Dictionary containing up to 5 pandas DataFrame with keys as below:

        ``label_quality`` : pandas.DataFrame
            pandas DataFrame in which each row corresponds to one example, with columns:

            * ``num_annotations``: the number of annotators that have labeled each example.
            * ``consensus_label``: the single label that is best for each example (you can control how it is derived from all annotators' labels via the argument: ``consensus_method``).
            * ``annotator_agreement``: the fraction of annotators that agree with the consensus label (only consider the annotators that labeled that particular example).
            * ``consensus_quality_score``: label quality score for consensus label, calculated by the method specified in ``quality_method``.

        ``detailed_label_quality`` : pandas.DataFrame
            Only returned if `return_detailed_quality=True`.
            Returns a pandas DataFrame with columns `quality_annotator_1`, `quality_annotator_2`, ..., `quality_annotator_M` where each entry is
            the label quality score for the labels provided by each annotator (is ``NaN`` for examples which this annotator did not label).

        ``annotator_stats`` : pandas.DataFrame
            Only returned if `return_annotator_stats=True`.
            Returns overall statistics about each annotator, sorted by lowest annotator_quality first.
            pandas DataFrame in which each row corresponds to one annotator (the row IDs correspond to annotator IDs), with columns:

            * ``annotator_quality``: overall quality of a given annotator's labels, calculated by the method specified in ``quality_method``.
            * ``num_examples_labeled``: number of examples annotated by a given annotator.
            * ``agreement_with_consensus``: fraction of examples where a given annotator agrees with the consensus label.
            * ``worst_class``: the class that is most frequently mislabeled by a given annotator.

        ``model_weight`` : float
            Only returned if `return_weights=True`. It is only applicable for ``quality_method == crowdlab``.
            The model weight specifies the weight of classifier model in weighted averages used to estimate label quality
            This number is an estimate of how trustworthy the model is relative the annotators.

        ``annotator_weight`` : np.ndarray
            Only returned if `return_weights=True`. It is only applicable for ``quality_method == crowdlab``.
            An array of shape ``(M,)`` where M is the number of annotators, specifying the weight of each annotator in weighted averages used to estimate label quality.
            These weights are estimates of how trustworthy each annotator is relative to the other annotators.

    """

    if isinstance(labels_multiannotator, np.ndarray):
        labels_multiannotator = pd.DataFrame(labels_multiannotator)

    if return_weights == True and quality_method != "crowdlab":
        raise ValueError(
            "Model and annotator weights are only applicable to the crowdlab quality method. "
            "Either set return_weights=False or quality_method='crowdlab'."
        )

    assert_valid_inputs_multiannotator(labels_multiannotator, pred_probs)

    # Count number of non-NaN values for each example
    num_annotations = labels_multiannotator.count(axis=1).to_numpy()

    # calibrate pred_probs
    if calibrate_probs:
        optimal_temp = find_best_temp_scaler(labels_multiannotator, pred_probs)
        pred_probs = temp_scale_pred_probs(pred_probs, optimal_temp)

    if not isinstance(consensus_method, list):
        consensus_method = [consensus_method]

    if "best_quality" in consensus_method or "majority_vote" in consensus_method:
        majority_vote_label = get_majority_vote_label(
            labels_multiannotator=labels_multiannotator,
            pred_probs=pred_probs,
            verbose=False,
        )
        (
            MV_annotator_agreement,
            MV_consensus_quality_score,
            MV_post_pred_probs,
            MV_model_weight,
            MV_annotator_weight,
        ) = _get_consensus_stats(
            labels_multiannotator=labels_multiannotator,
            pred_probs=pred_probs,
            num_annotations=num_annotations,
            consensus_label=majority_vote_label,
            quality_method=quality_method,
            verbose=verbose,
            label_quality_score_kwargs=label_quality_score_kwargs,
        )

    label_quality = pd.DataFrame({"num_annotations": num_annotations})
    valid_methods = ["majority_vote", "best_quality"]
    main_method = True

    for curr_method in consensus_method:
        # geting consensus label and stats
        if curr_method == "majority_vote":
            consensus_label = majority_vote_label
            annotator_agreement = MV_annotator_agreement
            consensus_quality_score = MV_consensus_quality_score
            post_pred_probs = MV_post_pred_probs
            model_weight = MV_model_weight
            annotator_weight = MV_annotator_weight

        elif curr_method == "best_quality":
            consensus_label = np.full(len(majority_vote_label), np.nan)
            for i in range(len(consensus_label)):
                max_pred_probs_ind = np.where(
                    MV_post_pred_probs[i] == np.max(MV_post_pred_probs[i])
                )[0]
                if len(max_pred_probs_ind) == 1:
                    consensus_label[i] = max_pred_probs_ind[0]
                else:
                    consensus_label[i] = majority_vote_label[i]
            consensus_label = consensus_label.astype("int64")  # convert all label types to int

            (
                annotator_agreement,
                consensus_quality_score,
                post_pred_probs,
                model_weight,
                annotator_weight,
            ) = _get_consensus_stats(
                labels_multiannotator=labels_multiannotator,
                pred_probs=pred_probs,
                num_annotations=num_annotations,
                consensus_label=consensus_label,
                quality_method=quality_method,
                verbose=verbose,
                label_quality_score_kwargs=label_quality_score_kwargs,
            )

        else:
            raise ValueError(
                f"""
                {curr_method} is not a valid consensus method!
                Please choose a valid consensus_method: {valid_methods}
                """
            )

        if verbose:
            # check if any classes no longer appear in the set of consensus labels
            check_consensus_label_classes(
                labels_multiannotator=labels_multiannotator,
                consensus_label=consensus_label,
                consensus_method=curr_method,
            )

        # saving stats into dataframe, computing additional stats if specified
        if main_method:
            (
                label_quality["consensus_label"],
                label_quality["consensus_quality_score"],
                label_quality["annotator_agreement"],
            ) = (
                consensus_label,
                consensus_quality_score,
                annotator_agreement,
            )

            label_quality = label_quality.reindex(
                columns=[
                    "consensus_label",
                    "consensus_quality_score",
                    "annotator_agreement",
                    "num_annotations",
                ]
            )

            if return_detailed_quality:
                # Compute the label quality scores for each annotators' labels
                detailed_label_quality = labels_multiannotator.apply(
                    _get_annotator_label_quality_score,
                    pred_probs=post_pred_probs,
                    label_quality_score_kwargs=label_quality_score_kwargs,
                )
                detailed_label_quality = detailed_label_quality.add_prefix("quality_annotator_")

            if return_annotator_stats:
                annotator_stats = _get_annotator_stats(
                    labels_multiannotator=labels_multiannotator,
                    pred_probs=post_pred_probs,
                    consensus_label=consensus_label,
                    num_annotations=num_annotations,
                    annotator_agreement=annotator_agreement,
                    model_weight=model_weight,
                    annotator_weight=annotator_weight,
                    consensus_quality_score=consensus_quality_score,
                    quality_method=quality_method,
                )

            main_method = False

        else:
            (
                label_quality[f"consensus_label_{curr_method}"],
                label_quality[f"consensus_quality_score_{curr_method}"],
                label_quality[f"annotator_agreement_{curr_method}"],
            ) = (
                consensus_label,
                consensus_quality_score,
                annotator_agreement,
            )

    labels_info = {
        "label_quality": label_quality,
    }

    if return_detailed_quality:
        labels_info["detailed_label_quality"] = detailed_label_quality
    if return_annotator_stats:
        labels_info["annotator_stats"] = annotator_stats
    if return_weights:
        labels_info["model_weight"] = model_weight
        labels_info["annotator_weight"] = annotator_weight

    return labels_info


def get_label_quality_multiannotator_ensemble(
    labels_multiannotator: Union[pd.DataFrame, np.ndarray],
    pred_probs: np.ndarray,
    *,
    calibrate_probs: bool = False,
    return_detailed_quality: bool = True,
    return_annotator_stats: bool = True,
    return_weights: bool = False,
    verbose: bool = True,
    label_quality_score_kwargs: dict = {},
) -> Dict[str, Any]:
    """Returns label quality scores for each example and for each annotator, based on predictions from an ensemble of models.

    This function is similar to :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>` but for settings where
    you have trained an ensemble of multiple classifier models rather than a single model.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame of np.ndarray
        Multiannotator labels in the same format expected by :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    pred_probs : np.ndarray
        An array of shape ``(P, N, K)`` where P is the number of models, consisting of predicted class probabilities from the ensemble models.
        Each set of predicted probabilities with shape ``(N, K)`` is in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
    calibrate_probs : bool, default = False
        Boolean value as expected by :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    return_detailed_quality: bool, default = True
        Boolean value as expected by :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    return_annotator_stats : bool, default = True
        Boolean value as expected by :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    return_weights : bool, default = False
        Boolean value as expected by :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    verbose : bool, default = True
        Boolean value as expected by :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    label_quality_score_kwargs : dict, optional
        Keyword arguments in the same format expected by py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.

    Returns
    -------
    labels_info : dict
        Dictionary containing up to 5 pandas DataFrame with keys as below:

        ``label_quality`` : pandas.DataFrame
            Similar to output as :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.

        ``detailed_label_quality`` : pandas.DataFrame
            Similar to output as :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.

        ``annotator_stats`` : pandas.DataFrame
            Similar to output as :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.

        ``model_weight`` : np.ndarray
            Only returned if `return_weights=True`.
            An array of shape ``(P,)`` where is the number of models in the ensemble, specifying the weight of each classifier model in weighted averages used to estimate label quality.
            These weigthts is an estimate of how trustworthy the model is relative the annotators.
            An array of shape ``(P,)`` where is the number of models in the ensemble, specifying the model weight used in weighted averages.

        ``annotator_weight`` : np.ndarray
            Only returned if `return_weights=True`.
            Similar to output as :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.

    See Also
    --------
    get_label_quality_multiannotator
    """
    if isinstance(labels_multiannotator, np.ndarray):
        labels_multiannotator = pd.DataFrame(labels_multiannotator)

    assert_valid_inputs_multiannotator(labels_multiannotator, pred_probs, ensemble=True)

    # Count number of non-NaN values for each example
    num_annotations = labels_multiannotator.count(axis=1).to_numpy()

    # temp scale pred_probs
    if calibrate_probs:
        for i in range(len(pred_probs)):
            curr_pred_probs = pred_probs[i]
            optimal_temp = find_best_temp_scaler(labels_multiannotator, curr_pred_probs)
            pred_probs[i] = temp_scale_pred_probs(curr_pred_probs, optimal_temp)

    label_quality = pd.DataFrame({"num_annotations": num_annotations})

    # get majority vote stats
    avg_pred_probs = np.mean(pred_probs, axis=0)
    majority_vote_label = get_majority_vote_label(
        labels_multiannotator=labels_multiannotator,
        pred_probs=avg_pred_probs,
        verbose=False,
    )
    (
        MV_annotator_agreement,
        MV_consensus_quality_score,
        MV_post_pred_probs,
        MV_model_weight,
        MV_annotator_weight,
    ) = _get_consensus_stats(
        labels_multiannotator=labels_multiannotator,
        pred_probs=pred_probs,
        num_annotations=num_annotations,
        consensus_label=majority_vote_label,
        verbose=verbose,
        ensemble=True,
        **label_quality_score_kwargs,
    )

    # get crowdlab stats
    consensus_label = np.full(len(majority_vote_label), np.nan)
    for i in range(len(consensus_label)):
        max_pred_probs_ind = np.where(MV_post_pred_probs[i] == np.max(MV_post_pred_probs[i]))[0]
        if len(max_pred_probs_ind) == 1:
            consensus_label[i] = max_pred_probs_ind[0]
        else:
            consensus_label[i] = majority_vote_label[i]
    consensus_label = consensus_label.astype("int64")  # convert all label types to int

    (
        annotator_agreement,
        consensus_quality_score,
        post_pred_probs,
        model_weight,
        annotator_weight,
    ) = _get_consensus_stats(
        labels_multiannotator=labels_multiannotator,
        pred_probs=pred_probs,
        num_annotations=num_annotations,
        consensus_label=consensus_label,
        verbose=verbose,
        ensemble=True,
        **label_quality_score_kwargs,
    )

    if verbose:
        # check if any classes no longer appear in the set of consensus labels
        check_consensus_label_classes(
            labels_multiannotator=labels_multiannotator,
            consensus_label=consensus_label,
            consensus_method="crowdlab",
        )

    (
        label_quality["consensus_label"],
        label_quality["consensus_quality_score"],
        label_quality["annotator_agreement"],
    ) = (
        consensus_label,
        consensus_quality_score,
        annotator_agreement,
    )

    label_quality = label_quality.reindex(
        columns=[
            "consensus_label",
            "consensus_quality_score",
            "annotator_agreement",
            "num_annotations",
        ]
    )

    if return_detailed_quality:
        # Compute the label quality scores for each annotators' labels
        detailed_label_quality = labels_multiannotator.apply(
            _get_annotator_label_quality_score,
            pred_probs=post_pred_probs,
            **label_quality_score_kwargs,
        )
        detailed_label_quality = detailed_label_quality.add_prefix("quality_annotator_")

    if return_annotator_stats:
        annotator_stats = _get_annotator_stats(
            labels_multiannotator=labels_multiannotator,
            pred_probs=post_pred_probs,
            consensus_label=consensus_label,
            num_annotations=num_annotations,
            annotator_agreement=annotator_agreement,
            model_weight=np.mean(model_weight),  # use average model weight when scoring annotators
            annotator_weight=annotator_weight,
            consensus_quality_score=consensus_quality_score,
        )

    labels_info = {
        "label_quality": label_quality,
    }

    if return_detailed_quality:
        labels_info["detailed_label_quality"] = detailed_label_quality
    if return_annotator_stats:
        labels_info["annotator_stats"] = annotator_stats
    if return_weights:
        labels_info["model_weight"] = model_weight
        labels_info["annotator_weight"] = annotator_weight

    return labels_info


def get_active_learning_scores(
    labels_multiannotator: Union[pd.DataFrame, np.ndarray],
    pred_probs: np.ndarray,
    pred_probs_unlabeled: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns an active learning quality score for each example in the dataset.

    We consider settings where one example can be labeled by one or more annotators and some examples have no labels at all so far.

    The score is in between 0 and 1, and can be used to prioritize what data to collect additional labels for.
    Lower scores indicate examples whose true label we are least confident about based on the current data;
    collecting additional labels for these low-scoring examples will be more informative than collecting labels for other examples.
    To use an annotation budget most efficiently, select a batch of examples with the lowest scores and collect one additional label for each example,
    and repeat this process after retraining your classifier.

    To analyze a fixed dataset labeled by multiple annotators rather than collecting additional labels, try the
    :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>` function instead.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame of np.ndarray
        2D pandas DataFrame or array of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators. Note that this function also works with
        datasets where there is only one annotator (M=1).
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
        Note that examples that have no annotator labels should not be included in this DataFrame/array.
    pred_probs : np.ndarray
        An array of shape ``(N, K)`` of predicted class probabilities from a trained classifier model.
        Predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
    pred_probs_unlabeled : np.ndarray, optional
        An array of shape ``(N, K)`` of predicted class probabilities from a trained classifier model for examples that have no annotator labels.
        Predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.

    Returns
    -------
    active_learning_scores : np.ndarray
        Array of shape ``(N,)`` indicating the active learning quality scores for each example.
        Examples with the lowest scores are those we should label next in order to maximally improve our classifier model.

    active_learning_scores_unlabeled : np.ndarray
        Array of shape ``(N,)`` indicating the active learning quality scores for each unlabeled example.
        Returns an empty array if no unlabeled data is provided.
        Examples with the lowest scores are those we should label next in order to maximally improve our classifier model
        (scores for unlabeled data are directly comparable with the `active_learning_scores` for labeled data).
    """

    if isinstance(labels_multiannotator, np.ndarray):
        labels_multiannotator = pd.DataFrame(labels_multiannotator)

    assert_valid_pred_probs(pred_probs=pred_probs, pred_probs_unlabeled=pred_probs_unlabeled)

    num_classes = get_num_classes(pred_probs=pred_probs)

    # if all examples are only labeled by a single annotator
    if labels_multiannotator.apply(lambda s: len(s.dropna()) == 1, axis=1).all():
        optimal_temp = 1.0  # do not temp scale for single annotator case, temperature is defined here for later use

        assert_valid_inputs_multiannotator(
            labels_multiannotator, pred_probs, allow_single_label=True
        )

        consensus_label = get_majority_vote_label(
            labels_multiannotator=labels_multiannotator,
            pred_probs=pred_probs,
            verbose=False,
        )
        quality_of_consensus_labeled = get_label_quality_scores(consensus_label, pred_probs)
        model_weight = 1
        annotator_weight = np.full(labels_multiannotator.shape[1], 1)
        avg_annotator_weight = np.mean(annotator_weight)

    else:
        optimal_temp = find_best_temp_scaler(labels_multiannotator, pred_probs)
        pred_probs = temp_scale_pred_probs(pred_probs, optimal_temp)

        multiannotator_info = get_label_quality_multiannotator(
            labels_multiannotator,
            pred_probs,
            return_annotator_stats=False,
            return_detailed_quality=False,
            return_weights=True,
        )

        quality_of_consensus_labeled = multiannotator_info["label_quality"][
            "consensus_quality_score"
        ]
        model_weight = multiannotator_info["model_weight"]
        annotator_weight = multiannotator_info["annotator_weight"]
        avg_annotator_weight = np.mean(annotator_weight)

    # compute scores for labeled data
    active_learning_scores = np.full(len(labels_multiannotator), np.nan)
    for i in range(len(active_learning_scores)):
        annotator_labels = labels_multiannotator.iloc[i]
        active_learning_scores[i] = np.average(
            (quality_of_consensus_labeled[i], 1 / num_classes),
            weights=(
                np.sum(annotator_weight[annotator_labels.notna()]) + model_weight,
                avg_annotator_weight,
            ),
        )

    # compute scores for unlabeled data
    if pred_probs_unlabeled is not None:
        pred_probs_unlabeled = temp_scale_pred_probs(pred_probs_unlabeled, optimal_temp)
        quality_of_consensus_unlabeled = np.max(pred_probs_unlabeled, axis=1)

        active_learning_scores_unlabeled = np.average(
            np.stack(
                [
                    quality_of_consensus_unlabeled,
                    np.full(len(quality_of_consensus_unlabeled), 1 / num_classes),
                ]
            ),
            weights=[model_weight, avg_annotator_weight],
            axis=0,
        )

    else:
        active_learning_scores_unlabeled = np.array([])

    return active_learning_scores, active_learning_scores_unlabeled


def get_active_learning_scores_ensemble(
    labels_multiannotator: Union[pd.DataFrame, np.ndarray],
    pred_probs: np.ndarray,
    pred_probs_unlabeled: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns an active learning quality score for each example in the dataset, based on predictions from an ensemble of models.

    This function is similar to :py:func:`get_active_learning_scores <cleanlab.multiannotator.get_active_learning_scores>` but allows for an
    ensemble of multiple classifier models to be trained and will aggregate predictions from the models to compute the active learning quality score.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame or np.ndarray
        Multiannotator labels in the same format expected by :py:func:`get_active_learning_scores <cleanlab.multiannotator.get_active_learning_scores>`.
    pred_probs : np.ndarray
        An array of shape ``(P, N, K)`` where P is the number of models, consisting of predicted class probabilities from the ensemble models.
        Note that this function also works with datasets where there is only one annotator (M=1).
        Each set of predicted probabilities with shape ``(N, K)`` is in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
    pred_probs_unlabeled : np.ndarray, optional
        An array of shape ``(P, N, K)`` where P is the number of models, consisting of predicted class probabilities from a trained classifier model
        for examples that have no annotated labels so far (but which we may want to label in the future, and hence compute active learning quality scores for).
        Each set of predicted probabilities with shape ``(N, K)`` is in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.

    Returns
    -------
    active_learning_scores : np.ndarray
        Similar to output as :py:func:`get_label_quality_scores <cleanlab.multiannotator.get_label_quality_scores>`.
    active_learning_scores_unlabeled : np.ndarray
        Similar to output as :py:func:`get_label_quality_scores <cleanlab.multiannotator.get_label_quality_scores>`.

    See Also
    --------
    get_active_learning_scores
    """

    if isinstance(labels_multiannotator, np.ndarray):
        labels_multiannotator = pd.DataFrame(labels_multiannotator)

    assert_valid_pred_probs(
        pred_probs=pred_probs, pred_probs_unlabeled=pred_probs_unlabeled, ensemble=True
    )

    num_classes = get_num_classes(pred_probs=pred_probs[0])

    # temp scale pred_probs

    # if all examples are only labeled by a single annotator
    if labels_multiannotator.apply(lambda s: len(s.dropna()) == 1, axis=1).all():
        # do not temp scale for single annotator case, temperature is defined here for later use
        optimal_temp = np.full(len(pred_probs), 1.0)

        assert_valid_inputs_multiannotator(
            labels_multiannotator, pred_probs, ensemble=True, allow_single_label=True
        )

        avg_pred_probs = np.mean(pred_probs, axis=0)
        consensus_label = get_majority_vote_label(
            labels_multiannotator=labels_multiannotator,
            pred_probs=avg_pred_probs,
            verbose=False,
        )
        quality_of_consensus_labeled = get_label_quality_scores(consensus_label, avg_pred_probs)
        model_weight = np.full(len(pred_probs), 1)
        annotator_weight = np.full(labels_multiannotator.shape[1], 1)
        avg_annotator_weight = np.mean(annotator_weight)

    else:
        optimal_temp = np.full(len(pred_probs), np.NaN)
        for i in range(len(pred_probs)):
            curr_pred_probs = pred_probs[i]
            curr_optimal_temp = find_best_temp_scaler(labels_multiannotator, curr_pred_probs)
            pred_probs[i] = temp_scale_pred_probs(curr_pred_probs, curr_optimal_temp)
            optimal_temp[i] = curr_optimal_temp

        multiannotator_info = get_label_quality_multiannotator_ensemble(
            labels_multiannotator,
            pred_probs,
            return_annotator_stats=False,
            return_detailed_quality=False,
            return_weights=True,
        )

        quality_of_consensus_labeled = multiannotator_info["label_quality"][
            "consensus_quality_score"
        ]
        model_weight = multiannotator_info["model_weight"]
        annotator_weight = multiannotator_info["annotator_weight"]
        avg_annotator_weight = np.mean(annotator_weight)

    # compute scores for labeled data
    active_learning_scores = np.full(len(labels_multiannotator), np.nan)
    for i in range(len(active_learning_scores)):
        annotator_labels = labels_multiannotator.iloc[i]
        active_learning_scores[i] = np.average(
            (quality_of_consensus_labeled[i], 1 / num_classes),
            weights=(
                np.sum(annotator_weight[annotator_labels.notna()]) + np.sum(model_weight),
                avg_annotator_weight,
            ),
        )

    # compute scores for unlabeled data
    if pred_probs_unlabeled is not None:
        for i in range(len(pred_probs_unlabeled)):
            pred_probs_unlabeled[i] = temp_scale_pred_probs(
                pred_probs_unlabeled[i], optimal_temp[i]
            )

        avg_pred_probs_unlabeled = np.mean(pred_probs_unlabeled, axis=0)
        consensus_label_unlabeled = get_majority_vote_label(
            np.argmax(pred_probs_unlabeled, axis=2).T,
            avg_pred_probs_unlabeled,
        )
        modified_pred_probs_unlabeled = np.average(
            np.concatenate(
                (
                    pred_probs_unlabeled,
                    np.full(pred_probs_unlabeled.shape[1:], 1 / num_classes)[np.newaxis, :, :],
                )
            ),
            weights=np.concatenate((model_weight, np.array([avg_annotator_weight]))),
            axis=0,
        )

        active_learning_scores_unlabeled = get_label_quality_scores(
            consensus_label_unlabeled, modified_pred_probs_unlabeled
        )
    else:
        active_learning_scores_unlabeled = np.array([])

    return active_learning_scores, active_learning_scores_unlabeled


def get_majority_vote_label(
    labels_multiannotator: Union[pd.DataFrame, np.ndarray],
    pred_probs: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Returns the majority vote label for each example, aggregated from the labels given by multiple annotators.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame or np.ndarray
        2D pandas DataFrame or array of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    pred_probs : np.ndarray, optional
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    verbose : bool, optional
        Important warnings and other printed statements may be suppressed if ``verbose`` is set to ``False``.
    Returns
    -------
    consensus_label: np.ndarray
        An array of shape ``(N,)`` with the majority vote label aggregated from all annotators.

        In the event of majority vote ties, ties are broken in the following order:
        using the model ``pred_probs`` (if provided) and selecting the class with highest predicted probability,
        using the empirical class frequencies and selecting the class with highest frequency,
        using an initial annotator quality score and selecting the class that has been labeled by annotators with higher quality,
        and lastly by random selection.
    """

    if isinstance(labels_multiannotator, np.ndarray):
        labels_multiannotator = pd.DataFrame(labels_multiannotator)

    if verbose:
        assert_valid_inputs_multiannotator(
            labels_multiannotator, pred_probs, allow_single_label=True
        )

    majority_vote_label = np.full(len(labels_multiannotator), np.nan)
    mode_labels_multiannotator = labels_multiannotator.mode(axis=1)

    nontied_idx = []
    tied_idx = dict()

    # obtaining consensus using annotator majority vote
    for idx in range(len(mode_labels_multiannotator)):
        label_mode = mode_labels_multiannotator.iloc[idx].dropna().astype(int).to_numpy()
        if len(label_mode) == 1:
            majority_vote_label[idx] = label_mode[0]
            nontied_idx.append(idx)
        else:
            tied_idx[idx] = label_mode

    # tiebreak 1: using pred_probs (if provided)
    if pred_probs is not None and len(tied_idx) > 0:
        for idx, label_mode in tied_idx.copy().items():
            max_pred_probs = np.where(
                pred_probs[idx, label_mode] == np.max(pred_probs[idx, label_mode])
            )[0]
            if len(max_pred_probs) == 1:
                majority_vote_label[idx] = label_mode[max_pred_probs[0]]
                del tied_idx[idx]
            else:
                tied_idx[idx] = label_mode[max_pred_probs]

    # tiebreak 2: using empirical class frequencies
    if len(tied_idx) > 0:
        if pred_probs is not None:
            num_classes = pred_probs.shape[1]
        else:
            num_classes = int(
                np.nanmax(labels_multiannotator.replace({pd.NA: np.NaN}).astype(float).values) + 1
            )
        class_frequencies = labels_multiannotator.apply(
            lambda s: pd.Series(np.bincount(s[s.notna()], minlength=num_classes)), axis=1
        ).sum()
        for idx, label_mode in tied_idx.copy().items():
            max_frequency = np.where(
                class_frequencies[label_mode] == np.max(class_frequencies[label_mode])
            )[0]
            if len(max_frequency) == 1:
                majority_vote_label[idx] = label_mode[max_frequency[0]]
                del tied_idx[idx]
            else:
                tied_idx[idx] = label_mode[max_frequency]

    # tiebreak 3: using initial annotator quality scores
    if len(tied_idx) > 0:
        nontied_majority_vote_label = majority_vote_label[nontied_idx]
        nontied_labels_multiannotator = labels_multiannotator.iloc[nontied_idx]
        annotator_agreement_with_consensus = nontied_labels_multiannotator.apply(
            lambda s: np.mean(s[pd.notna(s)] == nontied_majority_vote_label[pd.notna(s)]),
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
                majority_vote_label[idx] = label_mode[max_score[0]]
                del tied_idx[idx]
            else:
                tied_idx[idx] = label_mode[max_score]

    # if still tied, break by random selection
    if len(tied_idx) > 0:
        warnings.warn(
            f"breaking ties of examples {list(tied_idx.keys())} by random selection, you may want to set seed for reproducability"
        )
        for idx, label_mode in tied_idx.items():
            majority_vote_label[idx] = np.random.choice(label_mode)

    if verbose:
        # check if any classes no longer appear in the set of consensus labels
        check_consensus_label_classes(
            labels_multiannotator=labels_multiannotator,
            consensus_label=majority_vote_label,
            consensus_method="majority_vote",
        )

    return majority_vote_label.astype("int64")


def convert_long_to_wide_dataset(
    labels_multiannotator_long: pd.DataFrame,
) -> pd.DataFrame:
    """Converts a long format dataset to wide format which is suitable for passing into
    :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.

    Dataframe must contain three columns named:

    #. ``task`` representing each example labeled by the annotators
    #. ``annotator`` representing each annotator
    #. ``label`` representing the label given by an annotator for the corresponding task (i.e. example)

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


def _get_consensus_stats(
    labels_multiannotator: pd.DataFrame,
    pred_probs: np.ndarray,
    num_annotations: np.ndarray,
    consensus_label: np.ndarray,
    quality_method: str = "crowdlab",
    verbose: bool = True,
    ensemble: bool = False,
    label_quality_score_kwargs: dict = {},
) -> tuple:
    """Returns a tuple containing the consensus labels, annotator agreement scores, and quality of consensus

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    pred_probs : np.ndarray
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    num_annotations : np.ndarray
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    quality_method : str, default = "crowdlab" (Options: ["crowdlab", "agreement"])
        Specifies the method used to calculate the quality of the consensus label.
        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`
    label_quality_score_kwargs : dict, optional
        Keyword arguments to pass into ``get_label_quality_scores()``.
    verbose : bool, default = True
        Certain warnings and notes will be printed if ``verbose`` is set to ``True``.
    ensemble : bool, default = False
        Boolean flag to indicate whether the pred_probs passed are from ensemble models.

    Returns
    ------
    stats : tuple
        A tuple of (consensus_label, annotator_agreement, consensus_quality_score, post_pred_probs).
    """

    # compute the fraction of annotator agreeing with the consensus labels
    annotator_agreement = _get_annotator_agreement_with_consensus(
        labels_multiannotator=labels_multiannotator,
        consensus_label=consensus_label,
    )

    # compute posterior predicted probabilites
    if ensemble:
        post_pred_probs, model_weight, annotator_weight = _get_post_pred_probs_and_weights_ensemble(
            labels_multiannotator=labels_multiannotator,
            consensus_label=consensus_label,
            prior_pred_probs=pred_probs,
            num_annotations=num_annotations,
            annotator_agreement=annotator_agreement,
            quality_method=quality_method,
            verbose=verbose,
        )
    else:
        post_pred_probs, model_weight, annotator_weight = _get_post_pred_probs_and_weights(
            labels_multiannotator=labels_multiannotator,
            consensus_label=consensus_label,
            prior_pred_probs=pred_probs,
            num_annotations=num_annotations,
            annotator_agreement=annotator_agreement,
            quality_method=quality_method,
            verbose=verbose,
        )

    # compute quality of the consensus labels
    consensus_quality_score = _get_consensus_quality_score(
        consensus_label=consensus_label,
        pred_probs=post_pred_probs,
        num_annotations=num_annotations,
        annotator_agreement=annotator_agreement,
        quality_method=quality_method,
        label_quality_score_kwargs=label_quality_score_kwargs,
    )

    return (
        annotator_agreement,
        consensus_quality_score,
        post_pred_probs,
        model_weight,
        annotator_weight,
    )


def _get_annotator_stats(
    labels_multiannotator: pd.DataFrame,
    pred_probs: np.ndarray,
    consensus_label: np.ndarray,
    num_annotations: np.ndarray,
    annotator_agreement: np.ndarray,
    model_weight: np.ndarray,
    annotator_weight: np.ndarray,
    consensus_quality_score: np.ndarray,
    quality_method: str = "crowdlab",
) -> pd.DataFrame:
    """Returns a dictionary containing overall statistics about each annotator.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape ``(N, M)``,
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
    model_weight : float
        float specifying the model weight used in weighted averages,
        None if model weight is not used to compute quality scores
    annotator_weight : np.ndarray
        An array of shape ``(M,)`` where M is the number of annotators, specifying the annotator weights used in weighted averages,
        None if annotator weights are not used to compute quality scores
    consensus_quality_score : np.ndarray
        An array of shape ``(N,)`` with the quality score of the consensus.
    quality_method : str, default = "crowdlab" (Options: ["crowdlab", "agreement"])
        Specifies the method used to calculate the quality of the consensus label.
        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`

    Returns
    -------
    annotator_stats : pd.DataFrame
        Overall statistics about each annotator.
        For details, see the documentation of :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    """

    annotator_quality = _get_annotator_quality(
        labels_multiannotator=labels_multiannotator,
        pred_probs=pred_probs,
        consensus_label=consensus_label,
        num_annotations=num_annotations,
        annotator_agreement=annotator_agreement,
        model_weight=model_weight,
        annotator_weight=annotator_weight,
        quality_method=quality_method,
    )

    # Compute the number of labels labeled/ by each annotator
    num_examples_labeled = labels_multiannotator.count()

    # Compute the fraction of labels annotated by each annotator that agrees with the consensus label
    # TODO: check if we should drop singleton labels here
    agreement_with_consensus = labels_multiannotator.apply(
        lambda s: np.mean(s[pd.notna(s)] == consensus_label[pd.notna(s)]),
        axis=0,
    ).to_numpy()

    # Find the worst labeled class for each annotator
    worst_class = _get_annotator_worst_class(
        labels_multiannotator=labels_multiannotator,
        consensus_label=consensus_label,
        consensus_quality_score=consensus_quality_score,
    )

    # Create multi-annotator stats DataFrame from its columns
    annotator_stats = pd.DataFrame(
        {
            "annotator_quality": annotator_quality,
            "agreement_with_consensus": agreement_with_consensus,
            "worst_class": worst_class,
            "num_examples_labeled": num_examples_labeled,
        }
    )

    return annotator_stats.sort_values(by=["annotator_quality", "agreement_with_consensus"])


def _get_annotator_agreement_with_consensus(
    labels_multiannotator: pd.DataFrame,
    consensus_label: np.ndarray,
) -> np.ndarray:
    """Returns the fractions of annotators that agree with the consensus label per example. Note that the
    fraction for each example only considers the annotators that labeled that particular example.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape ``(N, M)``,
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
    verbose: bool = True,
) -> np.ndarray:
    """Returns the average agreement of each annotator with other annotators that label the same example.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    verbose : bool, default = True
        Certain warnings and notes will be printed if ``verbose`` is set to ``True``.

    Returns
    -------
    annotator_agreement : np.ndarray
        An array of shape ``(M,)`` where M is the number of annotators, with the agreement of each annotator with other
        annotators that labeled the same examples.
    """

    def get_single_annotator_agreement(
        labels_multiannotator: pd.DataFrame,
        num_annotations: np.ndarray,
        annotator_id: Union[int, str],
    ):
        annotator_agreement_per_example = labels_multiannotator.apply(
            lambda s: np.mean(s[pd.notna(s)].drop(annotator_id) == s[annotator_id]), axis=1
        )
        np.nan_to_num(annotator_agreement_per_example, copy=False, nan=0)
        try:
            annotator_agreement = np.average(
                annotator_agreement_per_example, weights=num_annotations - 1
            )
        except:
            annotator_agreement = np.NaN
        return annotator_agreement

    annotator_agreement_with_annotators = labels_multiannotator.apply(
        lambda s: get_single_annotator_agreement(
            labels_multiannotator[pd.notna(s)], num_annotations[pd.notna(s)], s.name
        )
    )

    # impute average annotator accuracy for any annotator that do not overlap with other annotators
    mask = annotator_agreement_with_annotators.isna()
    if np.sum(mask) > 0:
        if verbose:
            print(
                f"Annotator(s) {annotator_agreement_with_annotators[mask].index.values} did not annotate any examples that overlap with other annotators, \
                \nusing the average annotator agreeement among other annotators as this annotator's agreement."
            )

        avg_annotator_agreement = np.mean(annotator_agreement_with_annotators[~mask])
        annotator_agreement_with_annotators[mask] = avg_annotator_agreement

    return annotator_agreement_with_annotators.to_numpy()


def _get_post_pred_probs_and_weights(
    labels_multiannotator: pd.DataFrame,
    consensus_label: np.ndarray,
    prior_pred_probs: np.ndarray,
    num_annotations: np.ndarray,
    annotator_agreement: np.ndarray,
    quality_method: str = "crowdlab",
    verbose: bool = True,
) -> Tuple[np.ndarray, Any, Any]:
    """Return the posterior predicted probabilities of each example given a specified quality method.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    prior_pred_probs : np.ndarray
        An array of shape ``(N, K)`` of prior predicted probabilities, ``P(label=k|x)``, usually the out-of-sample predicted probability computed by a model.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    num_annotations : np.ndarray
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.
    annotator_agreement : np.ndarray
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.
    quality_method : str, default = "crowdlab" (Options: ["crowdlab", "agreement"])
        Specifies the method used to calculate the quality of the consensus label.
        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`
    verbose : bool, default = True
        Certain warnings and notes will be printed if ``verbose`` is set to ``True``.

    Returns
    -------
    post_pred_probs : np.ndarray
        An array of shape ``(N, K)`` with the posterior predicted probabilities.

    model_weight : float
        float specifying the model weight used in weighted averages,
        None if model weight is not used to compute quality scores

    annotator_weight : np.ndarray
        An array of shape ``(M,)`` where M is the number of annotators, specifying the annotator weights used in weighted averages,
        None if annotator weights are not used to compute quality scores

    """
    valid_methods = [
        "crowdlab",
        "agreement",
    ]

    # setting dummy variables for model and annotator weights that will be returned
    # only relevant for quality_method == crowdlab, return None for all other methods
    return_model_weight = None
    return_annotator_weight = None

    if quality_method == "crowdlab":
        num_classes = get_num_classes(pred_probs=prior_pred_probs)
        likelihood = np.mean(
            annotator_agreement[num_annotations != 1]
        )  # likelihood that any annotator will annotate the consensus label for any example

        # subsetting the dataset to only includes examples with more than one annotation
        mask = num_annotations != 1
        consensus_label_subset = consensus_label[mask]
        prior_pred_probs_subset = prior_pred_probs[mask]

        # compute most likely class error
        most_likely_class_error = np.clip(
            np.mean(
                consensus_label_subset
                != np.argmax(np.bincount(consensus_label_subset, minlength=num_classes))
            ),
            a_min=1e-6,
            a_max=None,
        )

        # compute adjusted annotator agreement (used as annotator weights)
        annotator_agreement_with_annotators = _get_annotator_agreement_with_annotators(
            labels_multiannotator, num_annotations, verbose
        )
        annotator_error = 1 - annotator_agreement_with_annotators
        adjusted_annotator_agreement = np.clip(
            1 - (annotator_error / most_likely_class_error), a_min=1e-6, a_max=None
        )

        # compute model weight
        model_error = np.mean(np.argmax(prior_pred_probs_subset, axis=1) != consensus_label_subset)
        model_weight = np.max([(1 - (model_error / most_likely_class_error)), 1e-6]) * np.sqrt(
            np.mean(num_annotations)
        )

        # compute weighted average
        post_pred_probs = np.full(prior_pred_probs.shape, np.nan)
        for i in range(len(labels_multiannotator)):
            example_mask = labels_multiannotator.iloc[i].notna()
            example = labels_multiannotator.iloc[i][example_mask]
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
                        ([model_weight], adjusted_annotator_agreement[example_mask])
                    ),
                )
                for true_label in range(num_classes)
            ]

        return_model_weight = model_weight
        return_annotator_weight = adjusted_annotator_agreement

    elif quality_method == "agreement":
        num_classes = get_num_classes(pred_probs=prior_pred_probs)
        label_counts = np.full((len(labels_multiannotator), num_classes), np.NaN)
        for i in range(len(labels_multiannotator)):
            s = labels_multiannotator.iloc[i]
            label_counts[i, :] = value_counts(s.dropna(), num_classes=num_classes)
        post_pred_probs = label_counts / num_annotations.reshape(-1, 1)

    else:
        raise ValueError(
            f"""
            {quality_method} is not a valid quality method!
            Please choose a valid quality_method: {valid_methods}
            """
        )

    return post_pred_probs, return_model_weight, return_annotator_weight


def _get_post_pred_probs_and_weights_ensemble(
    labels_multiannotator: pd.DataFrame,
    consensus_label: np.ndarray,
    prior_pred_probs: np.ndarray,
    num_annotations: np.ndarray,
    annotator_agreement: np.ndarray,
    quality_method: str = "crowdlab",
    verbose: bool = True,
) -> Tuple[np.ndarray, Any, Any]:
    """Return the posterior predicted class probabilites of each example given a specified quality method and prior predicted class probabilities from an ensemble of multiple classifier models.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    consensus_label : np.ndarray
        An array of shape ``(P, N, K)`` where P is the number of models, consisting of predicted class probabilities from the ensemble models.
        Each set of predicted probabilities with shape ``(N, K)`` is in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
    prior_pred_probs : np.ndarray
        An array of shape ``(N, K)`` of prior predicted probabilities, ``P(label=k|x)``, usually the out-of-sample predicted probability computed by a model.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    num_annotations : np.ndarray
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.
    annotator_agreement : np.ndarray
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.
    quality_method : str, default = "crowdlab" (Options: ["crowdlab", "agreement"])
        Specifies the method used to calculate the quality of the consensus label.
        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`
    verbose : bool, default = True
        Certain warnings and notes will be printed if ``verbose`` is set to ``True``.

    Returns
    -------
    post_pred_probs : np.ndarray
        An array of shape ``(N, K)`` with the posterior predicted probabilities.

    model_weight : np.ndarray
        An array of shape ``(P,)`` where P is the number of models in this ensemble, specifying the model weight used in weighted averages,
        ``None`` if model weight is not used to compute quality scores

    annotator_weight : np.ndarray
        An array of shape ``(M,)`` where M is the number of annotators, specifying the annotator weights used in weighted averages,
        ``None`` if annotator weights are not used to compute quality scores

    """

    num_classes = get_num_classes(pred_probs=prior_pred_probs[0])

    likelihood = np.mean(
        annotator_agreement[num_annotations != 1]
    )  # likelihood that any annotator will annotate the consensus label for any example

    # subsetting the dataset to only includes examples with more than one annotation
    mask = num_annotations != 1
    consensus_label_subset = consensus_label[mask]

    # compute most likely class error
    most_likely_class_error = np.clip(
        np.mean(
            consensus_label_subset
            != np.argmax(np.bincount(consensus_label_subset, minlength=num_classes))
        ),
        a_min=1e-6,
        a_max=None,
    )

    # compute adjusted annotator agreement (used as annotator weights)
    annotator_agreement_with_annotators = _get_annotator_agreement_with_annotators(
        labels_multiannotator, num_annotations, verbose
    )
    annotator_error = 1 - annotator_agreement_with_annotators
    adjusted_annotator_agreement = np.clip(
        1 - (annotator_error / most_likely_class_error), a_min=1e-6, a_max=None
    )

    # compute model weight
    model_weight = np.full(prior_pred_probs.shape[0], np.nan)
    for idx in range(prior_pred_probs.shape[0]):
        prior_pred_probs_subset = prior_pred_probs[idx][mask]

        model_error = np.mean(np.argmax(prior_pred_probs_subset, axis=1) != consensus_label_subset)
        model_weight[idx] = np.max([(1 - (model_error / most_likely_class_error)), 1e-6]) * np.sqrt(
            np.mean(num_annotations)
        )

    # compute weighted average
    post_pred_probs = np.full(prior_pred_probs[0].shape, np.nan)
    for i in range(len(labels_multiannotator)):
        example_mask = labels_multiannotator.iloc[i].notna()
        example = labels_multiannotator.iloc[i][example_mask]
        post_pred_probs[i] = [
            np.average(
                [prior_pred_probs[ind][i, true_label] for ind in range(prior_pred_probs.shape[0])]
                + [
                    likelihood
                    if annotator_label == true_label
                    else (1 - likelihood) / (num_classes - 1)
                    for annotator_label in example
                ],
                weights=np.concatenate((model_weight, adjusted_annotator_agreement[example_mask])),
            )
            for true_label in range(num_classes)
        ]

    return_model_weight = model_weight
    return_annotator_weight = adjusted_annotator_agreement

    return post_pred_probs, return_model_weight, return_annotator_weight


def _get_consensus_quality_score(
    consensus_label: np.ndarray,
    pred_probs: np.ndarray,
    num_annotations: np.ndarray,
    annotator_agreement: np.ndarray,
    quality_method: str = "crowdlab",
    label_quality_score_kwargs: dict = {},
) -> np.ndarray:
    """Return scores representing quality of the consensus label for each example.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    pred_probs : np.ndarray
        An array of shape ``(N, K)`` of posterior predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    num_annotations : np.ndarray
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.
    annotator_agreement : np.ndarray
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.
    quality_method : str, default = "crowdlab" (Options: ["crowdlab", "agreement"])
        Specifies the method used to calculate the quality of the consensus label.
        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`

    Returns
    -------
    consensus_quality_score : np.ndarray
        An array of shape ``(N,)`` with the quality score of the consensus.
    """

    valid_methods = [
        "crowdlab",
        "agreement",
    ]

    if quality_method == "crowdlab":
        consensus_quality_score = get_label_quality_scores(
            consensus_label, pred_probs, **label_quality_score_kwargs
        )

    elif quality_method == "agreement":
        consensus_quality_score = annotator_agreement

    else:
        raise ValueError(
            f"""
            {quality_method} is not a valid consensus quality method!
            Please choose a valid quality_method: {valid_methods}
            """
        )

    return consensus_quality_score


def _get_annotator_label_quality_score(
    annotator_label: pd.Series,
    pred_probs: np.ndarray,
    label_quality_score_kwargs: dict = {},
) -> pd.Series:
    """Returns quality scores for each datapoint.
    Very similar functionality as ``_get_consensus_quality_score`` with additional support for annotator labels that contain NaN values.
    For more info about parameters and returns, see the docstring of :py:func:`_get_consensus_quality_score <cleanlab.multiannotator._get_consensus_quality_score>`.
    """
    mask = pd.notna(annotator_label)

    annotator_label_quality_score_subset = get_label_quality_scores(
        labels=annotator_label[mask].to_numpy().astype("int64"),
        pred_probs=pred_probs[mask],
        **label_quality_score_kwargs,
    )

    annotator_label_quality_score = np.full(len(annotator_label), np.nan)
    annotator_label_quality_score[mask] = annotator_label_quality_score_subset
    return pd.Series(annotator_label_quality_score)


def _get_annotator_quality(
    labels_multiannotator: pd.DataFrame,
    pred_probs: np.ndarray,
    consensus_label: np.ndarray,
    num_annotations: np.ndarray,
    annotator_agreement: np.ndarray,
    model_weight: np.ndarray,
    annotator_weight: np.ndarray,
    quality_method: str = "crowdlab",
) -> pd.DataFrame:
    """Returns annotator quality score for each annotator.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape ``(N, M)``,
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
    model_weight : float
        An array of shape ``(P,)`` where P is the number of models in this ensemble, specifying the model weight used in weighted averages,
        ``None`` if model weight is not used to compute quality scores
    annotator_weight : np.ndarray
        An array of shape ``(M,)`` where M is the number of annotators, specifying the annotator weights used in weighted averages,
        ``None`` if annotator weights are not used to compute quality scores
    quality_method : str, default = "crowdlab" (Options: ["crowdlab", "agreement"])
        Specifies the method used to calculate the quality of the annotators.
        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`

    Returns
    -------
    annotator_quality : np.ndarray
        Quality scores of a given annotator's labels
    """

    valid_methods = [
        "crowdlab",
        "agreement",
    ]

    if quality_method == "crowdlab":
        annotator_lqs = labels_multiannotator.apply(
            lambda s: np.mean(
                get_label_quality_scores(s[pd.notna(s)].astype("int64"), pred_probs[pd.notna(s)])
            )
        )

        mask = num_annotations != 1
        labels_multiannotator_subset = labels_multiannotator[mask]
        consensus_label_subset = consensus_label[mask]

        annotator_agreement = labels_multiannotator_subset.apply(
            lambda s: np.mean(s[pd.notna(s)] == consensus_label_subset[pd.notna(s)]),
            axis=0,
        )

        avg_num_annotations_frac = np.mean(num_annotations) / len(annotator_weight)
        annotator_weight_adjusted = np.sum(annotator_weight) * avg_num_annotations_frac

        w = model_weight / (model_weight + annotator_weight_adjusted)
        annotator_quality = w * annotator_lqs + (1 - w) * annotator_agreement

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


def _get_annotator_worst_class(
    labels_multiannotator: pd.DataFrame,
    consensus_label: np.ndarray,
    consensus_quality_score: np.ndarray,
) -> np.ndarray:
    """Returns the class which each annotator makes the most errors in.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame
        2D pandas DataFrame of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    consensus_quality_score : np.ndarray
        An array of shape ``(N,)`` with the quality score of the consensus.

    Returns
    -------
    worst_class : np.ndarray
        The class that is most frequently mislabeled by a given annotator
    """

    def get_single_annotator_worst_class(s, consensus_quality_score):
        mask = pd.notna(s)
        class_accuracies = (s[mask] == consensus_label[mask]).groupby(s).mean()
        accuracy_min_idx = class_accuracies[class_accuracies == class_accuracies.min()].index.values

        if len(accuracy_min_idx) == 1:
            return accuracy_min_idx[0]

        # tiebreak 1: class counts
        class_count = s[mask].groupby(s).count()[accuracy_min_idx]
        count_max_idx = class_count[class_count == class_count.max()].index.values

        if len(count_max_idx) == 1:
            return count_max_idx[0]

        # tiebreak 2: consensus quality scores
        avg_consensus_quality = (
            pd.DataFrame(
                {"annotator_label": s, "consensus_quality_score": consensus_quality_score}
            )[mask]
            .groupby("annotator_label")
            .mean()["consensus_quality_score"][count_max_idx]
        )
        quality_max_ind = avg_consensus_quality[
            avg_consensus_quality == avg_consensus_quality.max()
        ].index.values

        # return first item even if there are ties - no better methods to tiebreak
        return quality_max_ind[0]

    worst_class = (
        labels_multiannotator.apply(
            lambda s: get_single_annotator_worst_class(s, consensus_quality_score)
        )
        .to_numpy()
        .astype("int64")
    )

    return worst_class

import numpy as np
import pytest
from copy import deepcopy
from cleanlab.benchmarking.noise_generation import generate_noise_matrix_from_trace
from cleanlab.benchmarking.noise_generation import generate_noisy_labels
from cleanlab import count
from cleanlab.multiannotator import (
    get_label_quality_multiannotator,
    convert_long_to_wide_dataset,
    get_majority_vote_label,
)
import pandas as pd


def make_data(
    means=[[3, 2], [7, 7], [0, 8]],
    covs=[[[5, -1.5], [-1.5, 1]], [[1, 0.5], [0.5, 4]], [[5, 1], [1, 5]]],
    sizes=[80, 40, 40],
    avg_trace=0.8,
    num_annotators=50,
    seed=1,  # set to None for non-reproducible randomness
):
    np.random.seed(seed=seed)

    m = len(means)  # number of classes
    n = sum(sizes)
    local_data = []
    labels = []
    test_data = []
    test_labels = []

    for idx in range(m):
        local_data.append(
            np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx])
        )
        test_data.append(
            np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx])
        )
        labels.append(np.array([idx for i in range(sizes[idx])]))
        test_labels.append(np.array([idx for i in range(sizes[idx])]))
    X_train = np.vstack(local_data)
    true_labels_train = np.hstack(labels)
    X_test = np.vstack(test_data)
    true_labels_test = np.hstack(test_labels)

    # Compute p(true_label=k)
    py = np.bincount(true_labels_train) / float(len(true_labels_train))

    noise_matrix = generate_noise_matrix_from_trace(
        m,
        trace=avg_trace * m,
        py=py,
        valid_noise_matrix=True,
        seed=seed,
    )

    # Generate our noisy labels using the noise_matrix for specified number of annotators.
    s = pd.DataFrame(
        np.vstack(
            [generate_noisy_labels(true_labels_train, noise_matrix) for _ in range(num_annotators)]
        ).transpose()
    )

    # column of labels without NaNs to test _get_worst_class
    complete_labels = deepcopy(s)

    # Each annotator only labels approximately 20% of the dataset
    # (unlabeled points represented with NaN)
    s = s.apply(lambda x: x.mask(np.random.random(n) < 0.8))
    s.dropna(axis=1, how="all", inplace=True)

    # Estimate pred_probs
    latent = count.estimate_py_noise_matrices_and_cv_pred_proba(
        X=X_train,
        labels=true_labels_train,
        cv_n_folds=3,
    )

    row_NA_check = pd.notna(s).any(axis=1)

    return {
        "X_train": X_train[row_NA_check],
        "true_labels_train": true_labels_train[row_NA_check],
        "X_test": X_test[row_NA_check],
        "true_labels_test": true_labels_test[row_NA_check],
        "labels": s[row_NA_check].reset_index(drop=True),
        "complete_labels": complete_labels,
        "pred_probs": latent[4][row_NA_check],
        "noise_matrix": noise_matrix,
    }


def make_data_long(data):
    data_long = data.stack().reset_index()
    data_long.columns = ["task", "annotator", "label"]

    return data_long


# Global to be used by all test methods. Only compute this once for speed.
data = make_data()


def test_convert_long_to_wide():
    labels_long = make_data_long(data["labels"])
    labels_wide = convert_long_to_wide_dataset(labels_long)

    assert isinstance(labels_wide, pd.DataFrame)

    # ensures labels_long contains all the non-NaN values of labels_wide
    assert labels_wide.count(axis=1).sum() == len(labels_long)

    # checks one index to make sure data is consistent acrros both dataframes
    example_long = labels_long[labels_long["task"] == 0].sort_values("annotator")
    example_wide = labels_wide.iloc[0].dropna()
    assert all(example_long["annotator"] == example_wide.index)
    assert all(example_long["label"].reset_index(drop=True) == example_wide.reset_index(drop=True))


def test_label_quality_scores_multiannotator():
    labels = data["labels"]
    pred_probs = data["pred_probs"]

    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs)
    assert isinstance(multiannotator_dict, dict)
    assert len(multiannotator_dict) == 3

    label_quality_multiannotator = multiannotator_dict["label_quality"]
    assert isinstance(label_quality_multiannotator, pd.DataFrame)
    assert len(label_quality_multiannotator) == len(labels)
    assert all(label_quality_multiannotator["num_annotations"] > 0)
    assert set(label_quality_multiannotator["consensus_label"]).issubset(np.unique(labels))
    assert all(
        (label_quality_multiannotator["annotator_agreement"] >= 0)
        & (label_quality_multiannotator["annotator_agreement"] <= 1)
    )
    assert all(
        (label_quality_multiannotator["consensus_quality_score"] >= 0)
        & (label_quality_multiannotator["consensus_quality_score"] <= 1)
    )

    annotator_stats = multiannotator_dict["annotator_stats"]
    assert isinstance(annotator_stats, pd.DataFrame)
    assert len(annotator_stats) == labels.shape[1]
    assert all(
        (annotator_stats["annotator_quality"] >= 0) & (annotator_stats["annotator_quality"] <= 1)
    )
    assert all(annotator_stats["num_examples_labeled"] > 0)
    assert all(
        (annotator_stats["agreement_with_consensus"] >= 0)
        & (annotator_stats["agreement_with_consensus"] <= 1)
    )
    assert set(annotator_stats["worst_class"]).issubset(np.unique(labels))

    detailed_label_quality = multiannotator_dict["detailed_label_quality"]
    assert detailed_label_quality.shape == labels.shape

    # test verbose=False
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs, verbose=False)

    # test passing a list into consensus_method
    multiannotator_dict = get_label_quality_multiannotator(
        labels, pred_probs, consensus_method=["majority_vote", "best_quality"]
    )

    # test different quality_methods
    # also testing passing labels as np.ndarray
    multiannotator_dict = get_label_quality_multiannotator(
        np.array(labels), pred_probs, quality_method="agreement"
    )

    # test returning annotator_stats
    multiannotator_dict = get_label_quality_multiannotator(
        labels, pred_probs, return_annotator_stats=False
    )
    assert isinstance(multiannotator_dict, dict)
    assert len(multiannotator_dict) == 2
    assert isinstance(multiannotator_dict["label_quality"], pd.DataFrame)
    assert isinstance(multiannotator_dict["detailed_label_quality"], pd.DataFrame)

    # test returning detailed_label_quality
    multiannotator_dict = get_label_quality_multiannotator(
        labels, pred_probs, return_detailed_quality=False
    )
    assert isinstance(multiannotator_dict, dict)
    assert len(multiannotator_dict) == 2
    assert isinstance(multiannotator_dict["label_quality"], pd.DataFrame)
    assert isinstance(multiannotator_dict["annotator_stats"], pd.DataFrame)

    # test return detailed and annotator stats
    multiannotator_dict = get_label_quality_multiannotator(
        labels, pred_probs, return_detailed_quality=False, return_annotator_stats=False
    )
    assert isinstance(multiannotator_dict, dict)
    assert len(multiannotator_dict) == 1
    assert isinstance(multiannotator_dict["label_quality"], pd.DataFrame)

    # test non-numeric annotator names
    labels_string_names = labels.add_prefix("anno_")
    multiannotator_dict = get_label_quality_multiannotator(
        labels_string_names, pred_probs, return_detailed_quality=False
    )

    # test incorrect consensus_method
    try:
        multiannotator_dict = get_label_quality_multiannotator(
            labels, pred_probs, consensus_method="fake_method"
        )
    except ValueError as e:
        assert "not a valid consensus method" in str(e)

    # test error catching when labels_multiannotator has NaN columns
    labels_NA = deepcopy(labels)
    labels_NA[0] = pd.NA
    try:
        multiannotator_dict = get_label_quality_multiannotator(
            labels_NA, pred_probs, return_annotator_stats=True
        )
    except ValueError as e:
        assert "cannot have columns with all NaN" in str(e)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_rare_class():
    labels = np.array(
        [
            [1, np.NaN, 2],
            [1, 1, 0],
            [2, 2, 0],
            [np.NaN, 2, 2],
            [np.NaN, 2, 1],
            [np.NaN, 2, 2],
        ]
    )

    pred_probs = np.array(
        [
            [0.4, 0.4, 0.2],
            [0.3, 0.6, 0.1],
            [0.05, 0.2, 0.75],
            [0.1, 0.4, 0.5],
            [0.2, 0.4, 0.4],
            [0.2, 0.4, 0.4],
        ]
    )

    consensus_label = get_majority_vote_label(labels)
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs)

    pred_probs_missing = np.array(
        [
            [0.8, 0.2],
            [0.6, 0.14],
            [0.95, 0.05],
            [0.5, 0.5],
            [0.4, 0.6],
            [0.4, 0.6],
        ]
    )

    try:
        multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs_missing)
    except ValueError as e:
        assert "do not match the number of classes in pred_probs" in str(e)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_get_consensus_label():
    labels = data["labels"]

    # getting consensus labels without pred_probs
    consensus_label = get_majority_vote_label(labels)

    # making synthetic data to test tiebreaks of get_consensus_label
    # also testing passing labels as np.ndarray
    labels_tiebreaks = np.array([[1, 2, 0], [1, 1, 0], [1, 0, 0], [2, 2, 2], [1, 2, 0], [1, 2, 0]])
    pred_probs_tiebreaks = np.array(
        [
            [0.4, 0.4, 0.2],
            [0.3, 0.6, 0.1],
            [0.75, 0.2, 0.05],
            [0.1, 0.4, 0.5],
            [0.2, 0.4, 0.4],
            [0.2, 0.4, 0.4],
        ]
    )

    consensus_label = get_majority_vote_label(labels_tiebreaks, pred_probs_tiebreaks)


def test_impute_nonoverlaping_annotators():
    labels = np.array(
        [
            [1, np.NaN, np.NaN],
            [np.NaN, 1, 0],
            [np.NaN, 0, 0],
            [np.NaN, 2, 2],
            [np.NaN, 2, 0],
            [np.NaN, 2, 0],
        ]
    )
    pred_probs = np.array(
        [
            [0.4, 0.4, 0.2],
            [0.3, 0.6, 0.1],
            [0.75, 0.2, 0.05],
            [0.1, 0.4, 0.5],
            [0.2, 0.4, 0.4],
            [0.2, 0.4, 0.4],
        ]
    )

    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs)

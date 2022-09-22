"""Test module for minimal dependencies.

This module is called by GitHub Actions to test that cleanlab can be imported without some of the heavy, optional dependencies.
This excludes pytest which is just used for running the workflow and the tests themselves.

Any new modules that are added to cleanlab must be added to this test module.
Define a new function that imports the module, performs a minimal test, and prints the result.
"""

import numpy as np


LABELS = np.array([1, 0, 0])
LABELS_MULTI = np.array([[1, 0, 0], [1, 0, 1], [1, 0, 0]])
PRED_PROBS = np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])

LABELS_TOKEN = [[0, 0, 1], [0, 1]]
PRED_PROBS_TOKEN = [
    np.array([[0.9, 0.1], [0.7, 0.3], [0.05, 0.95]]),
    np.array([[0.9, 0.1], [0.05, 0.95]]),
]


def test_cleanlab():
    import cleanlab

    print(cleanlab)


def test_classification():
    from cleanlab.classification import CleanLearning

    cl = CleanLearning()
    print(cl)


def test_count():
    from cleanlab.count import num_label_issues

    x = num_label_issues(labels=LABELS, pred_probs=PRED_PROBS)
    print(x)


def test_dataset():
    from cleanlab.dataset import rank_classes_by_label_quality

    x = rank_classes_by_label_quality(labels=LABELS, pred_probs=PRED_PROBS)
    print(x)


def test_filter():
    from cleanlab.filter import find_label_issues

    x = find_label_issues(labels=LABELS, pred_probs=PRED_PROBS)
    print(x)


def test_multiannotator():
    from cleanlab.multiannotator import get_majority_vote_label

    x = get_majority_vote_label(labels_multiannotator=LABELS_MULTI, pred_probs=PRED_PROBS)
    print(x)


def test_outlier():
    from cleanlab.outlier import OutOfDistribution

    ood = OutOfDistribution()
    print(ood)


def test_rank():
    from cleanlab.rank import get_self_confidence_for_each_label

    x = get_self_confidence_for_each_label(labels=LABELS, pred_probs=PRED_PROBS)
    print(x)


def test_typing():
    from cleanlab import typing

    assert typing.DatasetLike is not None
    assert typing.LabelLike is not None


def test_token_classification_filter():
    from cleanlab.token_classification.filter import find_label_issues

    x = find_label_issues(labels=LABELS_TOKEN, pred_probs=PRED_PROBS_TOKEN)
    print(x)


def test_token_classification_rank():
    from cleanlab.token_classification.rank import get_label_quality_scores

    x = get_label_quality_scores(labels=LABELS_TOKEN, pred_probs=PRED_PROBS_TOKEN)
    print(x)


def test_token_classification_summary():
    from cleanlab.token_classification.summary import display_issues

    display_issues(
        issues=[(2, 0), (0, 1)],
        tokens=[
            ["A", "?weird", "sentence"],
            ["A", "valid", "sentence"],
            ["An", "sentence", "with", "a", "typo"],
        ],
    )


def test_benchmarking_noise_generation():
    from cleanlab.benchmarking.noise_generation import generate_noise_matrix_from_trace

    X = generate_noise_matrix_from_trace(5, 2.3, valid_noise_matrix=False)
    print(X)

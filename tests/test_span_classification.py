from cleanlab.experimental.span_classification import (
    find_label_issues,
    display_issues,
    get_label_quality_scores,
)
import numpy as np
import pytest
import warnings

warnings.filterwarnings("ignore")
words = [["I", "love", "Cleanlab", "Inc"], ["A", "new", "park"]]

pred_probs_single_span = [
    np.array([0.3, 0.2, 0.9, 0.1]),
    np.array([0.1, 0.1, 0.9]),
]
labels_single_span = [[0, 0, 1, 1], [0, 0, 1]]
class_names_single_span = ["O", "Span"]

pred_probs_multi_span = [
    np.array([[0.9, 0.2, 0.3], [0.9, 0.9, 0.2], [0.9, 0.1, 0.7], [0.1, 0.1, 0.1]]),
    np.array([[0.1, 0.9, 0.1], [0.1, 0.9, 0.9], [0.1, 0.9, 0.9]]),
]
labels_multi_span = [
    [[0], [1, 2], [1, 3], [0]],
    [[1], [2, 3], [3]],
]
class_names_multi_span = ["O", "Span1", "Span2", "Span3"]


@pytest.mark.parametrize(
    "test_labels",
    [labels_single_span, [np.array(l) for l in labels_single_span]],
    ids=["list labels", "np.array labels"],
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_find_label_issues(test_labels):
    issues = find_label_issues(test_labels, pred_probs_single_span)
    assert isinstance(issues, list)
    assert len(issues) == 1
    assert issues[0] == (0, 3)


def test_find_label_issues_multi_span():
    issues = find_label_issues(labels_multi_span, pred_probs_multi_span)
    assert isinstance(issues, dict)
    assert len(issues) == 3
    assert issues[1] == [(0, 0), (1, 0)]
    assert issues[2] == [(1, 0), (1, 2)]
    assert issues[3] == []


issues = find_label_issues(labels_single_span, pred_probs_single_span)
issues_multi_span = find_label_issues(labels_multi_span, pred_probs_multi_span)


def test_display_issues():
    display_issues(issues, words)
    display_issues(issues, tokens=words, labels=labels_single_span)
    display_issues(issues, words, pred_probs=pred_probs_single_span)
    display_issues(issues, words, pred_probs=pred_probs_single_span, labels=labels_single_span)
    display_issues(
        issues,
        words,
        pred_probs=pred_probs_single_span,
        labels=labels_single_span,
        class_names=class_names_single_span,
        exclude=[(1, 0)],
    )

    display_issues(issues_multi_span, words)
    display_issues(issues_multi_span, tokens=words, labels=labels_multi_span)
    display_issues(issues_multi_span, words, pred_probs=pred_probs_multi_span)
    display_issues(
        issues_multi_span, words, pred_probs=pred_probs_multi_span, labels=labels_multi_span
    )
    display_issues(
        issues_multi_span,
        words,
        pred_probs=pred_probs_multi_span,
        labels=labels_multi_span,
        class_names=class_names_multi_span,
        exclude=[(1, 0)],
    )


@pytest.fixture(name="label_quality_scores")
def fixture_label_quality_scores():
    sentence_scores, token_info = get_label_quality_scores(
        labels_single_span, pred_probs_single_span
    )
    return sentence_scores, token_info


@pytest.fixture(name="label_quality_scores_multi_span")
def fixture_label_quality_scores_multi_span():
    sentence_scores, token_info = get_label_quality_scores(labels_multi_span, pred_probs_multi_span)
    return sentence_scores, token_info


def test_get_label_quality_scores(label_quality_scores):
    sentence_scores, token_info = label_quality_scores
    assert len(sentence_scores) == 2
    assert np.allclose(sentence_scores, [0.1, 0.9])
    assert len(token_info) == 2
    assert np.allclose(token_info[0], [0.7, 0.8, 0.9, 0.1])


def test_get_label_quality_scores_multi_span(label_quality_scores_multi_span):
    sentence_scores, token_info = label_quality_scores_multi_span
    assert len(sentence_scores) == 3
    assert np.allclose(sentence_scores[1], [0.1, 0.1])
    assert len(token_info) == 3
    assert len(token_info[1]) == 2
    assert np.allclose(token_info[1][0], [0.1, 0.9, 0.9, 0.9])

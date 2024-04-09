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

pred_probs = [
    np.array([0.3, 0.2, 0.9, 0.1]),
    np.array([0.1, 0.1, 0.9]),
]

labels = [[0, 0, 1, 1], [0, 0, 1]]
class_names = ["O", "Span"]


@pytest.mark.parametrize(
    "test_labels",
    [labels, [np.array(l) for l in labels]],
    ids=["list labels", "np.array labels"],
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_find_label_issues(test_labels):
    issues = find_label_issues(test_labels, pred_probs)
    assert isinstance(issues, list)
    assert len(issues) == 1
    assert issues[0] == (0, 3)


issues = find_label_issues(labels, pred_probs)


def test_display_issues():
    display_issues(issues, words)
    display_issues(issues, tokens=words, labels=labels)
    display_issues(issues, words, pred_probs=pred_probs)
    display_issues(issues, words, pred_probs=pred_probs, labels=labels)
    display_issues(issues, words, pred_probs=pred_probs, labels=labels, class_names=class_names)


@pytest.fixture(name="label_quality_scores")
def fixture_label_quality_scores():
    sentence_scores, token_info = get_label_quality_scores(labels, pred_probs)
    return sentence_scores, token_info


def test_get_label_quality_scores(label_quality_scores):
    sentence_scores, token_info = label_quality_scores
    assert len(sentence_scores) == 2
    assert np.allclose(sentence_scores, [0.1, 0.9])
    assert len(token_info) == 2
    assert np.allclose(token_info[0], [0.7, 0.8, 0.9, 0.1])

from cleanlab.internal.token_classification_utils import (
    get_sentence,
    filter_sentence,
    process_token,
    mapping,
    merge_probs,
    color_sentence,
)
from cleanlab.token_classification.filter import find_label_issues
from cleanlab.token_classification.rank import (
    get_label_quality_scores,
    issues_from_scores,
    softmin_sentence_score,
)
from cleanlab.token_classification.summary import (
    display_issues,
    common_label_issues,
    filter_by_token,
)
import numpy as np
import pytest

import warnings

warnings.filterwarnings("ignore")
words = [["Hello", "World"], ["#I", "love", "Cleanlab"], ["A"]]
sentences = ["Hello World", "#I love Cleanlab", "A"]

pred_probs = [
    np.array([[0.9, 0.1, 0], [0.6, 0.2, 0.2]]),
    np.array([[0.1, 0, 0.9], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]]),
    np.array([[0.1, 0.1, 0.8]]),
]


labels = [[0, 0], [1, 1, 1], [2]]

maps = [0, 1, 0, 1]
class_names = ["A", "B", "C", "D"]


def test_get_sentence():
    actual_sentences = list(map(get_sentence, words))
    assert actual_sentences == sentences

    # Test with allowed special characters
    words_separated_by_hyphen = ["Heading", "-", "Title"]
    assert get_sentence(words_separated_by_hyphen) == "Heading - Title"

    words_within_parentheses = ["Some", "reason", "(", "Explanation", ")"]
    assert get_sentence(words_within_parentheses) == "Some reason (Explanation)"


def test_filter_sentence():
    filtered_sentences, mask = filter_sentence(sentences)
    assert filtered_sentences == ["Hello World"]
    assert mask == [True, False, False]

    filtered_sentences, mask = filter_sentence(sentences, lambda x: len(x) > 1)
    assert filtered_sentences == ["Hello World", "#I love Cleanlab"]
    assert mask == [True, True, False]

    filtered_sentences, mask = filter_sentence(sentences, lambda x: "#" not in x)
    assert filtered_sentences == ["Hello World", "A"]
    assert mask == [True, False, True]


def test_process_token():
    test_cases = [
        ("Cleanlab", [("C", "a")], "aleanlab"),
        ("Cleanlab", [("C", "a"), ("a", "C")], "aleCnlCb"),
    ]
    for token, replacements, expected in test_cases:
        processed = process_token(token, replacements)
        assert processed == expected


def test_mapping():
    test_cases = [(l, expected) for l, expected in zip(labels, [[0, 0], [1, 1, 1], [0]])]
    for l, expected in test_cases:
        mapped = mapping(l, maps)
        assert mapped == expected


def test_merge_probs():
    merged_probs = merge_probs(pred_probs[0], maps)
    expected = np.array([[0.9, 0.1], [0.8, 0.2]])
    assert np.allclose(expected, merged_probs)

    merged_probs = merge_probs(pred_probs[1], maps)
    expected = np.array([[1.0, 0.0], [0.2, 0.8], [0.2, 0.8]])
    assert np.allclose(expected, merged_probs)

    merged_probs = merge_probs(pred_probs[2], maps)
    expected = np.array([[0.9, 0.1]])
    assert np.allclose(expected, merged_probs)


def test_merge_probs_with_normalization():
    # Ignore probabilities for class/entity 0
    norm_maps = [-1, 1, 0, 1]
    merged_probs = merge_probs(pred_probs[0], norm_maps)
    expected = np.array([[0.0, 1.0], [0.5, 0.5]])
    assert np.allclose(expected, merged_probs)

    merged_probs = merge_probs(pred_probs[1], norm_maps)
    expected = np.array([[1.0, 0.0], [1 / 9, 8 / 9], [1 / 9, 8 / 9]])
    assert np.allclose(expected, merged_probs)

    merged_probs = merge_probs(pred_probs[2], norm_maps)
    expected = np.array([[8 / 9, 1 / 9]])

    # Ignore probabilities for class/entity 1
    norm_maps = [0, -1, 0, 1]
    merged_probs = merge_probs(pred_probs[0], norm_maps)
    expected = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert np.allclose(expected, merged_probs)

    merged_probs = merge_probs(pred_probs[1], norm_maps)
    expected = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert np.allclose(expected, merged_probs)

    merged_probs = merge_probs(pred_probs[2], norm_maps)
    expected = np.array([[1.0, 0.0]])
    assert np.allclose(expected, merged_probs)


def test_color_sentence():
    colored = color_sentence(sentences[0], words[0][1])
    assert colored == "Hello \x1b[31mWorld\x1b[0m"


issues = find_label_issues(labels, pred_probs)


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
    assert issues[0] == (1, 0)


def test_softmin_sentence_score():
    token_scores = [[0.9, 0.6], [0.0, 0.8, 0.8], [0.8]]
    sentence_scores = softmin_sentence_score(token_scores)
    assert isinstance(sentence_scores, np.ndarray)
    assert np.allclose(sentence_scores, [0.60074, 1.8e-07, 0.8])

    # Temperature limits
    sentence_scores = softmin_sentence_score(token_scores, temperature=0)
    assert np.allclose(sentence_scores, [0.6, 0.0, 0.8])

    sentence_scores = softmin_sentence_score(token_scores, temperature=np.inf)
    assert np.allclose(sentence_scores, [0.75, 1.6 / 3, 0.8])


@pytest.fixture(name="label_quality_scores")
def fixture_label_quality_scores():
    sentence_scores, token_info = get_label_quality_scores(labels, pred_probs)
    return sentence_scores, token_info


def test_get_label_quality_scores(label_quality_scores):
    sentence_scores, token_info = label_quality_scores
    assert len(sentence_scores) == 3
    assert np.allclose(sentence_scores, [0.6, 0, 0.8])
    assert len(token_info) == 3
    assert np.allclose(token_info[0], [0.9, 0.6])
    sentence_scores_softmin, _ = get_label_quality_scores(
        labels, pred_probs, sentence_score_method="softmin", tokens=words
    )
    assert len(sentence_scores_softmin) == 3
    assert np.allclose(sentence_scores_softmin, [0.600741787, 1.8005624e-7, 0.8])

    with pytest.raises(AssertionError) as excinfo:
        get_label_quality_scores(
            labels, pred_probs, sentence_score_method="unsupported_method", tokens=words
        )
    assert "Select from the following methods:" in str(excinfo.value)


def test_issues_from_scores(label_quality_scores):
    sentence_scores, token_info = label_quality_scores
    issues = issues_from_scores(sentence_scores, token_info)
    assert len(issues) == 1
    assert issues[0] == (1, 0)
    issues_without = issues_from_scores(sentence_scores)
    assert len(issues_without) == 1
    assert issues_without[0] == 1


def test_display_issues():
    display_issues(issues, words)
    display_issues(issues, words, given_labels=labels)
    display_issues(issues, words, pred_probs=pred_probs)
    display_issues(issues, words, pred_probs=pred_probs, given_labels=labels)
    display_issues(
        issues, words, pred_probs=pred_probs, given_labels=labels, class_names=class_names
    )

    issues_sentence_only = [i for i, _ in issues]
    display_issues(issues_sentence_only, words)


def test_common_label_issues():
    common_label_issues(issues, words)
    common_label_issues(
        issues, words, labels=labels, pred_probs=pred_probs, class_names=class_names
    )


def test_filter_by_token():
    filter_by_token("Hello", issues, words)

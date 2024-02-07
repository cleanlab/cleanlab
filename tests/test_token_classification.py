from cleanlab.internal.token_classification_utils import (
    get_sentence,
    filter_sentence,
    process_token,
    mapping,
    merge_probs,
    color_sentence,
    _replace_sentence,
)
from cleanlab.token_classification.filter import find_label_issues
from cleanlab.token_classification.rank import (
    get_label_quality_scores,
    issues_from_scores,
    _softmin_sentence_score,
)
from cleanlab.token_classification.summary import (
    display_issues,
    common_label_issues,
    filter_by_token,
)
import numpy as np
import pandas as pd
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


# Color boundaries
C_L, C_R = "\x1b[31m", "\x1b[0m"


@pytest.mark.parametrize(
    "sentence,word,expected",
    [
        ("Hello World", "World", f"Hello {C_L}World{C_R}"),
        ("Hello World", "help", "Hello World"),
        ("If you and I were to meet", "I", f"If you and {C_L}I{C_R} were to meet"),
        ("If you and I were to meet", "If you and I", f"{C_L}If you and I{C_R} were to meet"),
        ("If you and I were to meet", "If you and I w", f"{C_L}If you and I w{C_R}ere to meet"),
        ("I think I know this", "I", f"{C_L}I{C_R} think {C_L}I{C_R} know this"),
        ("A good reason for a test", "a", f"A good reason for {C_L}a{C_R} test"),
        ("ab ab a b ab", "ab a", f"ab {C_L}ab a{C_R} b ab"),
        ("ab ab ab ab", "ab a", f"{C_L}ab a{C_R}b {C_L}ab a{C_R}b"),
        (
            "Alan John Percivale (A.j.p.) Taylor died",
            "(",
            f"Alan John Percivale {C_L}({C_R}A.j.p.) Taylor died",
        ),
    ],
    ids=[
        "single_word",
        "no_match",
        "ignore_subwords",
        "multi-token_match",
        "substring_replacement",
        "multiple_matches",
        "case_sensitive",
        "only_word_boundary",
        "non_overlapping_substrings",
        "issue_403-escape_special_regex_characters",
    ],
)
def test_color_sentence(monkeypatch: pytest.MonkeyPatch, sentence, word, expected):
    import os

    monkeypatch.setattr(os, "isatty", lambda fd: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.fileno", lambda: 1)

    colored = color_sentence(sentence, word)
    assert colored == expected


@pytest.mark.parametrize(
    "sentence,word,expected",
    [
        ("Hello World", "World", "Hello [EXPECTED]"),
        ("Hello World", "help", "Hello World"),
        ("If you and I were to meet", "I", "If you and [EXPECTED] were to meet"),
        ("If you and I were to meet", "If you and I", "[EXPECTED] were to meet"),
        ("If you and I were to meet", "If you and I w", "[EXPECTED]ere to meet"),
        ("I think I know this", "I", "[EXPECTED] think [EXPECTED] know this"),
        ("A good reason for a test", "a", "A good reason for [EXPECTED] test"),
        ("ab ab a b ab", "ab a", "ab [EXPECTED] b ab"),
        ("ab ab ab ab", "ab a", "[EXPECTED]b [EXPECTED]b"),
        (
            "Alan John Percivale (A.j.p.) Taylor died",
            "(",
            "Alan John Percivale [EXPECTED]A.j.p.) Taylor died",
        ),
    ],
    ids=[
        "single_word",
        "no_match",
        "ignore_subwords",
        "multi-token_match",
        "substring_replacement",
        "multiple_matches",
        "case_sensitive",
        "only_word_boundary",
        "non_overlapping_substrings",
        "issue_403-escape_special_regex_characters",
    ],
)
def test_replace_sentence(sentence, word, expected):
    new_sentence = _replace_sentence(sentence, word, "[EXPECTED]")
    assert new_sentence == expected


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
    issues2 = find_label_issues(
        test_labels, pred_probs, return_indices_ranked_by="normalized_margin", n_jobs=1
    )
    assert isinstance(issues2, list)
    # Compare results with low_memory=True. Pass unused argument n_jobs=1
    issues_lm = find_label_issues(test_labels, pred_probs, low_memory=True, n_jobs=1)
    intersection = len(list(set(issues).intersection(set(issues_lm))))
    union = len(set(issues)) + len(set(issues_lm)) - intersection
    assert float(intersection) / union > 0.95


def test_softmin_sentence_score():
    token_scores = [[0.9, 0.6], [0.0, 0.8, 0.8], [0.8]]
    sentence_scores = _softmin_sentence_score(token_scores)
    assert isinstance(sentence_scores, np.ndarray)
    assert np.allclose(sentence_scores, [0.60074, 1.8e-07, 0.8])

    # Temperature limits
    sentence_scores = _softmin_sentence_score(token_scores, temperature=0)
    assert np.allclose(sentence_scores, [0.6, 0.0, 0.8])

    sentence_scores = _softmin_sentence_score(token_scores, temperature=np.inf)
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
    sentence_scores, token_scores = label_quality_scores
    issues = issues_from_scores(sentence_scores, token_scores=token_scores)
    assert len(issues) == 1
    assert issues[0] == (1, 0)
    issues_without = issues_from_scores(sentence_scores)
    assert len(issues_without) == 1
    assert issues_without[0] == 1


def test_display_issues():
    display_issues(issues, words)
    display_issues(issues, tokens=words, labels=labels)
    display_issues(issues, words, pred_probs=pred_probs)
    display_issues(issues, words, pred_probs=pred_probs, labels=labels)
    display_issues(issues, words, pred_probs=pred_probs, labels=labels, class_names=class_names)

    exclude = [(1, 2)]  # Occurs in first token of second sentence "#I"
    display_issues(issues, words, pred_probs=pred_probs, labels=labels, exclude=exclude)

    top = 1
    display_issues(issues, words, pred_probs=pred_probs, labels=labels, top=top)

    issues_sentence_only = [i for i, _ in issues]
    display_issues(issues_sentence_only, words)


TEST_KWARGS = {"labels": labels, "pred_probs": pred_probs, "class_names": class_names}


@pytest.mark.parametrize(
    "test_issues",
    [issues, issues + [(1, 0)]],
    ids=["default issues", "augmented issues"],
)
@pytest.mark.parametrize(
    "test_kwargs",
    [
        {},
        TEST_KWARGS,
        {**TEST_KWARGS, "top": 1},
        {**TEST_KWARGS, "exclude": [(1, 2)]},
        {**TEST_KWARGS, "verbose": False},
    ],
    ids=["no kwargs", "labels+pred_probs+class_names", "...+top", "...+exclude", "...+no verbose"],
)
def test_common_label_issues(test_issues, test_kwargs):
    df = common_label_issues(test_issues, words, **test_kwargs)
    assert isinstance(df, pd.DataFrame)

    columns = df.columns.tolist()
    for col in ["token", "num_label_issues"]:
        assert col in columns
    if test_kwargs:
        for col in ["given_label", "predicted_label"]:
            assert col in columns


@pytest.mark.parametrize(
    "test_token,expected_issues",
    [
        ("Hello", []),
        ("#I", [(1, 0)]),
    ],
)
def test_filter_by_token(test_token, expected_issues):
    returned_issues = filter_by_token(test_token, issues, words)
    assert returned_issues == expected_issues

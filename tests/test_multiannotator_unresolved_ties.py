import warnings

import numpy as np
import pandas as pd
import pytest

from cleanlab.multiannotator import get_label_quality_multiannotator, get_majority_vote_label


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.mark.parametrize("as_dataframe", [False, True])
@pytest.mark.parametrize("with_probabilities", [False, True])
@pytest.mark.parametrize("seed", [0, 17])
def test_all_tied_votes_reach_the_documented_random_fallback(
    as_dataframe, with_probabilities, seed
):
    labels = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    if as_dataframe:
        labels = pd.DataFrame(labels, columns=["alice", "bob"], index=[8, 4, 6, 2])
    probabilities = np.full((4, 2), 0.5) if with_probabilities else None
    reference_rng = np.random.RandomState(seed)
    expected = [reference_rng.choice([0, 1]) for _ in range(4)]
    np.random.seed(seed)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = get_majority_vote_label(labels, probabilities)

    np.testing.assert_array_equal(actual, expected)
    assert any("breaking ties" in str(w.message) for w in caught)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


@pytest.mark.parametrize("seed", [3, 14])
def test_missing_annotations_keep_each_rows_unresolved_candidates(seed):
    labels = np.array([[0, 1, np.nan], [np.nan, 1, 2], [0, np.nan, 2]])
    candidates = [[0, 1], [1, 2], [0, 2]]
    reference_rng = np.random.RandomState(seed)
    expected = [reference_rng.choice(options) for options in candidates]
    np.random.seed(seed)

    with pytest.warns(UserWarning, match="breaking ties"):
        actual = get_majority_vote_label(labels, verbose=False)

    np.testing.assert_array_equal(actual, expected)


def test_model_resolved_ties_still_take_precedence_and_do_not_use_randomness():
    labels = np.array([[0, 1], [1, 0]])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9]])
    np.random.seed(12)
    expected_next = np.random.RandomState(12).random_sample()

    actual = get_majority_vote_label(labels, probabilities, verbose=False)

    np.testing.assert_array_equal(actual, [0, 1])
    assert np.random.random_sample() == expected_next


def test_multiannotator_quality_can_score_a_dataset_without_any_untied_majority():
    labels = pd.DataFrame([[0, 1], [1, 0], [0, 1], [1, 0]], index=[7, 2, 9, 3])
    np.random.seed(9)

    with pytest.warns(UserWarning, match="breaking ties"):
        result = get_label_quality_multiannotator(
            labels, np.full((4, 2), 0.5), quality_method="agreement", verbose=False
        )

    quality = result["label_quality"]
    np.testing.assert_array_equal(quality.index, labels.index)
    np.testing.assert_allclose(quality["consensus_quality_score"], np.full(4, 0.5))
    np.testing.assert_allclose(quality["annotator_agreement"], np.full(4, 0.5))
    assert np.isfinite(result["detailed_label_quality"]).all().all()

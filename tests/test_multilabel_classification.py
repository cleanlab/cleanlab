# Copyright (C) 2017-2022  Cleanlab Inc.
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


import itertools
import typing
import numpy as np
import pytest
import sklearn
from sklearn.utils.multiclass import is_multilabel
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from cleanlab.internal import multilabel_utils as mlutils


@pytest.fixture
def labels():
    return np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
    )


@pytest.fixture
def pred_probs():
    return np.array(
        [
            [0.9, 0.1, 0.2],
            [0.5, 0.6, 0.4],
            [0.75, 0.80, 0.85],
            [0.9, 0.85, 0.2],
            [0.9, 0.1, 0.85],
            [0.5, 0.6, 0.85],
            [0.9, 0.85, 0.85],
            [0.8, 0.4, 0.2],
            [0.9, 0.1, 0.85],
            [0.15, 0.95, 0.05],
        ]
    )


@pytest.fixture
def cv():
    return sklearn.model_selection.StratifiedKFold(
        n_splits=2,
        shuffle=True,
        random_state=42,
    )


@pytest.fixture
def dummy_features(labels):
    np.random.seed(42)
    return np.random.rand(labels.shape[0], 2)


@pytest.mark.parametrize("base_scorer", [scorer for scorer in mlutils.ClassLabelScorer])
@pytest.mark.parametrize("aggregator", [np.min, np.max, np.mean, None])
@pytest.mark.parametrize("strict", [True, False])
def test_multilabel_scorer(base_scorer, aggregator, strict, labels, pred_probs):
    scorer = mlutils.MultilabelScorer(base_scorer, aggregator, strict=strict)
    assert callable(scorer)

    test_scores = scorer(labels, pred_probs)
    assert isinstance(test_scores, np.ndarray)
    assert test_scores.shape == (labels.shape[0],)


@pytest.fixture
def scorer():
    return mlutils.MultilabelScorer(
        base_scorer=mlutils.ClassLabelScorer.SELF_CONFIDENCE,
        aggregator=np.min,
    )


def test_multilabel_scorer_extend_binary_pred_probs():
    method = mlutils.MultilabelScorer._stack_complement

    # Toy example
    pred_probs_class = np.array([0.1, 0.9, 0.3, 0.8])
    pred_probs_extended = method(pred_probs_class)
    pred_probs_expected = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.7, 0.3],
            [0.2, 0.8],
        ]
    )
    assert np.isclose(pred_probs_extended, pred_probs_expected).all()

    # Check preservation of probabilities
    pred_probs_class = np.random.rand(100)
    pred_probs_extended = method(pred_probs_class)
    assert np.sum(pred_probs_extended, axis=1).all() == 1


def test_fixture_for_multi_label_data(multilabeled_data):
    """Internal test to ensure that the multilabeled_data test fixture is correctly set up."""
    assert isinstance(multilabeled_data, dict)
    assert isinstance(multilabeled_data["labels"], np.ndarray)
    assert is_multilabel(multilabeled_data["labels"])
    assert isinstance(multilabeled_data["pred_probs"], np.ndarray)
    assert multilabeled_data["labels"].shape == multilabeled_data["pred_probs"].shape


def test_get_label_quality_scores_output(labels, pred_probs, scorer):
    # Check that the function returns a dictionary with the correct keys.
    scores = mlutils.get_label_quality_scores(labels, pred_probs, method=scorer)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (labels.shape[0],)
    assert np.all(scores >= 0) and np.all(scores <= 1)
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize(
    "given_labels,expected",
    [
        (
            pytest.lazy_fixture("labels"),
            np.array([1 / 10, 1 / 10, 2 / 10, 1 / 10, 1 / 10, 2 / 10, 1 / 10, 1 / 10]),
        ),
        (np.array([[0, 1], [0, 0], [1, 1]]), np.array([1 / 3, 1 / 3, 0, 1 / 3])),
        (np.array([[0, 1], [0, 0], [0, 1], [0, 1]]), np.array([1 / 4, 3 / 4, 0, 0])),
        (
            np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0]]),
            np.array([0 if i != 2**7 else 1 for i in range(2**9)]),
        ),
    ],
    ids=[
        "default",
        "Missing class assignment configuration",
        "Missing class",
        "Handle more than 8 classes",
    ],
)
def test_multilabel_py(given_labels, expected):
    py = mlutils.multilabel_py(given_labels)
    assert isinstance(py, np.ndarray)
    assert py.shape == (2 ** given_labels.shape[1],)
    assert np.isclose(py, expected).all()


@pytest.mark.parametrize("K", [2, 3, 4], ids=["K=2", "K=3", "K=4"])
def test_get_split_generator(cv, K):

    all_configurations = np.array(list(itertools.product([0, 1], repeat=K)))
    given_labels = np.repeat(all_configurations, 2, axis=0)

    split_generator = mlutils._get_split_generator(given_labels, cv)
    assert isinstance(split_generator, typing.Generator)

    train, test = next(split_generator)
    for split in (train, test):
        assert isinstance(split, np.ndarray)
        assert np.isin(split, np.arange(given_labels.shape[0])).all()

    # Test that the label distribution is relatively equal among the splits.
    train_labels, test_labels = given_labels[train], given_labels[test]
    _, train_counts = np.unique(train_labels, axis=0, return_counts=True)
    _, test_counts = np.unique(test_labels, axis=0, return_counts=True)
    # cv.get_n_splits() is 2, so we expect 1/2 of the labels in each split.
    assert np.all(train_counts == 1)
    assert np.all(test_counts == 1)


# Test split_generator with rare/missing multilabel configurations
@pytest.mark.parametrize("K", [2, 3, 4], ids=["K=2", "K=3", "K=4"])
def test_get_split_generator_rare_configurations(cv, K):

    all_configurations = np.array(list(itertools.product([0, 1], repeat=K)))
    given_labels = np.repeat(all_configurations, 2, axis=0)

    # Remove one configuration
    given_labels = given_labels[~np.all(given_labels == all_configurations[0], axis=1)]

    split_generator = mlutils._get_split_generator(given_labels, cv)
    train, test = next(split_generator)
    train_labels, test_labels = given_labels[train], given_labels[test]

    # Test that the label distribution is relatively equal among the splits.
    _, train_counts = np.unique(train_labels, axis=0, return_counts=True)
    _, test_counts = np.unique(test_labels, axis=0, return_counts=True)
    # cv.get_n_splits() is 2, so we expect 1/2 of the labels in each split.
    assert np.all(train_counts == 1)
    assert np.all(test_counts == 1)
    assert len(train_counts) == len(test_counts) == len(all_configurations) - 1

    # Remove one instance from labels
    given_labels = given_labels[1:, :]

    split_generator = mlutils._get_split_generator(given_labels, cv)
    train, test = next(split_generator)
    train_labels, test_labels = given_labels[train], given_labels[test]

    # Test that the label distribution is relatively equal among the splits.
    _, train_counts = np.unique(train_labels, axis=0, return_counts=True)
    _, test_counts = np.unique(test_labels, axis=0, return_counts=True)
    # cv.get_n_splits() is 2, so we expect 1/2 of the labels in each split,
    # except for the class with one fewer instances.
    assert len(train_counts) != len(test_counts)


def test_get_cross_validated_multilabel_pred_probs(dummy_features, labels, cv):
    clf = OneVsRestClassifier(LogisticRegression(random_state=0))
    pred_probs = mlutils.get_cross_validated_multilabel_pred_probs(
        dummy_features,
        labels,
        clf=clf,
        cv=cv,
    )
    assert isinstance(pred_probs, np.ndarray)
    assert pred_probs.shape == labels.shape
    assert np.all(pred_probs >= 0) and np.all(pred_probs <= 1)
    assert np.all(np.isfinite(pred_probs))

    # Gold master test - Ensure output is consistent
    assert dummy_features.shape == (10, 2)
    gold_master = np.array(
        [
            [0.203, 0.465, 0.612],
            [0.802, 0.596, 0.43],
            [0.776, 0.649, 0.391],
            [0.201, 0.439, 0.633],
            [0.203, 0.443, 0.584],
            [0.814, 0.572, 0.332],
            [0.201, 0.388, 0.544],
            [0.778, 0.646, 0.392],
            [0.796, 0.611, 0.387],
            [0.199, 0.381, 0.58],
        ]
    )
    assert np.allclose(pred_probs, gold_master, atol=5e-4)

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


import numpy as np
import pytest
from sklearn.utils.multiclass import is_multilabel

from cleanlab.internal import multilabel_rank_utils as mlrank


@pytest.fixture
def labels(multilabeled_data):
    return multilabeled_data["labels"]


@pytest.fixture
def pred_probs(multilabeled_data):
    return multilabeled_data["pred_probs"]


@pytest.mark.parametrize("base_scorer", [scorer for scorer in mlrank.BaseQualityScorer])
@pytest.mark.parametrize("aggregator", [np.min, np.max, np.mean, None])
@pytest.mark.parametrize("strict", [True, False])
def test_multilabel_scorer(base_scorer, aggregator, strict, labels, pred_probs):
    scorer = mlrank.MultilabelScorer(base_scorer, aggregator, strict=strict)
    assert callable(scorer)

    test_scores = scorer(labels, pred_probs)
    assert isinstance(test_scores, np.ndarray)
    assert test_scores.shape == (labels.shape[0],)


@pytest.fixture
def scorer():
    return mlrank.MultilabelScorer(
        base_scorer=mlrank.BaseQualityScorer.SELF_CONFIDENCE,
        aggregator=np.min,
    )


def test_multilabel_scorer_extend_binary_pred_probs():
    method = mlrank.MultilabelScorer._stack_complement

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
    scores = mlrank.get_label_quality_scores(labels, pred_probs, method=scorer)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (labels.shape[0],)
    assert np.all(scores >= 0) and np.all(scores <= 1)
    assert np.all(np.isfinite(scores))

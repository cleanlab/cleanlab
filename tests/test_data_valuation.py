# Copyright (C) 2017-2024  Cleanlab Inc.
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
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite
from hypothesis.extra.numpy import arrays

from sklearn.neighbors import NearestNeighbors

from cleanlab.data_valuation import _knn_shapley_score, data_shapley_knn
from cleanlab.internal.neighbor.knn_graph import create_knn_graph_and_index


class TestDataValuation:
    K = 3
    N = 100
    num_features = 10

    @pytest.fixture
    def features(self):
        return np.random.rand(self.N, self.num_features)

    @pytest.fixture
    def labels(self):
        return np.random.randint(0, 2, self.N)

    @pytest.fixture
    def knn_graph(self, features):
        knn = NearestNeighbors(n_neighbors=self.K).fit(features)
        knn_graph = knn.kneighbors_graph(mode="distance")
        return knn_graph

    def test_data_shapley_knn(self, labels, features):
        shapley = data_shapley_knn(labels, features=features, k=self.K)
        assert shapley.shape == (100,)
        assert np.all(shapley >= 0)
        assert np.all(shapley <= 1)

    def test_data_shapley_knn_with_knn_graph(self, labels, knn_graph):
        shapley = data_shapley_knn(labels, knn_graph=knn_graph, k=self.K)
        assert shapley.shape == (100,)
        assert np.all(shapley >= 0)
        assert np.all(shapley <= 1)


@composite
def valid_data(draw):
    """
    A custom strategy to generate valid labels, features, and k such that:
    - labels and features have the same length
    - k is less than the length of labels and features
    """
    # Generate a valid length for labels and features
    length = draw(st.integers(min_value=11, max_value=1000))

    # Generate labels and features of the same length
    labels = draw(
        arrays(
            dtype=np.int32,
            shape=length,
            elements=st.integers(min_value=0, max_value=length - 1),
        )
    )
    features = draw(
        arrays(
            dtype=np.float64,
            shape=(length, draw(st.integers(min_value=2, max_value=50))),
            elements=st.floats(min_value=-1.0, max_value=1.0),
        )
    )

    # Generate k such that it is less than the length of labels and features
    k = draw(st.integers(min_value=1, max_value=length - 1))

    return labels, features, k


class TestDataShapleyKNNScore:
    """This test class prioritizes testing the raw/untransformed outputs of the _knn_shapley_score function."""

    @settings(
        max_examples=1000, deadline=None
    )  # Increase the number of examples to test more cases
    @given(valid_data())
    def test_knn_shapley_score_property(self, data):
        labels, features, k = data

        knn_graph, _ = create_knn_graph_and_index(features, n_neighbors=k)
        neighbor_indices = knn_graph.indices.reshape(-1, k)

        scores = _knn_shapley_score(neighbor_indices, labels, k)

        # Shapley scores should be between -1 and 1
        assert scores.shape == (len(labels),)
        assert np.all(scores >= -1)
        assert np.all(scores <= 1)

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

from sklearn.neighbors import NearestNeighbors

from cleanlab.data_valuation import data_shapley_knn


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

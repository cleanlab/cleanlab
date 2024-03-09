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

from scipy.sparse import csr_matrix

from cleanlab.datavaluation import data_shapley_knn


class TestDataValuation:
    K = 3
    N = 100

    @pytest.fixture
    def labels(self):
        return np.random.randint(0, 2, self.N)

    @pytest.fixture
    def knn_graph(self):
        knn_graph = csr_matrix(np.random.rand(self.N, self.N))
        return knn_graph

    def test_data_shapley_knn(self, knn_graph, labels):
        shapley = data_shapley_knn(knn_graph, labels, k=self.K)
        assert shapley.shape == (100,)
        assert np.all(shapley >= 0)
        assert np.all(shapley <= 1)

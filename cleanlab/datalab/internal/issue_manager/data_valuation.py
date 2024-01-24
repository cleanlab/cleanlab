# Copyright (C) 2017-2023  Cleanlab Inc.
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
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from cleanlab.datalab.internal.issue_manager import IssueManager

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from cleanlab.datalab.datalab import Datalab


class DataValuationIssueManager(IssueManager):
    """
    Detect which examples in a dataset are least valuable via an approximate Data Shapely value.

    Examples
    --------
    .. code-block:: python

        >>> from cleanlab import Datalab
        >>> import numpy as np
        >>> from sklearn.neighbors import NearestNeighbors
        >>>
        >>> # Generate two distinct clusters
        >>> X = np.vstack([
        ...     np.random.normal(-1, 1, (25, 2)),
        ...     np.random.normal(1, 1, (25, 2)),
        ... ])
        >>> y = np.array([0]*25 + [1]*25)
        >>>
        >>> # Initialize Datalab with data
        >>> lab = Datalab(data={"y": y}, label_name="y")
        >>>
        >>> # Creating a knn_graph for data valuation
        >>> knn = NearestNeighbors(n_neighbors=10).fit(X)
        >>> knn_graph = knn.kneighbors_graph(mode='distance')
        >>>
        >>> # Specifying issue types for data valuation
        >>> issue_types = {"data_valuation": {}}
        >>> lab.find_issues(knn_graph=knn_graph, issue_types=issue_types)
    """

    description: ClassVar[
        str
    ] = """
    Examples that contribute minimally to a model's training
    receive lower valuation scores.
    Since the original knn-shapley value is in [-1, 1], we transform it to [0, 1] by:

    .. math::
        0.5 \times (\text{shapley} + 1)

    here shapley is the original knn-shapley value.
    """

    issue_name: ClassVar[str] = "data_valuation"
    issue_score_key: ClassVar[str]
    verbosity_levels: ClassVar[Dict[int, List[str]]] = {
        0: [],
        1: [],
        2: [],
        3: ["average_data_valuation"],
    }

    DEFAULT_THRESHOLD = 0.5

    def __init__(
        self,
        datalab: Datalab,
        threshold: Optional[float] = None,
        k: int = 10,
        **kwargs,
    ):
        super().__init__(datalab)
        self.k = k
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD

    def find_issues(
        self,
        **kwargs,
    ) -> None:
        """Calculate the data valuation score with a provided or existing knn graph.
        Based on KNN-Shapley value described in https://arxiv.org/abs/1911.07128
        The larger the score, the more valuable the data point is, the more contribution it will make to the model's training.

        Parameters
        ----------
        knn_graph : csr_matrix
            A sparse matrix representing the knn graph.
        """
        self.k = kwargs.get("k", self.k)
        knn_graph = self._process_knn_graph_from_inputs(kwargs)
        labels = self.datalab.labels
        if not isinstance(labels, np.ndarray):
            error_msg = (
                f"Expected labels to be a numpy array of shape (n_samples,) to use with DataValuationIssueManager, "
                f"but got {type(labels)} instead."
            )
            raise TypeError(error_msg)
        if knn_graph is None:
            raise ValueError(
                "knn_graph must be provided in kwargs or already stored in the Datalab instance\n"
                "It should be calculated by other issue managers if it is not provided via "
                "`Datalab.find_issues(knn_graph=knn_graph, ...)`"
            )
        if labels is None:
            raise ValueError("labels must be provided to run data valuation")

        scores = _knn_shapley_score(knn_graph, labels, self.k)

        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": scores < self.threshold,
                self.issue_score_key: scores,
            },
        )
        self.summary = self.make_summary(score=scores.mean())

        self.info = self.collect_info(self.issues)

    def _process_knn_graph_from_inputs(self, kwargs: Dict[str, Any]) -> Union[csr_matrix, None]:
        """Determine if a knn_graph is provided in the kwargs or if one is already stored in the associated Datalab instance."""
        knn_graph_kwargs: Optional[csr_matrix] = kwargs.get("knn_graph", None)
        knn_graph_stats = self.datalab.get_info("statistics").get("weighted_knn_graph", None)

        knn_graph: Optional[csr_matrix] = None
        if knn_graph_kwargs is not None:
            knn_graph = knn_graph_kwargs
        elif knn_graph_stats is not None:
            knn_graph = knn_graph_stats

        if isinstance(knn_graph, csr_matrix) and self.k > (knn_graph.nnz // knn_graph.shape[0]):
            self.k = knn_graph.nnz // knn_graph.shape[0]
            Warning(
                f"k is larger than the number of neighbors in the knn graph. Using k={self.k} instead."
            )
        return knn_graph

    def collect_info(self, issues: pd.DataFrame) -> dict:
        issues_info = {
            "num_low_valuation_issues": sum(issues[f"is_{self.issue_name}_issue"]),
            "average_data_valuation": issues[self.issue_score_key].mean(),
        }

        info_dict = {
            **issues_info,
        }

        return info_dict


def _knn_shapley_score(knn_graph: csr_matrix, labels: np.ndarray, k: int) -> np.ndarray:
    """Compute the Shapley values of data points based on a knn graph."""
    N = labels.shape[0]
    scores = np.zeros((N, N))
    dist = knn_graph.indices.reshape(N, -1)

    for y, s, dist_i in zip(labels, scores, dist):
        idx = dist_i[::-1]
        ans = labels[idx]
        s[idx[k - 1]] = float(ans[k - 1] == y)
        ans_matches = (ans == y).flatten()
        for j in range(k - 2, -1, -1):
            s[idx[j]] = s[idx[j + 1]] + float(int(ans_matches[j]) - int(ans_matches[j + 1]))
    return 0.5 * (np.mean(scores / k, axis=0) + 1)

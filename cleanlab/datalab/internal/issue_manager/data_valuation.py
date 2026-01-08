from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Union,
)


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from cleanlab.data_valuation import data_shapley_knn
from cleanlab.datalab.internal.issue_manager import IssueManager
from cleanlab.datalab.internal.issue_manager.knn_graph_helpers import (
    num_neighbors_in_knn_graph,
    set_knn_graph,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
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
        metric: Optional[Union[str, Callable]] = None,
        threshold: Optional[float] = None,
        k: int = 10,
        **kwargs,
    ):
        super().__init__(datalab)
        self.metric = metric
        self.k = k
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD

    def find_issues(
        self,
        features: Optional[npt.NDArray] = None,
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
        labels = self.datalab.labels
        if not isinstance(labels, np.ndarray):
            error_msg = (
                f"Expected labels to be a numpy array of shape (n_samples,) to use with DataValuationIssueManager, "
                f"but got {type(labels)} instead."
            )
            raise TypeError(error_msg)

        knn_graph, self.metric, _ = set_knn_graph(
            features=features,
            find_issues_kwargs=kwargs,
            metric=self.metric,
            k=self.k,
            statistics=self.datalab.get_info("statistics"),
        )

        # TODO: Check self.k against user-provided knn-graphs across all issue managers
        num_neighbors = num_neighbors_in_knn_graph(knn_graph)
        if self.k > num_neighbors:
            raise ValueError(
                f"The provided knn graph has {num_neighbors} neighbors, which is less than the required {self.k} neighbors. "
                "Please ensure that the knn graph you provide has at least as many neighbors as the required value of k."
            )

        scores = data_shapley_knn(labels, knn_graph=knn_graph, k=self.k)

        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": scores < self.threshold,
                self.issue_score_key: scores,
            },
        )
        self.summary = self.make_summary(score=scores.mean())

        self.info = self.collect_info(issues=self.issues, knn_graph=knn_graph)

    def collect_info(self, issues: pd.DataFrame, knn_graph: csr_matrix) -> dict:
        issues_info = {
            "num_low_valuation_issues": sum(issues[f"is_{self.issue_name}_issue"]),
            "average_data_valuation": issues[self.issue_score_key].mean(),
        }

        params_dict = {
            "metric": self.metric,
            "k": self.k,
            "threshold": self.threshold,
        }

        statistics_dict = self._build_statistics_dictionary(knn_graph=knn_graph)

        info_dict = {
            **issues_info,
            **params_dict,
            **statistics_dict,
        }

        return info_dict

    def _build_statistics_dictionary(self, knn_graph: csr_matrix) -> Dict[str, Dict[str, Any]]:
        statistics_dict: Dict[str, Dict[str, Any]] = {"statistics": {}}

        # Add the knn graph as a statistic if necessary
        graph_key = "weighted_knn_graph"
        old_knn_graph = self.datalab.get_info("statistics").get(graph_key, None)
        old_graph_exists = old_knn_graph is not None
        prefer_new_graph = (
            not old_graph_exists
            or (old_knn_graph is not None and knn_graph.nnz > old_knn_graph.nnz)
            or self.metric != self.datalab.get_info("statistics").get("knn_metric", None)
        )
        if prefer_new_graph:
            statistics_dict["statistics"][graph_key] = knn_graph
            if self.metric is not None:
                statistics_dict["statistics"]["knn_metric"] = self.metric

        return statistics_dict

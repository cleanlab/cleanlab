import numpy.typing as npt
from scipy.sparse import csr_matrix


from typing import Any, Dict, Optional, Tuple, cast

from cleanlab.internal.neighbor.knn_graph import create_knn_graph_and_index
from cleanlab.typing import Metric


def num_neighbors_in_knn_graph(knn_graph: csr_matrix) -> int:
    """Calculate the number of neighbors per row in a knn graph."""
    return knn_graph.nnz // knn_graph.shape[0]


def _process_knn_graph_from_inputs(
    kwargs: Dict[str, Any], statistics: Dict[str, Any], k_for_recomputation: int
) -> Optional[csr_matrix]:
    """Determine if a knn_graph is provided in the kwargs or if one is already stored in the associated Datalab instance."""
    knn_graph_kwargs: Optional[csr_matrix] = kwargs.get("knn_graph", None)
    knn_graph_stats = statistics.get("weighted_knn_graph", None)

    knn_graph: Optional[csr_matrix] = None
    if knn_graph_kwargs is not None:
        knn_graph = knn_graph_kwargs
        needs_recompute = False
    elif knn_graph_stats is not None:
        knn_graph = knn_graph_stats
        num_neighbors = num_neighbors_in_knn_graph(knn_graph) if knn_graph is not None else -1
        needs_recompute = k_for_recomputation > num_neighbors
        if needs_recompute:
            # If the provided knn graph is insufficient, then we need to recompute the knn graph
            # with the provided features
            knn_graph = None
    return knn_graph


def knn_exists(kwargs: Dict[str, Any], statistics: Dict[str, Any], k_needed: int) -> bool:
    """Check if a sufficiently large knn graph exists in the kwargs or statistics."""
    return (
        _process_knn_graph_from_inputs(kwargs, statistics, k_for_recomputation=k_needed) is not None
    )


def set_knn_graph(
    features: Optional[npt.NDArray],
    find_issues_kwargs: Dict[str, Any],
    metric: Optional[Metric],
    k: int,
    statistics: Dict[str, Any],
) -> Tuple[csr_matrix, Metric]:
    # This only fetches graph (optionally)
    knn_graph = _process_knn_graph_from_inputs(
        find_issues_kwargs, statistics, k_for_recomputation=k
    )
    old_knn_metric = statistics.get("knn_metric", metric)

    missing_knn_graph = knn_graph is None
    metric_changes = metric and metric != old_knn_metric
    if missing_knn_graph or metric_changes:
        assert features is not None, "Features must be provided to compute the knn graph."
        knn_graph, knn = create_knn_graph_and_index(features, n_neighbors=k, metric=metric)
        metric = knn.metric
    return cast(csr_matrix, knn_graph), cast(Metric, metric)

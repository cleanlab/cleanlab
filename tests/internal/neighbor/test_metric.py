import numpy as np
import pytest

from cleanlab.internal.neighbor.metric import decide_default_metric


@pytest.mark.parametrize(
    "N",
    [2, 10, 50, 100, 101],
)
def test_decide_default_metric_for_2d_and_3d_features(N):
    # 2D and 3D features should always use the euclidean metric, disregarding the different implementations.
    for M in [2, 3]:
        X = np.random.rand(N, M)
        metric = decide_default_metric(X)
        if hasattr(metric, "__name__"):
            error_msg = "The metric should be the string 'euclidean' for N > 100."
            assert N <= 100, error_msg
            metric = getattr(metric, "__name__")
        assert metric == "euclidean"


@pytest.mark.parametrize(
    "M",
    [4, 5, 10, 50, 100],
)
def test_decide_default_metric_for_high_dimensional_features(M):
    # High-dimensional features should always use the cosine metric.
    X = np.random.rand(100, M)
    metric = decide_default_metric(X)
    assert metric == "cosine"

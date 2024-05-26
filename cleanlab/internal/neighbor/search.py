from __future__ import annotations
from typing import TYPE_CHECKING

from sklearn.neighbors import NearestNeighbors


if TYPE_CHECKING:

    from cleanlab.typing import Metric


def construct_knn(n_neighbors: int, metric: Metric, **knn_kwargs) -> NearestNeighbors:
    """
    Constructs a k-nearest neighbors search object. You can implement a similar method to run cleanlab with your own approximate-KNN library.

    Parameters
    ----------
    n_neighbors :
        The number of nearest neighbors to consider.
    metric :
        The distance metric to use for computing distances between points.
        See :py:mod:`~cleanlab.internal.neighbor.metric` for more information.
    **knn_kwargs:
        Additional keyword arguments to be passed to the search index constructor.
        See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html for more details on the available options.

    Returns
    -------
    knn :
        A k-nearest neighbors search object compatible with the scikit-learn NearestNeighbors class interface.

        Implements:

            - `fit` method: Accepts a feature array `X` to fit the model.
                This enables subsequent neighbor searches on the data.
            - `kneighbors` method: Finds the K-neighbors of a point, returning distances and indices of the k-nearest neighbors. Handles two scenarios:
                1. When a query array `features: np.ndarray` is provided, it returns the distances and indices for each point in the query array.
                2. When no query array is provided (`features = None`), it returns neighbors for each indexed point without considering the query point as its own neighbor.
                Optionally, allows re-specification of the number of neighbors for each query point, defaulting to the constructor's value if not specified.

        Attributes:

            - `n_neighbors`: Number of neighbors to consider.
            - `metric`: Distance metric used to compute distances between points.
            - `metric_params`: Additional parameters for the distance metric function.

        Optional:

            - `kneighbors_graph` method: Not required but can be implemented for convenience.
            Responsibility shifted to :py:ref:`construct_knn_graph_from_index <cleanlab.internal.neighbor.neighbor.construct_knn_graph_from_index>`.

        Fitted Attributes:

            - `n_features_in_`: Number of features observed during fit.
            - `effective_metric_params_`: Metric parameters used in distance computation.
            - `effective_metric_`: Metric used for computing distances to neighbors.
            - `n_samples_fit_`: Number of samples in the fitted data.

        Additional:

            - `__sklearn_is_fitted__`: Method returning a boolean indicating if the object is fitted,
                useful for conducting an is_fitted validation, which verifies the presence of fitted attributes (typically ending with a trailing underscore).


    The above specifications ensure compatibility and provide a clear directive for developers needing to integrate alternative k-nearest neighbors implementations or modify existing functionalities.

    Note
    ----
    The `metric` argument should be a callable that takes two arguments (the two points) and returns the distance between them.
    The additional keyword arguments (`**knn_kwargs`) are passed directly to the underlying k-nearest neighbors search algorithm.

    """
    sklearn_knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, **knn_kwargs)

    return sklearn_knn

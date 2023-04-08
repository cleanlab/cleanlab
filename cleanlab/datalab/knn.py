"""This module contains the KNNGraph class, which is used to find the K nearest neighbors of data points in a dataset and
represent the dataset as weighted graph adjacency matrix (excluding self-loops).
"""

from abc import ABC
import warnings
import numpy as np

from typing import Any, Dict, Optional, Tuple, Union, cast
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors


class KNNInterface(ABC):
    """Necessary interface for any class handling KNN search to be compatible with `Datalab`."""

    n_neighbors: int
    metric: str

    def fit(self, features: Union[np.ndarray, csr_matrix]) -> None:
        raise NotImplementedError

    def kneighbors(
        self, features: Union[np.ndarray, csr_matrix], n_neighbors=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


KNNInterface.register(NearestNeighbors)


class KNN:
    """K-nearest neighbors (KNN) search class.

    This class is used to find the K nearest neighbors of data points in a
    dataset. This wraps existing KNN search implementations in scikit-learn
    or any other object that implements the `fit`/`kneighbors` interface.

    This class is opinionated in that it assumes that:
        - The KNN search object is stateful.
        - The `fit` method should only be called once.
        - The `kneighbors` method should return a tuple of neighbor indices
            and neighbor distances.

    Parameters
    ----------
    n_neighbors :
        Number of neighbors to search for.

        Note: This parameter is overridden if the `knn` parameter is passed.

    knn :
        KNN search object that implements the `fit`/`kneighbors` interface.
        If `None`, a NearestNeighbors search object is used.
    """

    def __init__(
        self, n_neighbors: int = 5, metric: Optional[str] = None, knn: Optional[KNNInterface] = None
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.knn = knn
        self.neighbor_indices: Optional[np.ndarray] = None
        self.neighbor_distances: Optional[np.ndarray] = None
        self._knn_graph: Optional[csr_matrix] = None
        self._fit_features: Optional[Union[np.ndarray, csr_matrix]] = None
        if self.knn is None:
            # Use the default KNN search implementation in scikit-learn.
            nearest_neighbors_kwargs = {}
            if self.metric is not None:
                nearest_neighbors_kwargs["metric"] = self.metric
            self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, **nearest_neighbors_kwargs)
        try:
            k = cast(int, self.knn.n_neighbors)  # Cast knn n_neighbors to int
            if k != self.n_neighbors:
                warnings.warn(
                    f"n_neighbors {self.n_neighbors} does not match n_neighbors "
                    f"{k} used to fit knn. "
                    "Most likely an existing NearestNeighbors object was passed in, "
                    "but a different n_neighbors was specified. "
                    "Using the n_neighbors found in the existing KNN search object."
                )
                self.n_neighbors = k
        except AttributeError:
            warnings.warn(
                "KNN search object does not have a n_neighbors attribute. "
                "Make sure that the number of neighbors being searched for "
                "does not differ from the n_neighbors parameter."
            )

        try:
            if self.knn.metric != self.metric:  # Cast knn metric to str
                if self.metric is not None:
                    warnings.warn(
                        f"metric {self.metric} does not match metric "
                        f"{self.knn.metric} used to fit knn. "
                        "Most likely an existing NearestNeighbors object was passed in, "
                        "but a different metric was specified. "
                        "Using the metric found in the existing KNN search object."
                    )
                self.metric = self.knn.metric
        except AttributeError:
            warnings.warn(
                "KNN search object does not have a metric attribute. "
                "Make sure that the metric being used "
                "does not differ from the metric parameter."
            )

    def fit(self, features: Union[np.ndarray, csr_matrix]) -> "KNN":
        """Fit the KNN search object.

        For convenience, this stores the features as an attribute for easy self-querying,
        regardless of how the search object builds its index.

        Parameters
        ----------
        features :
            A matrix of features for each data point.

            Warning
            -------
            While support for sparse matrices (corresponding to precomputed graphs) is planned,
            it is not fully implemented yet.
        """
        assert self.knn is not None, "KNN search object not initialized, cannot fit."
        self._fit_features = features
        self.knn.fit(self._fit_features)
        return self

    def kneighbors(
        self,
        features: Optional[Union[np.ndarray, csr_matrix]] = None,
        n_neighbors: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find the K nearest neighbors of each data point.

        Parameters
        ----------
        features :
            A matrix of features for each data point.

            Warning
            -------
            While support for sparse matrices (corresponding to precomputed graphs) is planned,
            it is not fully implemented yet.

        Returns
        -------
        neighbor_distances :
            An array of shape (n_queries, n_neighbors) containing the distances to the
            nearest neighbors of each query point.

        neighbor_indices :
            An array of shape (n_queries, n_neighbors) containing the indices of the
            nearest neighbors of each query point.

        Examples
        --------
        >>> from cleanlab.datalab.knn import KNN
        >>> knn = KNN(n_neighbors=2)
        >>> X = [[0.0, 0.4], [1.0, 0.2], [0.6, 0.2], [0.8, 1.0], [0.9, 1.0]]
        >>> knn.fit(X)
        >>> dists, ids = knn.kneighbors() # Ignores query points as neighbors
        >>> dists
        array([[0.63, 1.  ],
               [0.4 , 0.81],
               [0.4 , 0.63],
               [0.1 , 0.82],
               [0.1 , 0.81]])
        >>> ids
        array([[2, 3],
               [2, 4],
               [1, 0],
               [4, 1],
               [3, 1]])
        """
        index = cast(KNNInterface, self.knn)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if features is None:
            if (
                n_neighbors <= self.n_neighbors
                and self.neighbor_distances is not None
                and self.neighbor_indices is not None
            ):
                return (
                    self.neighbor_distances[:, :n_neighbors],
                    self.neighbor_indices[:, :n_neighbors],
                )
            n_neighbors += 1  # Ignore self-querying
            query = self._fit_features
            if query is None:
                raise NotFittedError(
                    "KNN search object not fit, cannot find neighbors. "
                    "Call the fit() method first."
                )
            try:
                neighbor_distances, neighbor_indices = index.kneighbors(
                    query, n_neighbors=n_neighbors
                )
                neighbor_distances = neighbor_distances[:, 1:]
                neighbor_indices = neighbor_indices[:, 1:]
            except TypeError as e:
                raise TypeError(
                    "KNN search object does not have a kneighbors method that accepts "
                    "the n_neighbors parameter. Make sure that the number of neighbors "
                    "being searched for does not differ from the n_neighbors parameter."
                ) from e
        else:
            query = features
            if self._fit_features is None:
                raise NotFittedError(
                    "KNN search object not fit, cannot find neighbors. "
                    "Call the fit() method first."
                )
            try:
                neighbor_distances, neighbor_indices = index.kneighbors(
                    query, n_neighbors=n_neighbors
                )
            except TypeError as e:
                raise TypeError(
                    "KNN search object does not have a kneighbors method that accepts "
                    "the n_neighbors parameter. Make sure that the number of neighbors "
                    "being searched for does not differ from the n_neighbors parameter."
                ) from e

        if features is None:
            cache_exists = self.neighbor_distances is not None and self.neighbor_indices is not None
            cache_too_small = cache_exists and (
                self.neighbor_distances.shape[0] < neighbor_distances.shape[0]
                or self.neighbor_distances.shape[1] < neighbor_distances.shape[1]
            )

            if not cache_exists or cache_too_small:
                self.neighbor_distances = neighbor_distances
                self.neighbor_indices = neighbor_indices
        return neighbor_distances, neighbor_indices

    def kneighbors_graph(
        self,
        features: Optional[Union[np.ndarray, csr_matrix]] = None,
        n_neighbors: Optional[int] = None,
    ) -> csr_matrix:
        """Find the K nearest neighbors of each data point and return a graph adjacency matrix.

        Parameters
        ----------
        features :
            A matrix of features for each data point. If this is not
            specified, then the KNN search object must have been fit before

            Warning
            -------
            While support for sparse matrices (corresponding to precomputed graphs) is planned,
            it is not fully implemented yet.

        n_neighbors :
            The number of neighbors to search for. If this is not
            specified, then the KNN search object must have been fit before
            and the number of neighbors used to fit the KNN search object
            will be used.

        Returns
        -------
        knn_graph :
            A sparse matrix containing the K nearest neighbors of each data point, weighted by the
            distance to the nearest neighbor.

        Examples
        --------
        >>> from cleanlab.datalab.knn import KNN
        >>> knn = KNN(n_neighbors=2)
        >>> X = [[0.0, 0.4], [1.0, 0.2], [0.6, 0.2], [0.8, 1.0], [0.9, 1.0]]
        >>> knn.fit(X)
        >>> graph = knn.kneighbors_graph()
        >>> graph.toarray()  # Don't do this for large graphs!
        array([[0.  , 0.  , 0.63, 1.  , 0.  ],
               [0.  , 0.  , 0.4 , 0.  , 0.81],
               [0.63, 0.4 , 0.  , 0.  , 0.  ],
               [0.  , 0.82, 0.  , 0.  , 0.1 ],
               [0.  , 0.81, 0.  , 0.1 , 0.  ]])
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if features is None:
            cache_available = (
                n_neighbors <= self.n_neighbors
                and self.neighbor_distances is not None
                and self.neighbor_indices is not None
            )
            if cache_available:
                neighbor_distances = cast(np.ndarray, self.neighbor_distances)
                neighbor_indices = cast(np.ndarray, self.neighbor_indices)
                dists, ids = neighbor_distances, neighbor_indices
            else:
                dists, ids = self.kneighbors(n_neighbors=n_neighbors)
        else:
            dists, ids = self.kneighbors(features, n_neighbors=n_neighbors)

        N = ids.shape[0]

        nnz = N * n_neighbors
        indptr = np.arange(0, nnz + 1, n_neighbors)
        self._knn_graph = csr_matrix((dists.ravel(), ids.ravel(), indptr), shape=(N, N))

        return self._knn_graph

    def radius_neighbors(
        self, features: Optional[np.ndarray] = None, radius: Optional[float] = None
    ):
        """Find the neighbors of each data point within a given radius.

        Warning
        -------
        This method is not yet fully implemented. For now, it is only available for
        KNN search objects that implement the radius_neighbors method themselves,
        such as sklearn.neighbors.NearestNeighbors.

        Parameters
        ----------
        features :
            A matrix of features for each data point. If this is not
            specified, then the KNN search object must have been fit before and have
            access to the fit features.

        radius :
            The radius to search for neighbors within. If this is not
            specified, then the default radius of the KNN search instance variable
            will be used.
        """

        kwargs: Dict[str, Any] = {"return_distance": False}
        if features is not None:
            kwargs["X"] = features
        if radius is not None:
            kwargs["radius"] = radius

        # Assume that the underlying KNN search object has a radius_neighbors method
        try:
            knn = cast(NearestNeighbors, self.knn)
            if not isinstance(knn, NearestNeighbors):
                warnings.warn(
                    "KNN search object is not an instance of sklearn.neighbors.NearestNeighbors. "
                    "Calling the radius_neighbors method may cause unexpected behavior."
                )
            return knn.radius_neighbors(**kwargs)
        except AttributeError as e:
            error_message = (
                "KNN search object does not have a radius_neighbors method. "
                "For now, only sklearn.neighbors.NearestNeighbors objects support "
                "the radius_neighbors method out of the box. "
                "Make sure that the radius_neighbors method is implemented "
                "for the KNN search object."
            )
            raise NotImplementedError(error_message) from e

    def add_item(self, X: np.ndarray) -> "KNN":
        """Add a vector/matrix to the fit features.

        This is a convenience method for setting/adding fit-features when the underlying
        KNN search object is already fit before instantiating this class.

        Warning
        -------
        This method does not interact with the KNN search object's internal
        index/cache. If you add items to the fit features, then you must
        rebuild the index/cache yourself.

        Parameters
        ----------
        X :
            A vector or matrix of features to add to the fit features.

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.neighbors import NearestNeighbors
        >>> from cleanlab.datalab.knn import KNN
        >>> knn_sklearn = NearestNeighbors(n_neighbors=2)
        >>> X = [[0.0, 0.4], [1.0, 0.2], [0.6, 0.2], [0.8, 1.0], [0.9, 1.0]]
        >>> knn_sklearn.fit(X)
        >>> dists, ids = knn_sklearn.kneighbors()
        >>> dists
        array([[0.63, 1.  ],
               [0.4 , 0.81],
               [0.4 , 0.63],
               [0.1 , 0.82],
               [0.1 , 0.81]])
        >>> ids
        array([[2, 3],
               [2, 4],
               [1, 0],
               [4, 1],
               [3, 1]])
        >>> knn = KNN(knn=knn_sklearn)
        >>> # knn.kneighbors() # This will fail because the `knn` has not been fit.
        >>> knn.add_item(X) # Provide hint to the `knn` that the internal index is already fit.
        >>> new_dists, new_ids = knn.kneighbors()
        >>> np.allclose(dists, new_dists)
        True
        >>> np.allclose(ids, new_ids)
        """
        if isinstance(X, list):
            X = np.array(X)
        X = X.reshape(1, -1) if X.ndim == 1 else X
        if self._fit_features is None:
            self._fit_features = X
            return self

        if X.shape[1] != self._fit_features.shape[1]:
            raise ValueError(
                "New item has a different number of features than the fit features."
                f"New item has {X.shape[1]} features, but fit features have "
                f"{self._fit_features.shape[1]} features."
            )
        self._fit_features = np.vstack((self._fit_features, X))

        return self

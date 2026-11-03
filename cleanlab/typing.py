from typing import Any, Callable, Dict, Iterator, Protocol, Tuple, Union
import numpy as np
import pandas as pd

LabelLike = Union[list, np.ndarray, pd.Series, pd.DataFrame]
"""Type for objects that behave like collections of labels."""


DatasetLike = Any
"""Type for objects that behave like datasets."""


class CrossValidationSplitter(Protocol):
    """Structural interface for cross-validation splitters (e.g. ``sklearn.model_selection.KFold``)."""

    def split(self, X: np.ndarray, y: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]: ...


class MultilabelClassifier(Protocol):
    """Estimator interface used when obtaining cross-validated predicted probabilities."""

    def fit(self, X: DatasetLike, y: np.ndarray) -> object: ...

    def predict_proba(self, X: DatasetLike) -> np.ndarray: ...

    def get_params(self, deep: bool = True) -> Dict[str, Any]: ...


###########################################################
# Types aliases used in cleanlab/internal/neighbor/ modules
###########################################################

FeatureArray = np.ndarray
"""A type alias for a 2D numpy array representing numerical features."""
Metric = Union[str, Callable]
"""A type alias for the distance metric to be used for neighbor search. It can be either a string
representing the metric name ("cosine" or "euclidean") or a callable representing the metric function from scipy (euclidean).

Valid values for metric are mentioned in the scikit-learn documentation for the sklearn.metrics.pairwise_distances function.

See Also
--------
sklearn.metrics.pairwise_distances: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn-metrics-pairwise-distances
"""

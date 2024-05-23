from typing import Any, Callable, Union
import numpy as np
import pandas as pd

LabelLike = Union[list, np.ndarray, pd.Series, pd.DataFrame]
"""Type for objects that behave like collections of labels."""


DatasetLike = Any
"""Type for objects that behave like datasets."""

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

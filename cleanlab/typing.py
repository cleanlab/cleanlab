from typing import Any, Union
import numpy as np
import pandas as pd

LabelLike = Union[list, np.ndarray, pd.Series, pd.DataFrame]
"""Type for objects that behave like collections of labels."""


DatasetLike = Any
"""Type for objects that behave like datasets."""

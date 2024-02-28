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
"""
This module contains the ModelOutput class, which is used internally within Datalab
to represent model outputs (e.g. predictions, probabilities, etc.) and process them
for issue finding.
This class and associated naming conventions are subject to change and is not meant
to be used by users.
"""


from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelOutput(ABC):
    """
    An abstract class for representing model outputs (e.g. predictions, probabilities, etc.)
    for internal use within Datalab. This class is not meant to be used by users.

    It is used internally within the issue-finding process Datalab runs to assign
    types to the data and process it accordingly.

    Parameters
    ----------
    data : array-like
        The model outputs. Not to be confused with the data used to train the model.
        This is mainly intended for NumPy arrays.
    """

    data: np.ndarray

    @abstractmethod
    def validate(self):
        """
        Validate the data format and content.
        E.g. a pred_probs object used for classification
        should be a 2D array with values between 0 and 1 and sum to 1 for each row.
        """
        pass

    @abstractmethod
    def collect(self):
        """
        Fetch the data for issue finding.
        Usually this is just the data itself, but sometimes it may be a transformation
        of the data (e.g. a 1D array of predictions from a 2D array of predicted probabilities).
        """
        pass


class MultiClassPredProbs(ModelOutput):
    """
    A class for representing a model's predicted probabilities for each class
    in a multi-class classification problem. This class is not meant to be used by users.
    """

    argument = "pred_probs"

    def validate(self):
        pred_probs = self.data
        if pred_probs.ndim != 2:
            raise ValueError("pred_probs must be a 2D array for multi-class classification")
        if not np.all((pred_probs >= 0) & (pred_probs <= 1)):
            incorrect_range = (np.min(pred_probs), np.max(pred_probs))
            raise ValueError(
                "Expected pred_probs to be between 0 and 1 for multi-label classification,"
                f" but got values in range {incorrect_range} instead."
            )
        if not np.allclose(np.sum(pred_probs, axis=1), 1):
            raise ValueError("pred_probs must sum to 1 for each row for multi-class classification")

    def collect(self):
        return self.data


class RegressionPredictions(ModelOutput):
    """
    A class for representing a model's predictions for a regression problem.
    This class is not meant to be used by users.
    """

    argument = "predictions"

    def validate(self):
        predictions = self.data
        if predictions.ndim != 1:
            raise ValueError("pred_probs must be a 1D array for regression")

    def collect(self):
        return self.data


class MultiLabelPredProbs(ModelOutput):
    """
    A class for representing a model's predicted probabilities for each class
    in a multilabel classification problem. This class is not meant to be used by users.
    """

    argument = "pred_probs"

    def validate(self):
        pred_probs = self.data
        if pred_probs.ndim != 2:
            raise ValueError(
                f"Expected pred_probs to be a 2D array for multi-label classification,"
                " but got {pred_probs.ndim}D array instead."
            )
        if not np.all((pred_probs >= 0) & (pred_probs <= 1)):
            incorrect_range = (np.min(pred_probs), np.max(pred_probs))
            raise ValueError(
                "Expected pred_probs to be between 0 and 1 for multi-label classification,"
                f" but got values in range {incorrect_range} instead."
            )

    def collect(self):
        return self.data

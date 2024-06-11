from dataclasses import dataclass
from typing import List, Optional, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")


@dataclass
class SpuriousCorrelations:
    data: pd.DataFrame
    labels: Union[np.ndarray, list]
    properties_of_interest: Optional[List[str]] = None

    def __post_init__(self):
        # Must have same number of rows
        if not len(self.data) == len(self.labels):
            raise ValueError(
                "The number of rows in the data dataframe must be the same as the number of labels."
            )

        # Set default properties_of_interest if not provided
        if self.properties_of_interest is None:
            self.properties_of_interest = self.data.columns.tolist()

        if not all(isinstance(p, str) for p in self.properties_of_interest):
            raise TypeError("properties_of_interest must be a list of strings.")

    def calculate_correlations(self) -> pd.DataFrame:
        """Calculates the spurious correlation scores for each property of interest found in the dataset."""
        baseline_accuracy = self._get_baseline()
        assert (
            self.properties_of_interest is not None
        ), "properties_of_interest must be set, but is None."
        property_scores = {
            str(property_of_interest): self.calculate_spurious_correlation(
                property_of_interest, baseline_accuracy
            )
            for property_of_interest in self.properties_of_interest
        }
        data_score = pd.DataFrame(list(property_scores.items()), columns=["property", "score"])
        return data_score

    def _get_baseline(self) -> float:
        """Calculates the baseline accuracy of the dataset. The baseline model is predicting the most common label."""
        baseline_accuracy = np.bincount(self.labels).argmax() / len(self.labels)
        return float(baseline_accuracy)

    def calculate_spurious_correlation(
        self, property_of_interest, baseline_accuracy: float
    ) -> float:
        """Scores the dataset based on a given property of interest.

        Parameters
        ----------
        property_of_interest :
            The property of interest to score the dataset on.

        baseline_accuracy :
            The accuracy of the baseline model.

        Returns
        -------
        score :
            A correlation score of the dataset's labels to the property of interest.
        """
        X = self.data[property_of_interest].values.reshape(-1, 1)
        y = self.labels
        mean_accuracy = _train_and_eval(X, y)
        return relative_room_for_improvement(baseline_accuracy, float(mean_accuracy))


def _train_and_eval(X, y, cv=5) -> float:
    classifier = GaussianNB()  # TODO: Make this a parameter
    cv_accuracies = cross_val_score(classifier, X, y, cv=cv, scoring="accuracy")
    mean_accuracy = float(np.mean(cv_accuracies))
    return mean_accuracy


def relative_room_for_improvement(
    baseline_accuracy: float, mean_accuracy: float, eps: float = 1e-8
) -> float:
    """
    Calculate the relative room for improvement given a baseline and trial accuracy.

    This function computes the ratio of the difference between perfect accuracy (1.0)
    and the trial accuracy to the difference between perfect accuracy and the baseline accuracy.
    If the baseline accuracy is perfect (i.e., 1.0), an epsilon value is added to the denominator
    to avoid division by zero.

    Parameters
    ----------
    baseline_accuracy :
        The accuracy of the baseline model. Must be between 0 and 1.
    mean_accuracy :
        The accuracy of the trial model being compared. Must be between 0 and 1.
    eps :
        A small constant to avoid division by zero when baseline accuracy is 1. Defaults to 1e-8.

    Returns
    -------
    score :
        The relative room for improvement, bounded between 0 and 1.
    """
    numerator = 1 - mean_accuracy
    denominator = 1 - baseline_accuracy
    if baseline_accuracy == 1:
        denominator += eps
    return min(1, numerator / denominator)

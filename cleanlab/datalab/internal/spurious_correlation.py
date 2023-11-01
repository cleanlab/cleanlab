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
    """Calculates the spurious correlation scores for a given dataset.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from cleanlab.datalab.internal.spurious_correlation import SpuriousCorrelations
    >>>
    >>> # Generate a dataset with properties potentially correlated with the labels
    >>> data = pd.DataFrame({
    >>>     "property_a": [-0.28, 0.99, -0.1, 0.81, -0.84, -0.66, 3.12, 0.77, 0.28, 0.28, -0.39, -0.38,
    ...                     -0.2,  1.28, 0.18, 1.64, 1.24, -0.22, 0.73, -0.55],
    ...     "property_b": [0.75, 0.55, -0.05, 1.09, 0.07, -0.03, 0.68, 0.61, 0.31, 0.7,  0.89, -0.27,
    ...                     0.49, 0.05, 0.57, 0.69, 0.89, 1.01, 0.76, 0.77],
    ... })
    >>> labels = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    >>>
    >>> # Calculate the spurious correlation scores
    >>> SpuriousCorrelations(data, labels).calculate_correlations()
          property	   score
    0   property_a	0.315789
    1   property_b	0.052632

    """

    data: pd.DataFrame
    """A dataframe containing the data to be scored."""
    labels: Union[np.ndarray, list]
    """The colletion of labels to compute the spurious correlation scores on."""
    properties_of_interest: Optional[List[str]] = None
    """A list of strings in the dataframe to score the dataset on. If None, all columns in the dataframe will be scored."""

    def __post_init__(self):
        # Must have same number of rows
        if not len(self.data) == len(self.labels):
            raise ValueError(
                "The number of rows in the data dataframe must be the same as the number of labels."
            )

        # Set default properties_of_interest if not provided
        if self.properties_of_interest is None:
            self.properties_of_interest = self.data.columns.tolist()

        assert all(
            isinstance(p, str) for p in self.properties_of_interest
        ), "properties_of_interest must be a list of strings."

        assert all(
            p in self.data.columns for p in self.properties_of_interest
        ), "properties_of_interest must be a subset of the columns in the data dataframe."

    def calculate_correlations(self) -> pd.DataFrame:
        """Calculates the spurious correlation scores for each property of interest found in the dataset.

        See Also
        --------
        relative_room_for_improvement
        """
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

        See Also
        --------
        relative_room_for_improvement
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
):
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

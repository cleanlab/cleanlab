import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from cleanlab import Datalab


class SpuriousCorrelations:
    def __init__(self, data: Datalab) -> None:
        self.data = data
        self.issues: pd.DataFrame = data.issues
        self.labels: np.ndarray = data.labels
        self.issue_summary: pd.DataFrame = data.issue_summary

    def spurious_correlations(self) -> pd.DataFrame:
        baseline_accuracy = np.bincount(self.labels).argmax() / len(self.labels)
        property_scores = {}
        image_properties_of_interest = [
            "outlier",
            "near_duplicate",
            "non_iid",
            "low_information",
            "dark",
            "blurry",
            "light",
            "grayscale",
            "odd_aspect_ratio",
            "odd_size",
        ]
        image_properties = [
            i
            for i in image_properties_of_interest
            if i in self.issue_summary["issue_type"].tolist()
        ]
        for property_of_interest in image_properties:
            if (
                self.issue_summary[self.issue_summary["issue_type"] == property_of_interest][
                    "num_issues"
                ].values[0]
                > 0
            ):
                S = self.calculate_spurious_correlation(property_of_interest, baseline_accuracy)
                property_scores[f"{property_of_interest}"] = S
        data_score = pd.DataFrame(
            list(property_scores.items()), columns=["image_property", "label_prediction_error"]
        )
        return data_score

    def calculate_spurious_correlation(
        self, property_of_interest: str, baseline_accuracy: float, cv_folds: int = 5
    ) -> float:
        X = self.issues[f"{property_of_interest}_score"].values.reshape(-1, 1)
        y = self.labels
        classifier = GaussianNB()
        cv_accuracies = cross_val_score(classifier, X, y, cv=cv_folds, scoring="accuracy")
        mean_accuracy = np.mean(cv_accuracies)
        eps = 1e-8
        S = min(1, (1 - mean_accuracy) / (1 - baseline_accuracy + eps))
        return S

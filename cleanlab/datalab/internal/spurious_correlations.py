import numpy as np 
import pandas as pd
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.naive_bayes import GaussianNB
from statistics import mode
import warnings 
warnings.filterwarnings('ignore')

from datalab import DataLab

class SpuriousCorrelations:

    def __init__(self, data: DataLab) -> None:
        self.data = data
        self.issues = data.issues
        self.labels = data.labels  

    def spurious_correlations(self) -> pd.DataFrame:
        baseline_accuracy = np.bincount(self.labels).argmax() / len(self.labels)
        image_properties = ["near_duplicate_score", "blurry_score", "light_score", "low_information_score", "dark_score", "grayscale_score", "odd_aspect_ratio_score", "odd_size_score"]
        property_scores = {}
        for property_of_interest in image_properties:
            S = self.calculate_spurious_correlation(property_of_interest, baseline_accuracy)
            property_scores[f'{property_of_interest}'] = S
        data_score = pd.DataFrame(list(property_scores.items()), columns=['image_property', 'Overall_score'])
        return data_score

    def calculate_spurious_correlation(self, property_of_interest, baseline_accuracy):
        X = self.issues[property_of_interest].values.reshape(-1, 1)
        y = self.labels  
        classifier = GaussianNB()
        cv_accuracies = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
        mean_accuracy = np.mean(cv_accuracies)
        eps = 1e-8
        S = min(1, (1 - mean_accuracy) / (1 - baseline_accuracy + eps))
        return S


import numpy as np
import pandas as pd
import pytest
from cleanlab import Datalab


SEED = 42


class TestAllIdenticalExamplesDataset:
    N = 20
    K = 5

    @pytest.fixture
    def dataset(self):
        np.random.seed(SEED)
        # Create the dataset with all identical examples
        X = np.full((self.N, self.K), fill_value=np.random.rand(self.K))
        return {"X": X}

    @pytest.fixture
    def dataset_with_one_unique_example(self, dataset):
        # Add one unique example to the dataset
        dataset["X"] = np.vstack([dataset["X"], np.random.rand(self.K)])
        return dataset

    @pytest.fixture
    def lab(self, dataset):
        return Datalab(data=dataset)

    @pytest.fixture
    def lab_with_one_unique_example(self, dataset_with_one_unique_example):
        return Datalab(data=dataset_with_one_unique_example)

    def test_issue_detection(self, lab, dataset):

        lab.find_issues(features=dataset["X"])

        outlier_issues = lab.get_issues("outlier")
        expected_outlier_issues = pd.DataFrame(
            [{"is_outlier_issue": False, "outlier_score": 1.0}] * self.N
        )
        pd.testing.assert_frame_equal(outlier_issues, expected_outlier_issues)

        near_duplicate_issues = lab.get_issues("near_duplicate")
        expected_near_duplicate_issues = pd.DataFrame(
            [
                {
                    "is_near_duplicate_issue": True,
                    "near_duplicate_score": 0.0,
                    "distance_to_nearest_neighbor": np.finfo(np.float64).epsneg,
                }
                for i in range(self.N)
            ]
        )
        pd.testing.assert_frame_equal(
            near_duplicate_issues.drop(columns="near_duplicate_sets"),
            expected_near_duplicate_issues,
        )

    def test_issue_detection_with_one_unique_example(
        self, lab_with_one_unique_example, dataset_with_one_unique_example
    ):
        lab = lab_with_one_unique_example
        dataset = dataset_with_one_unique_example

        lab.find_issues(features=dataset["X"])

        outlier_issues = lab.get_issues("outlier")
        expected_outlier_issues = pd.DataFrame(
            [{"is_outlier_issue": False, "outlier_score": 1.0}] * self.N
            + [{"is_outlier_issue": True, "outlier_score": 0.0}]
        )

        pd.testing.assert_frame_equal(outlier_issues, expected_outlier_issues)

        near_duplicate_issues = lab.get_issues("near_duplicate")
        expected_near_duplicate_issues = pd.DataFrame(
            [
                {
                    "is_near_duplicate_issue": True,
                    "near_duplicate_score": 0.0,
                    "distance_to_nearest_neighbor": np.finfo(np.float64).epsneg,
                }
                for i in range(self.N)
            ]
            + [
                {
                    "is_near_duplicate_issue": False,
                    "near_duplicate_score": 1.0,
                    "distance_to_nearest_neighbor": 3.212390e-01,
                }
            ]
        )
        pd.testing.assert_frame_equal(
            near_duplicate_issues.drop(columns="near_duplicate_sets"),
            expected_near_duplicate_issues,
        )

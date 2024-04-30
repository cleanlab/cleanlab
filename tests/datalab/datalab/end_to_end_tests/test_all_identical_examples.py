import numpy as np
import pandas as pd
import pytest
from cleanlab import Datalab


class TestAllIdenticalExamplesDataset:
    @pytest.fixture
    def dataset(self, request):
        N, K = request.param
        # Create the dataset with all identical examples
        X = np.full((N, K), fill_value=np.random.rand(K))
        return {"X": X}

    @pytest.fixture
    def dataset_with_one_unique_example(self, request):
        N, K = request.param
        # Create the dataset with all identical examples
        X = np.full((N, K), fill_value=np.random.rand(K))

        # Add one unique example to the dataset
        X = np.vstack([X, np.random.rand(K)])
        return {"X": X}

    @pytest.fixture
    def lab(self, dataset):
        return Datalab(data=dataset)

    @pytest.fixture
    def lab_with_one_unique_example(self, dataset_with_one_unique_example):
        return Datalab(data=dataset_with_one_unique_example)

    @pytest.mark.parametrize(
        "dataset",
        [((N, K)) for N in [11, 20, 50, 100, 150] for K in [2, 3, 5, 10, 20]],
        indirect=["dataset"],
        ids=lambda x: f"N={x[0]}, K={x[1]}",
    )
    def test_issue_detection(self, dataset):
        lab = Datalab(data=dataset)
        lab.find_issues(features=dataset["X"])

        outlier_issues = lab.get_issues("outlier")
        expected_outlier_issues = pd.DataFrame(
            [{"is_outlier_issue": False, "outlier_score": 1.0}] * len(outlier_issues)
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
                for i in range(len(near_duplicate_issues))
            ]
        )
        pd.testing.assert_frame_equal(
            near_duplicate_issues.drop(columns="near_duplicate_sets"),
            expected_near_duplicate_issues,
            check_exact=False,
            atol=5e-16,
        )

    @pytest.mark.parametrize(
        "dataset_with_one_unique_example",
        [((N, K)) for N in [11, 20, 50, 100, 150] for K in [2, 3, 5, 10, 20]],
        indirect=["dataset_with_one_unique_example"],
        ids=lambda x: f"N={x[0]}, K={x[1]}",
    )
    def test_issue_detection_with_one_unique_example(self, dataset_with_one_unique_example):
        dataset = dataset_with_one_unique_example
        lab = Datalab(data=dataset)
        lab.find_issues(features=dataset["X"])

        outlier_issues = lab.get_issues("outlier")
        expected_outlier_issues = pd.DataFrame(
            [{"is_outlier_issue": False, "outlier_score": 1.0}] * (len(outlier_issues) - 1)
            + [{"is_outlier_issue": True, "outlier_score": 0.0}]
        )

        pd.testing.assert_frame_equal(outlier_issues, expected_outlier_issues)

        near_duplicate_issues = lab.get_issues("near_duplicate")
        expected_near_duplicate_issues = pd.DataFrame(
            [
                {
                    "is_near_duplicate_issue": True,
                    "near_duplicate_score": 0.0,
                }
                for i in range(len(near_duplicate_issues) - 1)
            ]
            + [
                {
                    "is_near_duplicate_issue": False,
                    "near_duplicate_score": 1.0,
                }
            ]
        )
        pd.testing.assert_frame_equal(
            near_duplicate_issues.drop(
                columns=["near_duplicate_sets", "distance_to_nearest_neighbor"]
            ),
            expected_near_duplicate_issues,
        )

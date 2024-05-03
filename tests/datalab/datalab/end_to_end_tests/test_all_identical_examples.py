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

        # All labels for identical points should be the same
        y = ["a"] * N
        # One of the labels is different, so it should be flagged with a label issue
        y[-1] = "b"

        return {"X": X, "y": y}

    @pytest.fixture
    def dataset_with_one_unique_example(self, request):
        N, K = request.param
        # Create the dataset with all identical examples
        X = np.full((N, K), fill_value=np.random.rand(K))

        # Add one unique example to the dataset
        X = np.vstack([X, np.random.rand(K)])

        # All labels for identical points should be the same, let's make them all 0, along with the unique example
        y = ["a"] * (N + 1)
        # One of the N points has a noisy label, it should be flagged with a label issue
        y[-2] = "b"

        return {"X": X, "y": y}

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
        lab = Datalab(data=dataset, label_name="y")
        pred_probs = np.full((len(dataset["y"]), 2), fill_value=[1.0, 0.0])
        lab.find_issues(features=dataset["X"], pred_probs=pred_probs)

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

        lab_issues = lab.get_issues("label")[["is_label_issue", "label_score"]]
        expected_lab_issues = pd.DataFrame(
            [{"is_label_issue": False, "label_score": 1.0}] * (len(lab_issues) - 1)
            + [
                {"is_label_issue": False, "label_score": 0.0}
            ]  # The confident threshold for this examble is extremely low, but won't flag it as a label issue
        )
        pd.testing.assert_frame_equal(lab_issues, expected_lab_issues)

    @pytest.mark.parametrize(
        "dataset_with_one_unique_example",
        [((N, K)) for N in [11, 20, 50, 100, 150] for K in [2, 3, 5, 10, 20]],
        indirect=["dataset_with_one_unique_example"],
        ids=lambda x: f"N={x[0]}, K={x[1]}",
    )
    def test_issue_detection_with_one_unique_example(self, dataset_with_one_unique_example):
        dataset = dataset_with_one_unique_example
        lab = Datalab(data=dataset, label_name="y")
        pred_probs = np.full((len(dataset["y"]), 2), fill_value=[1.0, 0.0])
        lab.find_issues(features=dataset["X"], pred_probs=pred_probs)

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

        lab_issues = lab.get_issues("label")[["is_label_issue", "label_score"]]
        expected_lab_issues = pd.DataFrame(
            [{"is_label_issue": False, "label_score": 1.0}] * (len(lab_issues) - 2)
            + [
                {"is_label_issue": False, "label_score": 0.0}
            ]  # The confident threshold for this examble is extremely low, but won't flag it as a label issue
            + [{"is_label_issue": False, "label_score": 1.0}]
        )
        pd.testing.assert_frame_equal(lab_issues, expected_lab_issues)

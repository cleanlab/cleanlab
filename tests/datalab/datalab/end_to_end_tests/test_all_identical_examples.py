import numpy as np
import pandas as pd
import pytest
from cleanlab import Datalab


class TestAllIdenticalExamplesDataset:
    """There are 4 types of datasets this class tests:

    1. A dataset with all identical data points. They all have identical labels, EXCEPT THE LAST ONE.
    2. A dataset with all identical data points, except for one extra/unique example. One of the identical examples has a noisy label. The unique example has the same label as the majority of the dataset.
    3. A regression dataset with all identical data points. They all have identical targets, EXCEPT THE LAST ONE.
    4. A regression dataset with all identical data points, except for one extra/unique example. One of the identical examples has a noisy target. The unique example has the same target as the majority of the dataset.

    Each test goes through the same motions:

    - Set up Datalab instance
    - Find default issues from features and pred_probs/predictions
    - Assert that the following issue dataframes look as expected for each dataset type.
      - label issues
      - outlier issues
      - near-duplicate issues

    There are a few more issue types that are tested in some cases, but they are less prone to be affected by future changes in the codebase.
    - underperforming group issues (for classification)
    - class imbalance issues (for classification)

    These tests focus on the issue detection capabilities of Datalab, and not the quality of the predictions or the features.
    """

    @pytest.fixture
    def dataset(self, request):
        N, M = request.param
        # Create the dataset with all identical examples
        X = np.full((N, M), fill_value=np.random.rand(M))

        # All labels for identical points should be the same
        y = ["a"] * N
        # One of the labels is different, so it should be flagged with a label issue
        y[-1] = "b"

        return {"X": X, "y": y}

    @pytest.fixture
    def dataset_with_one_unique_example(self, request):
        N, M = request.param
        # Create the dataset with all identical examples
        X = np.full((N, M), fill_value=np.random.rand(M))

        # Add one unique example to the dataset
        X = np.vstack([X, np.random.rand(M)])

        # All labels for identical points should be the same, let's make them all 0, along with the unique example
        y = ["a"] * (N + 1)
        # One of the N points has a noisy label, it should be flagged with a label issue
        y[-2] = "b"

        return {"X": X, "y": y}

    @pytest.fixture
    def regression_dataset(self, request):
        N, M = request.param
        # Create the dataset with all identical examples
        X = np.full((N, M), fill_value=np.random.rand(M))

        # All labels for identical points should be the same
        y = np.full(N, fill_value=np.random.rand())
        # Flip one of the targets to introduce a label issue
        y[-1] += 10

        return {"X": X, "y": y}

    @pytest.fixture
    def regression_dataset_with_one_unique_example(self, request):
        N, M = request.param
        # Create the dataset with all identical examples
        X = np.full((N, M), fill_value=np.random.rand(M))

        # All labels for identical points should be the same
        y = np.full(N, fill_value=np.random.rand())
        # Flip one of the targets to introduce a label issue
        y[-1] += 10

        # Add one unique example to the dataset, but it has the same target as the majority of the dataset
        X = np.vstack([X, np.random.rand(M)])
        y = np.append(y, [y[0]])

        return {"X": X, "y": y}

    @pytest.mark.parametrize(
        "dataset",
        [((N, M)) for N in [11, 20, 50, 100, 150] for M in [2, 3, 5, 10, 20]],
        indirect=["dataset"],
        ids=lambda x: f"N={x[0]}, M={x[1]}",
    )
    def test_issue_detection(self, dataset):
        lab = Datalab(data=dataset, label_name="y")
        N = len(dataset["y"])
        pred_probs = np.full((N, 2), fill_value=[1.0, 0.0])
        lab.find_issues(features=dataset["X"], pred_probs=pred_probs)

        outlier_issues = lab.get_issues("outlier")
        expected_outlier_issues = pd.DataFrame(
            [{"is_outlier_issue": False, "outlier_score": 1.0}] * N
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
                for i in range(N)
            ]
        )
        pd.testing.assert_frame_equal(
            near_duplicate_issues.drop(columns="near_duplicate_sets"),
            expected_near_duplicate_issues,
            check_exact=False,
            atol=5e-16,
        )

        label_issues = lab.get_issues("label")[["is_label_issue", "label_score"]]
        expected_label_issues = pd.DataFrame(
            [{"is_label_issue": False, "label_score": 1.0}] * (N - 1)
            + [
                {"is_label_issue": False, "label_score": 0.0}
            ]  # The confident threshold for this examble is extremely low, but won't flag it as a label issue
        )
        pd.testing.assert_frame_equal(label_issues, expected_label_issues)

        underperforming_group_issues = lab.get_issues("underperforming_group")
        expected_underperforming_group_issues = pd.DataFrame(
            [
                {
                    "is_underperforming_group_issue": False,
                    "underperforming_group_score": 1.0,
                }
            ]
            * N  # Only a single data point performs poorly, so it's not in any group
        )
        pd.testing.assert_frame_equal(
            underperforming_group_issues, expected_underperforming_group_issues
        )

        if N > 20:
            # Default threshold for class imbalance is 0.1*1/K, where K is the number of classes. So for 20 examples and 2 classes,
            # the threshold is 5% (so 1 example out of 20 is NOT considered class imbalance)
            class_imbalance_issues = lab.get_issues("class_imbalance")[
                ["is_class_imbalance_issue", "class_imbalance_score"]
            ]
            expected_class_imbalance_issues = pd.DataFrame(
                [
                    {
                        "is_class_imbalance_issue": False,
                        "class_imbalance_score": 1.0,
                    }
                ]
                * (N - 1)
                + [
                    {
                        "is_class_imbalance_issue": True,
                        "class_imbalance_score": 1 / N,
                    }
                ]
            )
            pd.testing.assert_frame_equal(class_imbalance_issues, expected_class_imbalance_issues)

    @pytest.mark.parametrize(
        "dataset_with_one_unique_example",
        [((N, M)) for N in [11, 20, 50, 100, 150] for M in [2, 3, 5, 10, 20]],
        indirect=["dataset_with_one_unique_example"],
        ids=lambda x: f"N={x[0]}, M={x[1]}",
    )
    def test_issue_detection_with_one_unique_example(self, dataset_with_one_unique_example):
        dataset = dataset_with_one_unique_example
        N = len(dataset["y"])
        lab = Datalab(data=dataset, label_name="y")
        pred_probs = np.full((N, 2), fill_value=[1.0, 0.0])
        lab.find_issues(features=dataset["X"], pred_probs=pred_probs)

        outlier_issues = lab.get_issues("outlier")
        expected_outlier_issues = pd.DataFrame(
            [{"is_outlier_issue": False, "outlier_score": 1.0}] * (N - 1)
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
                for i in range(N - 1)
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

        label_issues = lab.get_issues("label")[["is_label_issue", "label_score"]]
        expected_label_issues = pd.DataFrame(
            [{"is_label_issue": False, "label_score": 1.0}] * (N - 2)
            + [
                {"is_label_issue": False, "label_score": 0.0}
            ]  # The confident threshold for this examble is extremely low, but won't flag it as a label issue
            + [{"is_label_issue": False, "label_score": 1.0}]
        )
        pd.testing.assert_frame_equal(label_issues, expected_label_issues)

        underperforming_group_issues = lab.get_issues("underperforming_group")
        expected_underperforming_group_issues = pd.DataFrame(
            [
                {
                    "is_underperforming_group_issue": False,
                    "underperforming_group_score": 1.0,
                }
            ]
            * N  # Only a single data point performs poorly, so it's not in any group
        )
        pd.testing.assert_frame_equal(
            underperforming_group_issues, expected_underperforming_group_issues
        )

        if N > 20:
            # Default threshold for class imbalance is 0.1*1/K, where K is the number of classes. So for 20 examples and 2 classes,
            # the threshold is 5% (so 1 example out of 20 is NOT considered class imbalance)
            class_imbalance_issues = lab.get_issues("class_imbalance")[
                ["is_class_imbalance_issue", "class_imbalance_score"]
            ]
            expected_class_imbalance_issues = pd.DataFrame(
                [
                    {
                        "is_class_imbalance_issue": False,
                        "class_imbalance_score": 1.0,
                    }
                ]
                * (N - 2)
                + [
                    {
                        "is_class_imbalance_issue": True,
                        "class_imbalance_score": 1 / N,
                    }
                ]
                + [
                    {
                        "is_class_imbalance_issue": False,
                        "class_imbalance_score": 1.0,
                    }
                ]
            )
            pd.testing.assert_frame_equal(class_imbalance_issues, expected_class_imbalance_issues)

    @pytest.mark.parametrize(
        "regression_dataset",
        [((N, M)) for N in [11, 20, 50, 100] for M in [3, 4, 6, 8, 10]],
        indirect=["regression_dataset"],
        ids=lambda x: f"N={x[0]}, M={x[1]}",
    )
    def test_regression_issue_detection(self, regression_dataset):
        lab = Datalab(data=regression_dataset, label_name="y", task="regression")
        predictions = np.full(len(y := regression_dataset["y"]), fill_value=y[0])

        lab.find_issues(pred_probs=predictions, features=regression_dataset["X"])

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

        label_issues = lab.get_issues("label")[["is_label_issue", "label_score"]]
        expected_label_issues = pd.DataFrame(
            [{"is_label_issue": False, "label_score": 1.0}] * (len(label_issues) - 1)
            + [
                {"is_label_issue": True, "label_score": 0.0}
            ]  # Expect last example to be flagged as a label issue
        )
        pd.testing.assert_frame_equal(label_issues, expected_label_issues)

    @pytest.mark.parametrize(
        "regression_dataset_with_one_unique_example",
        [((N, M)) for N in [11, 20, 50, 100] for M in [3, 4, 6, 8, 10]],
        indirect=["regression_dataset_with_one_unique_example"],
        ids=lambda x: f"N={x[0]}, M={x[1]}",
    )
    def test_regression_issue_detection_with_one_unique_example(
        self, regression_dataset_with_one_unique_example
    ):
        dataset = regression_dataset_with_one_unique_example
        lab = Datalab(data=dataset, label_name="y", task="regression")
        predictions = np.full(len(y := dataset["y"]), fill_value=y[0])

        lab.find_issues(pred_probs=predictions, features=dataset["X"])

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

        label_issues = lab.get_issues("label")[["is_label_issue", "label_score"]]
        expected_label_issues = pd.DataFrame(
            [{"is_label_issue": False, "label_score": 1.0}] * (len(label_issues) - 2)
            + [
                {"is_label_issue": True, "label_score": 0.0}
            ]  # Expect second to last example to be flagged as a label issue
            + [{"is_label_issue": False, "label_score": 1.0}]
        )
        pd.testing.assert_frame_equal(label_issues, expected_label_issues)

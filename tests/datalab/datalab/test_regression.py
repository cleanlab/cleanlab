# Copyright (C) 2017-2024  Cleanlab Inc.
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


import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors
from cleanlab.datalab.datalab import Datalab


SEED = 42


class TestDatalabForRegression:
    true_columns = [
        "true_label_error_idx",
        "true_outlier_idx",
        "true_duplicate_idx",
        "true_non_iid_idx",
        "true_underperforming_group_idx",
    ]

    @pytest.fixture
    def regression_data(self, num_examples=400, num_features=3, error_frac=0.025, error_noise=0.5):
        np.random.seed(SEED)
        X = np.random.random(size=(num_examples, num_features))
        coefficients = np.random.uniform(-5, 5, size=num_features)
        coefficients[2] *= 0.005  # make the third feature less important

        num_errors = int(num_examples * error_frac)
        label_error_idx = np.random.choice(num_examples, num_errors)

        # Move the examples with label errors closer to the main mass of examples
        direction_towards_center = X[label_error_idx] - 0.5
        direction_towards_center /= np.linalg.norm(direction_towards_center, axis=1)[:, None]
        X[label_error_idx] -= direction_towards_center * 0.2

        # Add near-duplicates
        n_duplicates = 5
        X = np.vstack([X, X[:n_duplicates] + 1e-6 * np.random.randn(n_duplicates, num_features)])
        true_duplicate_ids = np.hstack(
            [np.arange(n_duplicates), np.arange(num_examples, num_examples + n_duplicates)]
        )

        # Add outliers
        n_outliers = 2
        X_outliers = np.random.normal(
            loc=[0.5, 0.5, 1.2], scale=0.05, size=(n_outliers, num_features)
        )
        X = np.vstack([X, X_outliers])
        true_outlier_ids = len(X) - n_outliers + np.arange(n_outliers)

        # Non-iid examples
        # n_non_iid = 15
        # # (linearly increasing features with some Gaussian noise)
        # X_non_iid = np.linspace(0, 0.005, n_non_iid)[:, None] + np.random.normal(
        #     loc=[0.5, 0.5, 0.1], scale=0.005, size=(n_non_iid, num_features)
        # )
        # X = np.vstack([X, X_non_iid])
        # true_non_iid_ids = len(X) - n_non_iid + np.arange(n_non_iid)

        # Underperforming group
        n_underperforming = 10
        # A linearly spaced blob with several targets that are too noisy

        X_underperforming = np.array(
            [
                np.linspace(0, 0.025, n_underperforming),
                np.linspace(0, 0.010, n_underperforming),
                [0] * n_underperforming,
            ]
        ).T + np.random.normal(
            loc=[-0.05, 1.01, 1.01], scale=0.0005, size=(n_underperforming, num_features)
        )
        X = np.vstack([X, X_underperforming])
        true_underperforming_group_ids = len(X) - n_underperforming + np.arange(n_underperforming)

        # Use the underperforming group as the non-iid group
        true_non_iid_ids = np.copy(true_underperforming_group_ids)

        true_y = np.dot(X, coefficients)

        # add extra noisy examples

        label_noise = np.clip(
            np.random.normal(loc=error_noise, scale=error_noise / 4, size=num_errors),
            1.5 * error_noise,
            4 * error_noise,
        ) * np.random.choice([-1, 1], size=num_errors)
        y = true_y.copy()
        y[label_error_idx] += label_noise
        error_idx = np.argsort(abs(y - true_y))[-num_errors:]  # get the noisiest examples idx

        # Add some noise to the underperforming group
        y[true_underperforming_group_ids[-4:]] += np.random.normal(loc=0, scale=1e-6, size=4)

        # Ensure that the MSE is not too large
        from sklearn.linear_model import LinearRegression

        linear_model = LinearRegression()

        # Validate that the label noise affects the MSE for a linear model
        from sklearn.metrics import mean_squared_error

        y_pred_true = linear_model.fit(X, true_y).predict(X)
        mse_true = mean_squared_error(true_y, y_pred_true)
        assert float(mse_true) < 1e-10

        y_pred = linear_model.fit(X, y).predict(X)
        mse = mean_squared_error(y, y_pred)
        assert 1e-3 < float(mse) < 5e-2

        knn_graph = (
            NearestNeighbors(n_neighbors=10, metric="euclidean")
            .fit(X)
            .kneighbors_graph(mode="distance")
        )
        return {
            "X": X,
            "y": y,
            "true_y": true_y,
            "knn_graph": knn_graph,
            "error_idx": error_idx,
            "duplicate_ids": true_duplicate_ids,
            "outlier_ids": true_outlier_ids,
            "non_iid_ids": true_non_iid_ids,
            "underperforming_group_ids": true_underperforming_group_ids,
        }

    @pytest.fixture
    def lab(self, regression_data):
        X, y = regression_data["X"], regression_data["y"]
        test_df = pd.DataFrame(X, columns=["c1", "c2", "c3"])
        test_df["y"] = y
        lab = Datalab(data=test_df, label_name="y", task="regression")
        return lab

    def test_available_issue_types(self, lab):
        assert set(lab.list_default_issue_types()) == set(
            ["label", "outlier", "near_duplicate", "non_iid", "null"]
        )
        assert set(lab.list_possible_issue_types()) == set(
            ["label", "outlier", "near_duplicate", "non_iid", "null", "data_valuation"]
        )

    def test_regression_with_features_finds_label_issues(self, lab, regression_data):
        """Test that the regression issue checks finds 40 label issues, based on the
        numerical features."""
        X = regression_data["X"]
        issue_types = {"label": {"clean_learning_kwargs": {"seed": SEED}, "fine_search_size": 10}}
        lab.find_issues(features=X, issue_types=issue_types)
        lab.report()

        issues = lab.get_issues("label")
        issue_ids = issues.query("is_label_issue").index
        expected_issue_ids = regression_data["error_idx"]

        # jaccard similarity
        intersection = len(list(set(issue_ids).intersection(set(expected_issue_ids))))
        union = len(set(issue_ids)) + len(set(expected_issue_ids)) - intersection
        expected_jaccard_similarity_bound = 0.7
        has_high_jaccard = float(intersection) / union > expected_jaccard_similarity_bound
        if not has_high_jaccard:
            # Then check if MAP@len(expected_issue_ids) is high for the scores
            def average_precision_at_k(y_true, y_pred, k) -> float:
                """Compute average precision at k."""
                y_true = np.asarray(y_true)
                y_pred = np.asarray(y_pred)
                sort_indices = np.argsort(y_pred)[::-1]
                y_true = y_true[sort_indices]
                return float(np.mean(y_true[:k]))

            ap_at_k = average_precision_at_k(
                # y_true are the expected issues
                y_true=np.isin(np.arange(len(issues)), expected_issue_ids),
                # y_pred are the scores
                y_pred=1 - issues["label_score"],
                k=len(expected_issue_ids),
            )
            ap_at_k_bound = 0.9
            ap_at_k_is_high_enough = ap_at_k > ap_at_k_bound
            print(f"AP@{len(expected_issue_ids)}: {ap_at_k}")
            assert (
                ap_at_k_is_high_enough
            ), f"AP@{len(expected_issue_ids)} should be > {ap_at_k_bound}, got {ap_at_k}"

        # FPR
        fpr = len(list(set(issue_ids).difference(set(expected_issue_ids)))) / len(issue_ids)
        expected_fpr_bound = 0.2
        has_low_fpr = fpr < expected_fpr_bound

        if not (has_high_jaccard and has_low_fpr):
            ids_of_union = list(set(issue_ids).union(set(expected_issue_ids)))
            _issues = issues.loc[ids_of_union]
            ids_keys = [
                "error_idx",
                "outlier_ids",
                "duplicate_ids",
                "non_iid_ids",
                "underperforming_group_ids",
            ]
            for new_col, ids_key in zip(self.true_columns, ids_keys):
                _issues[new_col] = _issues.index.isin(regression_data[ids_key])
            # Display dataframe
            print("Expected label issue ids", expected_issue_ids, "\n")
            _label_issues = _issues.query("is_label_issue")
            print(_label_issues["label_score"])

            # Display expected label issues
            print("\nExpected label issues")
            print(
                _issues.assign(true_label=regression_data["true_y"][ids_of_union])
                .query("true_label_error_idx")[
                    [
                        "is_label_issue",
                        "label_score",
                        "given_label",
                        "predicted_label",
                        "true_label",
                    ]
                ]
                .sort_values("label_score")
            )

            for col in self.true_columns:
                print(f"\nColumn: {col}")
                print(_label_issues.query(col)["label_score"])

            error_messages = [
                (
                    has_high_jaccard,
                    f"- Jaccard similarity is too low. Should be > {expected_jaccard_similarity_bound}, got {float(intersection) / union}",
                ),
                (
                    has_low_fpr,
                    f"- FPR is too high. Should be less than or equal to {expected_fpr_bound}, got {fpr}",
                ),
            ]
            error_msg = "The following test(s) failed for the default threshold:\n"
            error_msg += "\n".join(
                [msg for (test_passes, msg) in error_messages if not test_passes]
            )

            raise AssertionError(error_msg)

    def test_regression_with_predictions_finds_label_issues(self, lab, regression_data):
        """Test that the regression issue checks find 9 label issues, based on the
        predictions of a model.

        Instead of running a model, we use the ground-truth to emulate a perfect model's predictions.

        Testing the default behavior, we expect to find some label issues with a given mean score.
        Increasing a threshold for flagging issues will flag more issues, but won't change the score.
        """

        # Use ground-truth to emulate a perfect model's predictions
        y_pred = regression_data["true_y"]
        issue_types = {"label": {}}
        lab.find_issues(pred_probs=y_pred, issue_types=issue_types)
        summary = lab.get_issue_summary()

        issues = lab.get_issues("label")

        true_columns = [
            "true_label_error_idx",
            "true_outlier_idx",
            "true_duplicate_idx",
            "true_non_iid_idx",
            "true_underperforming_group_idx",
        ]
        issue_ids = issues.query("is_label_issue").index
        expected_issue_ids = regression_data["error_idx"]

        # jaccard similarity
        intersection = len(list(set(issue_ids).intersection(set(expected_issue_ids))))
        union = len(set(issue_ids)) + len(set(expected_issue_ids)) - intersection
        expected_jaccard_similarity_bound = 0.8
        has_high_jaccard = float(intersection) / union > expected_jaccard_similarity_bound

        # FPR
        fpr = len(list(set(issue_ids).difference(set(expected_issue_ids)))) / len(issue_ids)
        expected_fpr_bound = 0.0
        has_no_fpr = fpr <= expected_fpr_bound

        if not (has_high_jaccard and has_no_fpr):
            # Work on showing more debugging information if end-to-end test fails
            ids_of_union = list(set(issue_ids).union(set(expected_issue_ids)))
            _issues = issues.loc[ids_of_union]
            new_columns = [
                "true_label_error_idx",
                "true_outlier_idx",
                "true_duplicate_idx",
                "true_non_iid_idx",
                "true_underperforming_group_idx",
            ]
            ids_keys = [
                "error_idx",
                "outlier_ids",
                "duplicate_ids",
                "non_iid_ids",
                "underperforming_group_ids",
            ]
            for new_col, ids_key in zip(new_columns, ids_keys):
                _issues[new_col] = _issues.index.isin(regression_data[ids_key])
            # Display dataframe
            print("Expected label issue ids", expected_issue_ids, "\n")
            _label_issues = _issues.query("is_label_issue")
            print(_label_issues)
            for col in true_columns:
                print(f"\nColumn: {col}")
                print(_label_issues.query(col)["label_score"])

            error_messages = [
                (
                    has_high_jaccard,
                    f"- Jaccard similarity is too low. Should be > {expected_jaccard_similarity_bound}, got {float(intersection) / union}",
                ),
                (
                    has_no_fpr,
                    f"- FPR is too high. Should be less than or equal to {expected_fpr_bound}, got {fpr}",
                ),
            ]
            error_msg = "The following test(s) failed for the default threshold:\n"
            error_msg += "\n".join(
                [msg for (test_passes, msg) in error_messages if not test_passes]
            )
            raise AssertionError(error_msg)

        # Try running with a different threshold
        lab.find_issues(pred_probs=y_pred, issue_types={"label": {"threshold": 0.2}})
        issues = lab.get_issues("label")
        issue_ids = issues.query("is_label_issue").index

        intersection = len(list(set(issue_ids).intersection(set(expected_issue_ids))))
        union = len(set(issue_ids)) + len(set(expected_issue_ids)) - intersection

        different_threshold_jaccard = float(intersection) / union
        assert float(intersection) / union > 0.3

    def test_regression_with_model_and_features_finds_label_issues(self, lab, regression_data):
        """Test that the regression issue checks find label issue with another model."""
        from sklearn.ensemble import RandomForestRegressor, StackingRegressor
        from sklearn.linear_model import RANSACRegressor
        from sklearn.svm import LinearSVR

        # Make an ensemble model of several models
        estimators = [
            ("ransac", RANSACRegressor(random_state=SEED)),
            ("svr", LinearSVR(random_state=SEED)),
        ]
        model = StackingRegressor(
            estimators=estimators, final_estimator=RandomForestRegressor(random_state=SEED)
        )
        X = regression_data["X"]
        issue_types = {"label": {"clean_learning_kwargs": {"model": model, "seed": SEED}}}
        lab.find_issues(features=X, issue_types=issue_types)

        issues = lab.get_issues("label")
        issue_ids = issues.query("is_label_issue").index
        expected_issue_ids = regression_data[
            "error_idx"
        ]  # Set to 5% of the data, but random noise may be too small to detect

        # jaccard similarity
        intersection = len(list(set(issue_ids).intersection(set(expected_issue_ids))))
        union = len(set(issue_ids)) + len(set(expected_issue_ids)) - intersection
        expected_jaccard_similarity_bound = 0.3
        has_high_jaccard = float(intersection) / union >= expected_jaccard_similarity_bound

        # FPR
        fpr = len(list(set(issue_ids).difference(set(expected_issue_ids)))) / len(issue_ids)
        expected_fpr_bound = 0.3
        has_low_fpr = fpr <= expected_fpr_bound

        if not (has_high_jaccard and has_low_fpr):
            ids_of_union = list(set(issue_ids).union(set(expected_issue_ids)))
            _issues = issues.loc[ids_of_union]
            ids_keys = [
                "error_idx",
                "outlier_ids",
                "duplicate_ids",
                "non_iid_ids",
                "underperforming_group_ids",
            ]
            for new_col, ids_key in zip(self.true_columns, ids_keys):
                _issues[new_col] = _issues.index.isin(regression_data[ids_key])
            # Display dataframe
            print("Expected label issue ids", expected_issue_ids, "\n")
            _label_issues = _issues.query("is_label_issue")
            print(_label_issues)

            for col in self.true_columns:
                print(f"\nColumn: {col}")
                print(_label_issues.query(col)["label_score"])

            error_messages = [
                (
                    has_high_jaccard,
                    f"- Jaccard similarity is too low. Should be > {expected_jaccard_similarity_bound}, got {float(intersection) / union}",
                ),
                (
                    has_low_fpr,
                    f"- FPR is too high. Should be less than or equal to {expected_fpr_bound}, got {fpr}",
                ),
            ]
            error_msg = "The following test(s) failed for the default threshold:\n"
            error_msg += "\n".join(
                [msg for (test_passes, msg) in error_messages if not test_passes]
            )
            raise AssertionError(error_msg)

    @pytest.mark.parametrize(
        "argument_name, data_key, expected_issue_types_in_summary",
        [
            ("features", "X", set(["label", "outlier", "near_duplicate", "non_iid", "null"])),
            ("pred_probs", "true_y", set(["label"])),
            ("knn_graph", "knn_graph", set(["outlier", "near_duplicate", "non_iid"])),
        ],
        ids=["features only", "pred_probs only", "knn_graph only"],
    )
    def test_find_issues_defaults(
        self, lab, regression_data, argument_name, data_key, expected_issue_types_in_summary
    ):
        """Test that the regression issue checks find various issues with the default settings."""
        input_dict = {argument_name: regression_data[data_key]}
        lab.find_issues(**input_dict)
        issue_summary = lab.get_issue_summary()
        assert issue_summary["num_issues"].sum() > 0
        assert set(issue_summary["issue_type"].values) == expected_issue_types_in_summary

    @pytest.mark.parametrize(
        "argument_name, data_key",
        [
            ("features", "X"),
            ("knn_graph", "knn_graph"),
        ],
        ids=["features only", "knn_graph only"],
    )
    def test_find_outliers(self, lab, regression_data, argument_name, data_key):
        """Test that the regression issue checks find 2 outlier issues."""

        input_dict = {argument_name: regression_data[data_key]}
        # Other tests are too sensitive to having more obvious outliers
        issue_types = {
            "outlier": {"threshold": 0.20}
        }  # Lower threshold for a more conservative result
        lab.find_issues(**input_dict, issue_types=issue_types)
        lab.report()

        issues = lab.get_issues("outlier")
        issue_ids = issues.query("is_outlier_issue").index
        expected_issue_ids = regression_data["outlier_ids"]

        # jaccard similarity
        intersection = len(list(set(issue_ids).intersection(set(expected_issue_ids))))
        union = len(set(issue_ids)) + len(set(expected_issue_ids)) - intersection
        assert float(intersection) / union >= 0.5

        # FPR
        fpr = len(list(set(issue_ids).difference(set(expected_issue_ids)))) / len(issue_ids)
        assert fpr == 0.0

    @pytest.mark.parametrize(
        "argument_name, data_key",
        [
            ("features", "X"),
            ("knn_graph", "knn_graph"),
        ],
        ids=["features only", "knn_graph only"],
    )
    def test_find_near_duplicates(self, lab, regression_data, argument_name, data_key):
        """Test that the regression issue checks find 5 near-duplicate issues."""

        input_dict = {argument_name: regression_data[data_key]}
        issue_types = {"near_duplicate": {"threshold": 0.01}}
        lab.find_issues(**input_dict, issue_types=issue_types)

        issues = lab.get_issues("near_duplicate")
        issue_ids = issues.query("is_near_duplicate_issue").index
        expected_issue_ids = regression_data["duplicate_ids"]

        # jaccard similarity
        intersection = len(list(set(issue_ids).intersection(set(expected_issue_ids))))
        union = len(set(issue_ids)) + len(set(expected_issue_ids)) - intersection
        expected_jaccard_similarity_bound = 0.8
        has_high_jaccard = float(intersection) / union > expected_jaccard_similarity_bound

        # FPR
        fpr = len(list(set(issue_ids).difference(set(expected_issue_ids)))) / len(issue_ids)
        expected_fpr_bound = 0.2
        has_low_fpr = fpr <= expected_fpr_bound

        if not (has_high_jaccard and has_low_fpr):
            ids_of_union = list(set(issue_ids).union(set(expected_issue_ids)))
            _issues = issues.loc[ids_of_union]

            ids_keys = [
                "error_idx",
                "outlier_ids",
                "duplicate_ids",
                "non_iid_ids",
                "underperforming_group_ids",
            ]
            for new_col, ids_key in zip(self.true_columns, ids_keys):
                _issues[new_col] = _issues.index.isin(regression_data[ids_key])
            _near_duplicate_issues = _issues.query("is_near_duplicate_issue")
            print(_near_duplicate_issues["near_duplicate_sets"])

            for col in self.true_columns:
                print(f"\nColumn: {col}")
                print(_near_duplicate_issues.query(col)["near_duplicate_sets"])

            error_messages = [
                (
                    has_high_jaccard,
                    f"- Jaccard similarity is too low. Should be > {expected_jaccard_similarity_bound}, got {float(intersection) / union}",
                ),
                (
                    has_low_fpr,
                    f"- FPR is too high. Should be less than or equal to {expected_fpr_bound}, got {fpr}",
                ),
            ]
            error_msg = "The following test(s) failed for the default threshold:\n"
            error_msg += "\n".join(
                [msg for (test_passes, msg) in error_messages if not test_passes]
            )

            raise AssertionError(error_msg)

    @pytest.mark.parametrize(
        "argument_name, data_key",
        [
            ("features", "X"),
            ("knn_graph", "knn_graph"),
        ],
        ids=["features only", "knn_graph only"],
    )
    def test_find_non_iid(self, lab, regression_data, argument_name, data_key):
        """Test that the regression issue checks find 30 non-iid issues."""

        input_dict = {argument_name: regression_data[data_key]}
        lab.find_issues(
            **input_dict,
            issue_types={
                "non_iid": {"num_permutations": 1000, "seed": SEED, "significance_threshold": 0.3}
            },
        )

        issues = lab.get_issues("non_iid")
        issue_ids = issues.query("is_non_iid_issue").index
        expected_issue_ids = regression_data["non_iid_ids"]

        # omitting jaccard similarity and FPR for non-iid issues

        # FPR
        fp = list(set(issue_ids).difference(set(expected_issue_ids)))
        fpr = len(fp) / len(issue_ids) if len(issue_ids) > 0 else 0.0
        expected_fpr_bound = 0.0
        has_low_fpr = fpr <= expected_fpr_bound

        # AP@len(expected_issue_ids)
        def average_precision_at_k(y_true, y_pred, k) -> float:
            """Compute average precision at k."""
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            sort_indices = np.argsort(y_pred)[::-1]
            y_true = y_true[sort_indices]
            return float(np.mean(y_true[:k]))

        ap_at_k = average_precision_at_k(
            # y_true are the expected issues
            y_true=np.isin(np.arange(len(issues)), expected_issue_ids),
            # y_pred are the scores
            y_pred=1 - issues["non_iid_score"],
            k=len(expected_issue_ids),
        )
        expected_ap_at_k_bound = 0.8
        has_high_enough_ap_at_k = ap_at_k >= expected_ap_at_k_bound

        if not (has_low_fpr and has_high_enough_ap_at_k):
            ids_of_union = list(set(issue_ids).union(set(expected_issue_ids)))
            _issues = issues.loc[ids_of_union]
            ids_keys = [
                "error_idx",
                "outlier_ids",
                "duplicate_ids",
                "non_iid_ids",
                "underperforming_group_ids",
            ]
            for new_col, ids_key in zip(self.true_columns, ids_keys):
                _issues[new_col] = _issues.index.isin(regression_data[ids_key])
            _non_iid_issues = _issues.query("is_non_iid_issue")
            print(
                _issues[["is_non_iid_issue", "non_iid_score"]]
                .sort_values("non_iid_score")
                .head(len(expected_issue_ids))
            )

            for col in self.true_columns:
                print(f"\nColumn: {col}")
                # print(_non_iid_issues.query(col)["non_iid_sets"])

            error_messages = [
                (
                    has_low_fpr,
                    f"- FPR is too high. Should be less than or equal to {expected_fpr_bound}, got {fpr}",
                ),
                (
                    has_high_enough_ap_at_k,
                    f"- AP@{len(expected_issue_ids)} should be greater than or equal to {expected_ap_at_k_bound}, got {ap_at_k}",
                ),
            ]
            error_msg = "The following test(s) failed for the default threshold:\n"
            error_msg += "\n".join(
                [msg for (test_passes, msg) in error_messages if not test_passes]
            )

            raise AssertionError(error_msg)

    def test_find_null_issues(self, lab, regression_data):
        """Test that the regression issue checks find 0 null issues."""
        X = regression_data["X"]
        lab.find_issues(features=X, issue_types={"null": {}})
        summary = lab.get_issue_summary("null")
        assert summary["num_issues"].values[0] == 0

        rand_ids = np.random.choice(X.shape[0], 10, replace=False)
        for i in rand_ids:
            j = np.random.choice(X.shape[1], 1, replace=False)
            X[i, j] = np.nan

        rand_ids_full = np.random.choice(X.shape[0], 3, replace=False)
        for i in rand_ids_full:
            X[i, :] = np.nan
        lab.find_issues(features=X, issue_types={"null": {}})
        issues = lab.get_issues("null")
        null_issues = issues.query("is_null_issue")
        assert set(rand_ids_full) == set(null_issues.index.tolist())

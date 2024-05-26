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

from itertools import islice

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression

from cleanlab.datalab.datalab import Datalab
from cleanlab.experimental.datalab.data_monitor import DataMonitor
from cleanlab.benchmarking.noise_generation import (
    generate_noise_matrix_from_trace,
    generate_noisy_labels,
)

SEED = 42


class SetupClass:
    num_examples = 2000
    test_size = 0.1

    @pytest.fixture
    def data(self):
        np.random.seed(SEED)

        BINS = {
            0: [-np.inf, 3.3],
            1: [3.3, 6.6],
            2: [6.6, +np.inf],
        }

        X = np.random.rand(self.num_examples, 2) * 5
        f = np.sum(X, axis=1)
        y = np.array([k for f_i in f for k, v in BINS.items() if v[0] <= f_i < v[1]])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=SEED
        )

        py = np.bincount(y) / float(len(y))
        m = len(BINS)

        noise_matrix = generate_noise_matrix_from_trace(
            m,
            trace=0.9 * m,
            py=py,
            valid_noise_matrix=True,
            seed=SEED,
        )

        noisy_labels_train = generate_noisy_labels(y_train, noise_matrix)
        noisy_labels_test = generate_noisy_labels(y_test, noise_matrix)

        return {
            "X_train": X_train,
            "noisy_labels_train": noisy_labels_train,
            "X_test": X_test,
            "noisy_labels_test": noisy_labels_test,
        }

    @pytest.fixture
    def pred_probs_train(self, data):
        model = LogisticRegression()
        pred_probs = cross_val_predict(
            estimator=model,
            X=data["X_train"],
            y=data["noisy_labels_train"],
            cv=5,
            method="predict_proba",
        )
        return pred_probs

    @pytest.fixture
    def pred_probs_test(self, data):
        model = LogisticRegression()
        model.fit(data["X_train"], data["noisy_labels_train"])
        pred_probs = model.predict_proba(data["X_test"])
        return pred_probs

    @pytest.fixture
    def pred_probs_combined(self, data):
        model = LogisticRegression()
        pred_probs = cross_val_predict(
            estimator=model,
            X=np.concatenate((data["X_test"], data["X_train"])),
            y=np.concatenate((data["noisy_labels_test"], data["noisy_labels_train"])),
            cv=5,
            method="predict_proba",
        )
        return pred_probs

    @pytest.fixture
    def datalab(self, pred_probs_train, data):
        dataset = {"labels": data["noisy_labels_train"]}
        lab = Datalab(data=dataset, label_name="labels")
        issue_types = {"label": {}, "outlier": {}}
        lab.find_issues(
            pred_probs=pred_probs_train, features=data["X_train"], issue_types=issue_types
        )
        return lab


class TestDataMonitorReuseStatisticInfo(SetupClass):
    # TODO: Rename class

    def test_reuse_statistics_info(self, datalab):
        # TODO: Rename test
        """Test that the DataMonitor has the same info as the Datalab instance."""
        monitor = DataMonitor(datalab=datalab)
        for k in datalab.get_info().keys():
            assert monitor.info[k].keys() == datalab.get_info(k).keys()

    def test_set_trained_statistics(self, datalab, pred_probs_test, data):
        # TODO: Rename test
        """Check the label-specific info is available in the DataMonitor instance for detecting label issues.
        The info should not be affected by the test data."""
        monitor = DataMonitor(datalab=datalab)
        monitor.find_issues(
            labels=data["noisy_labels_test"], pred_probs=pred_probs_test, features=data["X_test"]
        )
        trained_statistics = datalab.get_info("label")
        test_statistics = monitor.info["label"]
        for k, v in trained_statistics.items():
            if k in ["confident_joint", "confident_thresholds"]:
                assert np.array_equal(v, test_statistics[k])

    def test_find_issues_with_datalab(self, datalab, pred_probs_test, pred_probs_combined, data):
        """Verify that running Datalab on a train+test dataset should give very similar results
        to running DataMonitor on the test dataset (with a Datalab instance having only seen the training dataset).
        """
        # Run issue checks on combined dataset
        combined_data = {
            "labels": np.concatenate((data["noisy_labels_test"], data["noisy_labels_train"]))
        }
        lab_all = Datalab(data=combined_data, label_name="labels")
        lab_all.find_issues(pred_probs=pred_probs_combined, issue_types={"label": {}})

        # Monitor test data only
        test_data = {"labels": data["noisy_labels_test"]}
        monitor = DataMonitor(datalab=datalab)
        monitor.find_issues(
            labels=test_data["labels"], pred_probs=pred_probs_test, features=data["X_test"]
        )

        # Compare results
        lab_results = lab_all.get_issues()[: int(self.num_examples * self.test_size)]
        monitor_results = monitor.issues
        similarity = sum(
            lab_results[["is_label_issue"]].values == monitor_results[["is_label_issue"]].values
        ) / len(lab_results)
        assert similarity >= 0.93

    def test_find_issues_with_datalab_multi_different_size_batch(
        self, datalab, pred_probs_test, pred_probs_combined, data
    ):
        """Test that the DataMonitor can find label issues in test data over smaller batches,
        perfomring similarly to Datalab on the combined dataset.
        """
        # Run issue checks on combined dataset and fetch results
        combined_data = {
            "labels": np.concatenate((data["noisy_labels_test"], data["noisy_labels_train"]))
        }
        lab_all = Datalab(data=combined_data, label_name="labels")
        lab_all.find_issues(pred_probs=pred_probs_combined, issue_types={"label": {}})
        lab_results = lab_all.get_issues()[: int(self.num_examples * self.test_size)]

        # Monitor test data in smaller batches, collecting results for each batch
        monitor_results = pd.DataFrame()
        batch_sizes = [20, 50, 100, 30]
        start_positions = [0] * len(batch_sizes)
        for i in range(1, len(batch_sizes)):
            start_positions[i] = start_positions[i - 1] + batch_sizes[i - 1]
        for batch_size, start_position in zip(batch_sizes, start_positions):
            data_batch = {
                "labels": data["noisy_labels_test"][start_position : start_position + batch_size],
                "pred_probs": pred_probs_test[start_position : start_position + batch_size],
                "features": data["X_test"][start_position : start_position + batch_size],
            }
            monitor = DataMonitor(datalab=datalab)
            monitor.find_issues(**data_batch)
            monitor_results = pd.concat([monitor_results, monitor.issues], axis=0)

        # Compare results
        similarity = sum(
            lab_results[["is_label_issue"]].values == monitor_results[["is_label_issue"]].values
        ) / len(lab_results)
        assert similarity >= 0.90

    def test_default_issue_types(self, datalab, data, pred_probs_test):
        """Test that the DataMonitor checks for the correct types of issues by default."""
        # TODO: Run this test with features as well
        # Check for issues on test data
        dataset = {"labels": data["noisy_labels_test"]}
        monitor = DataMonitor(datalab=datalab)
        monitor.find_issues(
            labels=dataset["labels"], pred_probs=pred_probs_test, features=data["X_test"]
        )

        expected_issue_types_keys = ["label", "outlier"]

        # Verify that the correct types of issues were checked for
        issue_summary = monitor.issue_summary
        issue_types_found = issue_summary["issue_type"]
        assert set(issue_types_found) == set(expected_issue_types_keys)


# Helper function for testing streaming workflows
def batch_slices(iterable, start, batch_size, drop_last=False):
    """
    Generator that yields slices of the specified batch size from the iterable starting from the start index.
    If drop_last is True, the last batch will be dropped if it is smaller than the batch size.
    """
    it = iter(iterable)
    # Skip items until the start index
    skipped = list(islice(it, start))
    while True:
        batch = list(islice(it, batch_size))
        if not batch or (drop_last and len(batch) < batch_size):
            break
        yield np.array(batch)


# Test class for streaming data
class TestDataMonitorInit(SetupClass):
    # This test class looks at the initialization of the DataMonitor class
    def test_data_monitor_creation(self, datalab, data):
        data = {"labels": data["noisy_labels_test"]}
        monitor = DataMonitor(datalab=datalab)
        assert isinstance(monitor, DataMonitor)

    def test_data_monitor_with_streaming_data(self, datalab, data):
        """Test that the issues DataFrame in the DataMonitor is updated after streaming data in batches.
        Also verify that the issues DataFrame has the expected values.
        """
        monitor = DataMonitor(datalab=datalab)
        assert isinstance(monitor, DataMonitor)

        features = data["X_test"][:20]
        labels = data["noisy_labels_test"][:20]
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression()
        clf.fit(data["X_train"], data["noisy_labels_train"])
        pred_probs = clf.predict_proba(features)

        singleton_stream = (
            {
                "features": f[np.newaxis, :],
                "pred_probs": p[np.newaxis, :],
                "labels": l[np.newaxis],
            }
            for f, p, l in zip(features[:5], pred_probs[:5], labels[:5])
        )

        batch_stream = (
            {
                "features": f,
                "pred_probs": p,
                "labels": l,
            }
            for f, p, l in zip(
                batch_slices(features[5:], 0, 5),
                batch_slices(pred_probs[5:], 0, 5),
                batch_slices(labels[5:], 0, 5),
            )
        )

        assert monitor.issues.empty

        for i, eg in enumerate(singleton_stream):
            monitor.find_issues(**eg)

        for i, batch in enumerate(batch_stream):
            monitor.find_issues(**batch)
        issues = monitor.issues
        assert (not issues.empty) and (len(issues) == len(labels))
        assert list(issues["is_label_issue"]) == [False] * (len(labels) - 1) + [True]
        expected_ids_sorted_by_label_score = [
            19,
            15,
            3,
            9,
            8,
            14,
            17,
            7,
            11,
            0,
            10,
            2,
            1,
            6,
            12,
            5,
            4,
            18,
            16,
            13,
        ]

        assert all(issues["label_score"].iloc[:-1] > 0.3) and issues["label_score"].iloc[-1] < 0.1

        # Test that there's a high correlation between the expected order of the examples and the order of the examples in the issues dataframe
        from scipy.stats import spearmanr

        corr, _ = spearmanr(
            expected_ids_sorted_by_label_score, list(issues.sort_values("label_score").index)
        )
        assert corr > 0.9

    def test_outlier_detection(self, datalab, data):
        monitor = DataMonitor(datalab=datalab)
        assert isinstance(monitor, DataMonitor)

        features = data["X_test"][:20]

        # Make the first example an outlier
        features[0] += 5

        labels = data["noisy_labels_test"][:20]

        # Correct the label of the outlier
        labels[0] = np.where(features[0].sum() < 3.3, 0, np.where(features[0].sum() < 6.6, 1, 2))
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression()
        clf.fit(data["X_train"], data["noisy_labels_train"])
        pred_probs = clf.predict_proba(features)

        singleton_stream = (
            {
                "features": f[np.newaxis, :],
                "pred_probs": p[np.newaxis, :],
                "labels": l[np.newaxis],
            }
            for f, p, l in zip(features[:5], pred_probs[:5], labels[:5])
        )

        batch_stream = (
            {
                "features": f,
                "pred_probs": p,
                "labels": l,
            }
            for f, p, l in zip(
                batch_slices(features[5:], 0, 5),
                batch_slices(pred_probs[5:], 0, 5),
                batch_slices(labels[5:], 0, 5),
            )
        )

        for i, eg in enumerate(singleton_stream):
            monitor.find_issues(**eg)

        issues = monitor.issues
        outlier_issue_mask = issues["is_outlier_issue"].to_numpy()
        expected_mask = np.array([True, False, False, False, False])
        np.testing.assert_array_equal(outlier_issue_mask, expected_mask)

        outlier_scores = issues["outlier_score"].to_numpy()
        expected_scores = np.array([6.4e-15, 0.38, 0.33, 0.42, 0.33])
        np.testing.assert_allclose(outlier_scores, expected_scores, atol=1e-2)

    def test_only_on_pred_probs(self, data):
        train_dataset = {"labels": data["noisy_labels_train"], "X_train": data["X_train"]}
        lab = Datalab(data=train_dataset, label_name="labels", task="classification")

        from sklearn.model_selection import cross_val_predict
        from sklearn.linear_model import LogisticRegression

        pred_probs = cross_val_predict(
            LogisticRegression(),
            train_dataset["X_train"],
            train_dataset["labels"],
            cv=5,
            method="predict_proba",
        )

        lab.find_issues(pred_probs=pred_probs)

        # Set up monitor
        monitor = DataMonitor(datalab=lab)

        # Set up stream of test data
        features = data["X_test"][:20]
        labels = data["noisy_labels_test"][:20]

        clf = LogisticRegression()
        clf.fit(train_dataset["X_train"], train_dataset["labels"])

        pred_probs = clf.predict_proba(features)

        singleton_stream = (
            {
                "pred_probs": clf.predict_proba(f[np.newaxis, :]),
                "labels": l[np.newaxis],
            }
            for f, l in zip(features, labels)
        )

        for eg in singleton_stream:
            monitor.find_issues(**eg)

        issues = monitor.issues

        # Only the "label" monitor is configured
        assert set(["label"]) == set(monitor.monitors.keys())

        # Only label issues should have been checked
        assert set(issues.columns) == set(["is_label_issue", "label_score"])
        # All the "test" examples should been checked
        assert len(issues) == len(features)

    def test_only_on_features(self, data):
        train_dataset = {"X_train": data["X_train"]}
        lab = Datalab(data=train_dataset, task="classification")

        lab.find_issues(features=train_dataset["X_train"])

        # Set up monitor
        monitor = DataMonitor(datalab=lab)

        # Set up stream of test data
        features = data["X_test"][:20]

        singleton_stream = ({"features": f[np.newaxis, :]} for f in features)

        for eg in singleton_stream:
            monitor.find_issues(**eg)

        issues = monitor.issues

        # Only the "outlier" monitor is configured
        assert set(["outlier"]) == set(monitor.monitors.keys())

        # Only outlier issues should have been checked
        assert set(issues.columns) == set(["is_outlier_issue", "outlier_score"])
        # All the "test" examples should been checked
        assert len(issues) == len(features)

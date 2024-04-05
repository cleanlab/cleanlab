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
from datasets import Dataset
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
        data = {"labels": data["noisy_labels_train"]}
        lab = Datalab(data=data, label_name="labels")
        lab.find_issues(pred_probs=pred_probs_train, issue_types={"label": {}})
        return lab


class TestDataMonitorReuseStatisticInfo(SetupClass):

    def test_reuse_statistics_info(self, datalab, data):
        data = {"labels": data["noisy_labels_test"]}
        monitor = DataMonitor(datalab=datalab)
        for k in datalab.get_info().keys():
            assert monitor.info[k].keys() == datalab.get_info(k).keys()

    def test_set_trained_statistics(self, datalab, pred_probs_test, data):
        monitor = DataMonitor(datalab=datalab)
        monitor.find_issues(labels=data["noisy_labels_test"], pred_probs=pred_probs_test)
        trained_statistics = datalab.get_info("label")
        test_statistics = monitor.info["label"]
        for k, v in trained_statistics.items():
            if k in ["confident_joint", "confident_thresholds"]:
                assert np.array_equal(v, test_statistics[k])

    def test_find_issues_with_datalab(self, datalab, pred_probs_test, pred_probs_combined, data):
        combined_data = {
            "labels": np.concatenate((data["noisy_labels_test"], data["noisy_labels_train"]))
        }
        lab_all = Datalab(data=combined_data, label_name="labels")
        lab_all.find_issues(pred_probs=pred_probs_combined, issue_types={"label": {}})
        test_data = {"labels": data["noisy_labels_test"]}
        monitor = DataMonitor(datalab=datalab)
        monitor.find_issues(labels=test_data["labels"], pred_probs=pred_probs_test)
        lab1_result = lab_all.get_issues()[: int(self.num_examples * self.test_size)]
        lab2_result = monitor.issues
        similarity = sum(
            lab1_result[["is_label_issue"]].values == lab2_result[["is_label_issue"]].values
        ) / len(lab1_result)
        assert similarity >= 0.93

    def test_find_issues_with_datalab_multi_different_size_batch(
        self, datalab, pred_probs_test, pred_probs_combined, data
    ):
        combined_data = {
            "labels": np.concatenate((data["noisy_labels_test"], data["noisy_labels_train"]))
        }
        lab_all = Datalab(data=combined_data, label_name="labels")
        lab_all.find_issues(pred_probs=pred_probs_combined, issue_types={"label": {}})
        lab1_result = lab_all.get_issues()[: int(self.num_examples * self.test_size)]
        lab2_result = pd.DataFrame()
        batch_sizes = [20, 50, 100, 30]
        start_positions = [0] * len(batch_sizes)
        for i in range(1, len(batch_sizes)):
            start_positions[i] = start_positions[i - 1] + batch_sizes[i - 1]
        for batch_size, start_position in zip(batch_sizes, start_positions):
            test_data = {
                "labels": data["noisy_labels_test"][start_position : start_position + batch_size]
            }
            monitor = DataMonitor(datalab=datalab)
            monitor.find_issues(
                labels=test_data["labels"],
                pred_probs=pred_probs_test[start_position : start_position + batch_size],
            )
            lab2_result = pd.concat([lab2_result, monitor.issues], axis=0)
        similarity = sum(
            lab1_result[["is_label_issue"]].values == lab2_result[["is_label_issue"]].values
        ) / len(lab1_result)
        assert similarity >= 0.90

    def test_default_issue_types(self, datalab, data, pred_probs_test):
        dataset = {"labels": data["noisy_labels_test"]}
        monitor = DataMonitor(datalab=datalab)
        monitor.find_issues(labels=dataset["labels"], pred_probs=pred_probs_test)

        expected_issue_types_keys = ["label"]

        issue_summary = monitor.issue_summary
        issue_types_found = issue_summary["issue_type"]
        assert set(issue_types_found) == set(expected_issue_types_keys)


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


class TestDataMonitorInit(SetupClass):
    # This test class looks at the initialization of the DataMonitor class
    def test_data_monitor_creation(self, datalab, data):
        data = {"labels": data["noisy_labels_test"]}
        monitor = DataMonitor(datalab=datalab)
        assert isinstance(monitor, DataMonitor)

    def test_data_monitor_with_streaming_data(self, datalab, data):
        monitor = DataMonitor(datalab=datalab)
        assert isinstance(monitor, DataMonitor)

        # The size of the dataset should be 0
        # assert len(monitor.data) == 0

        features = data["X_test"][:20]
        labels = data["noisy_labels_test"][:20]
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression()
        clf.fit(data["X_train"], data["noisy_labels_train"])
        pred_probs = clf.predict_proba(features)

        singleton_stream = (
            {
                # "features": f[np.newaxis, :],  # TODO: Support features in the future
                "pred_probs": p[np.newaxis, :],
                "labels": l[np.newaxis],
            }
            for f, p, l in zip(features[:5], pred_probs[:5], labels[:5])
        )

        batch_stream = (
            {
                # "features": f,  # TODO: Support features in the future
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

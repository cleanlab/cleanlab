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
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression

from cleanlab.datalab.datalab import Datalab
from cleanlab.experimental.datalab.TestDatalab import TestDatalab
from cleanlab.benchmarking.noise_generation import (
    generate_noise_matrix_from_trace,
    generate_noisy_labels,
)

SEED = 42


class TestDataLabReuseStatisticInfo:
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
    def trained_datalab(self, pred_probs_train, data):
        data = {"labels": data["noisy_labels_train"]}
        lab = Datalab(data=data, label_name="labels")
        lab.find_issues(pred_probs=pred_probs_train, issue_types={"label": {}})
        return lab

    def test_reuse_statistics_info(self, trained_datalab, data):
        data = {"labels": data["noisy_labels_test"]}
        lab = TestDatalab(trained_datalab=trained_datalab, data=data, label_name="labels")
        for k in trained_datalab.get_info().keys():
            assert lab.get_info(k).keys() == trained_datalab.get_info(k).keys()

    def test_set_trained_statistics(self, trained_datalab, pred_probs_test, data):
        data = {"labels": data["noisy_labels_test"]}
        lab = TestDatalab(trained_datalab=trained_datalab, data=data, label_name="labels")
        lab.find_issues(pred_probs=pred_probs_test, issue_types={"label": {}})
        trained_statistics = trained_datalab.get_info("label")
        test_statistics = lab.get_info("label")
        for k, v in trained_statistics.items():
            if k in ["confident_joint"]:
                assert np.array_equal(v, test_statistics[k])

    def test_find_issue_with_trained_datalab(
        self, trained_datalab, pred_probs_test, pred_probs_combined, data
    ):
        combined_data = {
            "labels": np.concatenate((data["noisy_labels_test"], data["noisy_labels_train"]))
        }
        lab_all = Datalab(data=combined_data, label_name="labels")
        lab_all.find_issues(pred_probs=pred_probs_combined, issue_types={"label": {}})
        test_data = {"labels": data["noisy_labels_test"]}
        lab_test = TestDatalab(trained_datalab=trained_datalab, data=test_data, label_name="labels")
        lab_test.find_issues(pred_probs=pred_probs_test, issue_types={"label": {}})
        lab1_result = lab_all.get_issues()[: int(self.num_examples * self.test_size)]
        lab2_result = lab_test.get_issues()
        similarity = sum(
            lab1_result[["is_label_issue"]].values == lab2_result[["is_label_issue"]].values
        ) / len(lab1_result)
        assert similarity >= 0.93

    def test_find_issue_with_trained_datalab_multi_different_size_batch(
        self, trained_datalab, pred_probs_test, pred_probs_combined, data
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
            lab_test = TestDatalab(
                trained_datalab=trained_datalab, data=test_data, label_name="labels"
            )
            lab_test.find_issues(
                pred_probs=pred_probs_test[start_position : start_position + batch_size],
                issue_types={"label": {}},
            )
            lab2_result = pd.concat([lab2_result, lab_test.get_issues()], axis=0)
        similarity = sum(
            lab1_result[["is_label_issue"]].values == lab2_result[["is_label_issue"]].values
        ) / len(lab1_result)
        assert similarity >= 0.90

    def test_default_issue_types(self, trained_datalab, data, pred_probs_test):
        dataset = {"labels": data["noisy_labels_test"]}
        features = data["X_test"]
        lab = TestDatalab(trained_datalab=trained_datalab, data=dataset, label_name="labels")
        lab.find_issues(pred_probs=pred_probs_test, features=features)

        expected_issue_types_keys = ["label", "null", "class_imbalance"]

        issue_summary = lab.get_issue_summary()
        issue_types_found = issue_summary["issue_type"]
        assert set(issue_types_found) == set(expected_issue_types_keys)

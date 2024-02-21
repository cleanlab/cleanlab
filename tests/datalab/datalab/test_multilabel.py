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
import pytest
from sklearn.neighbors import NearestNeighbors


from cleanlab.datalab.datalab import Datalab
from cleanlab.internal.multilabel_utils import int2onehot


from cleanlab.benchmarking.noise_generation import (
    generate_noise_matrix_from_trace,
    generate_noisy_labels,
)


class TestDatalabForMultilabelClassification:
    @pytest.fixture
    def multilabel_data(
        self,
        means=[[-5, 3.5], [0, 2], [-3, 6]],
        covs=[[[3, -1.5], [-1.5, 1]], [[5, -1.5], [-1.5, 1]], [[3, -1.5], [-1.5, 1]]],
        boxes_coordinates=[[-3.5, 0, -1.5, 1.7], [-1, 3, 2, 4], [-5, 2, -3, 4], [-3, 2, -1, 4]],
        box_multilabels=[[0, 1], [1, 2], [0, 2], [0, 1, 2]],
        sizes=[100, 80, 100],
        avg_trace=0.9,
        seed=5,
    ):
        np.random.seed(seed=seed)
        n = sum(sizes)
        num_classes = len(means)
        m = num_classes + len(
            box_multilabels
        )  # number of classes by treating each multilabel as 1 unique label
        local_data = []
        labels = []
        for i in range(0, len(means)):
            local_data.append(
                np.random.multivariate_normal(mean=means[i], cov=covs[i], size=sizes[i])
            )
            labels += [[i]] * sizes[i]

        def make_multi(X, Y, bx1, by1, bx2, by2, label_list):
            ll = np.array([bx1, by1])  # lower-left
            ur = np.array([bx2, by2])  # upper-right

            inidx = np.all(np.logical_and(X.tolist() >= ll, X.tolist() <= ur), axis=1)
            for i in range(0, len(Y)):
                if inidx[i]:
                    Y[i] = label_list
            return Y

        X_train = np.vstack(local_data)

        for i in range(0, len(box_multilabels)):
            bx1, by1, bx2, by2 = boxes_coordinates[i]
            multi_label = box_multilabels[i]
            labels = make_multi(X_train, labels, bx1, by1, bx2, by2, multi_label)

        d = {}
        for i in labels:
            if str(i) not in d:
                d[str(i)] = len(d)
        inv_d = {v: k for k, v in d.items()}
        labels_idx = [d[str(i)] for i in labels]
        py = np.bincount(labels_idx) / float(len(labels_idx))
        noise_matrix = generate_noise_matrix_from_trace(
            m,
            trace=avg_trace * m,
            py=py,
            valid_noise_matrix=True,
            seed=seed,
        )
        noisy_labels_idx = generate_noisy_labels(labels_idx, noise_matrix)
        noisy_labels = [eval(inv_d[i]) for i in noisy_labels_idx]
        error_idx = np.where(labels_idx != noisy_labels_idx)[0]
        pred_probs = np.full((n, num_classes), fill_value=0.1)
        labels_onehot = int2onehot(labels, K=num_classes)
        pred_probs[labels_onehot == 1] = 0.9

        knn_graph = (
            NearestNeighbors(n_neighbors=15, metric="euclidean")
            .fit(X_train)
            .kneighbors_graph(mode="distance")
        )
        return {
            "X": X_train,
            "true_y": labels,
            "y": noisy_labels,
            "error_idx": error_idx,
            "pred_probs": pred_probs,
            "knn_graph": knn_graph,
        }

    @pytest.fixture
    def lab(self, multilabel_data):
        X, y = multilabel_data["X"], multilabel_data["y"]
        data = {"X": X, "y": y}
        lab = Datalab(data=data, label_name="y", task="multilabel")
        return lab

    def test_available_issue_types(self, lab):
        assert set(lab.list_default_issue_types()) == set(
            ["label", "near_duplicate", "non_iid", "outlier", "null"]
        )
        assert set(lab.list_possible_issue_types()) == set(
            ["label", "near_duplicate", "non_iid", "outlier", "null", "data_valuation"]
        )

    @pytest.mark.parametrize(
        "argument_name, data_key",
        [
            ("pred_probs", "pred_probs"),
            # TODO: Add support for finding multilabel issues from features
            # ("features", "X"),
            # TODO: Add support for finding multilabel issues from knn_graph
            # ("knn_graph", "knn_graph"),
        ],
        ids=[
            "pred_probs only",
            # "features only",
            # "knn_graph only",
        ],
    )
    def test_find_label_issues(self, lab, multilabel_data, argument_name, data_key):
        """Test that the multilabel classification issue checks finds at least 90% of the
        label issues."""
        input_dict = {argument_name: multilabel_data[data_key]}
        issue_types = {"label": {}}
        lab.find_issues(**input_dict, issue_types=issue_types)
        lab.report()

        issues = lab.get_issues("label")
        issue_ids = issues.query("is_label_issue").index
        expected_issue_ids = multilabel_data["error_idx"]

        # jaccard similarity
        intersection = len(list(set(issue_ids).intersection(set(expected_issue_ids))))
        union = len(set(issue_ids)) + len(set(expected_issue_ids)) - intersection
        assert float(intersection) / union >= 0.9

    @pytest.mark.parametrize(
        "argument_name, data_key, expected_issue_types_in_summary",
        [
            ("pred_probs", "pred_probs", ["label"]),
            ("features", "X", ["near_duplicate", "non_iid", "outlier", "null"]),
            ("knn_graph", "knn_graph", ["near_duplicate", "non_iid", "outlier"]),
        ],
        ids=[
            "pred_probs only",
            "features only",
            "knn_graph only",
        ],
    )
    def test_find_issues_defaults(
        self, lab, multilabel_data, argument_name, data_key, expected_issue_types_in_summary
    ):
        """Test that the multilabel classification issue checks for various issues by default."""
        input_dict = {argument_name: multilabel_data[data_key]}
        lab.find_issues(**input_dict)
        issue_summary = lab.get_issue_summary()
        assert issue_summary["num_issues"].sum() > 0
        assert set(issue_summary["issue_type"].values) == set(expected_issue_types_in_summary)

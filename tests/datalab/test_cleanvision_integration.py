import matplotlib.pyplot as plt
import numpy as np
import pytest
import pandas as pd

from cleanlab import Datalab
import cleanlab.datalab.internal.adapter.imagelab as imagelab

LABEL_NAME = "label"
IMAGE_NAME = "image"
IMAGELAB_ISSUE_TYPES = [
    "dark",
    "light",
    "low_information",
    "odd_aspect_ratio",
    "odd_size",
    "grayscale",
    "blurry",
]
SEED = 42


class TestCleanvisionIntegration:
    @pytest.fixture
    def features(self, image_dataset):
        np.random.seed(SEED)
        return np.random.rand(len(image_dataset), 5)

    @pytest.fixture
    def num_imagelab_issues(self):
        return 7

    @pytest.fixture
    def num_datalab_issues(self):
        return 3

    @pytest.fixture
    def pred_probs(self, image_dataset):
        np.random.seed(SEED)
        return np.random.rand(len(image_dataset), 2)

    @pytest.fixture
    def set_plt_show(self, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda: None)

    @pytest.mark.usefixtures("set_plt_show")
    def test_imagelab_issues_checked(
        self, image_dataset, pred_probs, features, capsys, num_imagelab_issues, num_datalab_issues
    ):
        datalab = Datalab(data=image_dataset, label_name=LABEL_NAME, image_key=IMAGE_NAME)
        datalab.find_issues(pred_probs=pred_probs, features=features)
        captured = capsys.readouterr()
        assert (
            "Finding dark, light, low_information, odd_aspect_ratio, odd_size, grayscale, blurry images"
            in captured.out
        )
        # unable to check for non iid as feature space is too small, skipping it in interest of time
        assert "Failed to check for these issue types: [NonIIDIssueManager]" in captured.out
        assert len(datalab.issues) == len(image_dataset)

        # add up imagelab + datalab issues
        assert len(datalab.issues.columns) == (num_imagelab_issues + num_datalab_issues) * 2
        assert len(datalab.issue_summary) == num_imagelab_issues + num_datalab_issues

        all_keys = IMAGELAB_ISSUE_TYPES + [
            "statistics",
            "label",
            "outlier",
            "near_duplicate",
            # "non_iid",
        ]

        assert set(all_keys) == set(datalab.info.keys())
        datalab.report()
        captured = capsys.readouterr()

        for issue_type in IMAGELAB_ISSUE_TYPES:
            assert issue_type in captured.out

        df = pd.DataFrame(
            {
                "issue_type": [
                    "dark",
                    "light",
                    "low_information",
                    "odd_aspect_ratio",
                    "odd_size",
                    "grayscale",
                    "blurry",
                    "label",
                    "outlier",
                    "near_duplicate",
                ],
                "num_issues": [1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
            }
        )
        expected_count = df.sort_values(by="issue_type")["num_issues"].tolist()
        count = datalab.issue_summary.sort_values(by="issue_type")["num_issues"].tolist()
        assert count == expected_count
        assert datalab.issue_summary["num_issues"].sum() == df["num_issues"].sum()

    @pytest.mark.usefixtures("set_plt_show")
    def test_imagelab_max_prevalence(
        self,
        image_dataset,
        pred_probs,
        features,
        capsys,
        num_datalab_issues,
        monkeypatch,
    ):
        max_prevalence = 0
        monkeypatch.setattr(imagelab, "IMAGELAB_ISSUES_MAX_PREVALENCE", max_prevalence)
        datalab = Datalab(data=image_dataset, label_name=LABEL_NAME, image_key=IMAGE_NAME)

        datalab.find_issues(pred_probs=pred_probs, features=features)
        captured = capsys.readouterr()
        assert (
            "Finding dark, light, low_information, odd_aspect_ratio, odd_size, grayscale, blurry images"
            in captured.out
        )
        assert (
            f"from potential issues in the dataset as it exceeds max_prevalence={max_prevalence}"
            in captured.out
        )

        issue_summary = datalab.get_issue_summary()
        assert (
            len(issue_summary) == 1 + num_datalab_issues
        )  # adding 1 as no low_information issues present

    def test_imagelab_issues_not_checked(
        self, image_dataset, pred_probs, features, capsys, num_datalab_issues
    ):
        datalab = Datalab(data=image_dataset, label_name=LABEL_NAME)
        datalab.find_issues(pred_probs=pred_probs, features=features)
        captured = capsys.readouterr()
        assert (
            "Finding dark, light, low_information, odd_aspect_ratio, odd_size, grayscale, blurry images"
            not in captured.out
        )
        assert len(datalab.issues) == len(image_dataset)
        assert len(datalab.issues.columns) == num_datalab_issues * 2
        assert len(datalab.issue_summary) == num_datalab_issues

        all_keys = [
            "statistics",
            "label",
            "outlier",
            "near_duplicate",
        ]

        assert set(all_keys) == set(datalab.info.keys())
        datalab.report()
        captured = capsys.readouterr()

        for issue_type in IMAGELAB_ISSUE_TYPES:
            assert issue_type not in captured.out

    @pytest.mark.usefixtures("set_plt_show")
    def test_incremental_issue_check(self, image_dataset, pred_probs, features, capsys):
        datalab = Datalab(data=image_dataset, label_name=LABEL_NAME, image_key=IMAGE_NAME)
        datalab.find_issues(pred_probs=pred_probs, features=features, issue_types={"label": {}})

        assert len(datalab.issues) == len(image_dataset)
        assert len(datalab.issues.columns) == 2
        assert len(datalab.issue_summary) == 1

        all_keys = ["statistics", "label"]

        assert set(all_keys) == set(datalab.info.keys())

        datalab.report()
        captured = capsys.readouterr()
        assert "label" in captured.out

        datalab.find_issues(issue_types={"image_issue_types": {"dark": {}}})

        assert len(datalab.issues) == len(image_dataset)
        assert len(datalab.issues.columns) == 4
        assert len(datalab.issue_summary) == 2

        all_keys = ["statistics", "label", "dark"]

        assert set(all_keys) == set(datalab.info.keys())

        datalab.report()
        captured = capsys.readouterr()
        assert "label" in captured.out
        assert "dark" in captured.out

        with pytest.warns() as record:
            datalab.find_issues(
                issue_types={"image_issue_types": {"dark": {"threshold": 0.5}, "light": {}}}
            )
            assert len(record) == 3
            assert (
                "Overwriting columns ['is_dark_issue', 'dark_score'] in self.issues with columns from imagelab."
                == record[0].message.args[0]
            )
            assert (
                "Overwriting ['dark'] rows in self.issue_summary from imagelab."
                == record[1].message.args[0]
            )
            assert "Overwriting key dark in self.info" == record[2].message.args[0]

        assert len(datalab.issues) == len(image_dataset)
        assert len(datalab.issues.columns) == 6
        assert len(datalab.issue_summary) == 3

        all_keys = ["statistics", "label", "dark", "light"]

        assert set(all_keys) == set(datalab.info.keys())

        datalab.report()
        captured = capsys.readouterr()
        assert "label" in captured.out
        assert "dark" in captured.out

    @pytest.mark.usefixtures("set_plt_show")
    def test_labels_not_required_for_imagelab_issues(
        self, image_dataset, features, capsys, num_imagelab_issues
    ):
        datalab = Datalab(data=image_dataset, image_key=IMAGE_NAME)
        datalab.find_issues()
        captured = capsys.readouterr()
        assert (
            "Finding dark, light, low_information, odd_aspect_ratio, odd_size, grayscale, blurry images"
            in captured.out
        )
        assert len(datalab.issues) == len(image_dataset)
        assert len(datalab.issues.columns) == num_imagelab_issues * 2
        assert len(datalab.issue_summary) == num_imagelab_issues

        all_keys = IMAGELAB_ISSUE_TYPES + ["statistics"]

        assert set(all_keys) == set(datalab.info.keys())
        datalab.report()
        captured = capsys.readouterr()

        for issue_type in IMAGELAB_ISSUE_TYPES:
            assert issue_type in captured.out

import pytest
from datasets import load_dataset
import numpy as np
from PIL import Image
from cleanlab import Datalab


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


class TestCleanvisionIntegration:
    @pytest.fixture
    def labels(self):
        return ["class_0", "class_1", "class_2"]

    @pytest.fixture
    def nimages_per_class(self):
        return 5

    def generate_image(self):
        arr = np.random.randint(low=0, high=256, size=(300, 300, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        return img

    @pytest.fixture
    def generate_imagefolder(self, tmp_path, labels, nimages_per_class):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        for label in labels:
            class_dir = data_dir / label
            class_dir.mkdir()
            for i in range(nimages_per_class):
                img = self.generate_image()
                img_name = f"image_{i}.png"
                fn = class_dir / img_name
                img.save(fn)
        return data_dir

    @pytest.fixture
    def image_dataset(self, generate_imagefolder):
        data_dir = generate_imagefolder
        dataset = load_dataset("imagefolder", data_dir=data_dir, split="train")
        return dataset

    @pytest.fixture
    def features(self, image_dataset):
        return np.random.rand(len(image_dataset), 5)

    @pytest.fixture
    def pred_probs(self, image_dataset):
        return np.random.rand(len(image_dataset), 3)

    def test_imagelab_issues_checked(self, image_dataset, pred_probs, features, capsys):
        datalab = Datalab(data=image_dataset, label_name=LABEL_NAME, image_key=IMAGE_NAME)
        datalab.find_issues(pred_probs=pred_probs, features=features)
        captured = capsys.readouterr()
        assert (
            "Finding dark, light, low_information, odd_aspect_ratio, odd_size, grayscale, blurry images"
            in captured.out
        )
        assert len(datalab.issues) == len(image_dataset)
        assert len(datalab.issues.columns) == 22
        assert len(datalab.issue_summary) == 11

        all_keys = IMAGELAB_ISSUE_TYPES + [
            "statistics",
            "label",
            "outlier",
            "near_duplicate",
            "non_iid",
        ]

        assert set(all_keys) == set(datalab.info.keys())
        datalab.report()
        captured = capsys.readouterr()

        for issue_type in IMAGELAB_ISSUE_TYPES:
            assert issue_type in captured.out

    def test_imagelab_issues_not_checked(self, image_dataset, pred_probs, features, capsys):
        datalab = Datalab(data=image_dataset, label_name=LABEL_NAME)
        datalab.find_issues(pred_probs=pred_probs, features=features)
        captured = capsys.readouterr()
        assert (
            "Finding dark, light, low_information, odd_aspect_ratio, odd_size, grayscale, blurry images"
            not in captured.out
        )
        assert len(datalab.issues) == len(image_dataset)
        assert len(datalab.issues.columns) == 8
        assert len(datalab.issue_summary) == 4

        all_keys = [
            "statistics",
            "label",
            "outlier",
            "near_duplicate",
            "non_iid",
        ]

        assert set(all_keys) == set(datalab.info.keys())
        datalab.report()
        captured = capsys.readouterr()

        for issue_type in IMAGELAB_ISSUE_TYPES:
            assert issue_type not in captured.out

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

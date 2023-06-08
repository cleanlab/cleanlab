from cleanlab.datalab.adapter.imagelab import create_imagelab, ImagelabIssueFinderAdapter
from datasets import load_dataset
import os

import pytest
import numpy as np
from PIL import Image


class TestImagelabAdapater:
    def test_create_imagelab(self, image_dataset):
        imagelab = create_imagelab(image_dataset, "image")
        assert imagelab is not None
        assert hasattr(imagelab, "issues")
        assert hasattr(imagelab, "issue_summary")
        assert hasattr(imagelab, "info")

    def test_imagelab_default_issue_types(self):
        default_issues = ImagelabIssueFinderAdapter._get_datalab_specific_default_issue_types()
        assert set(default_issues) == set(
            [
                "dark",
                "light",
                "low_information",
                "odd_aspect_ratio",
                "odd_size",
                "grayscale",
                "blurry",
            ]
        )

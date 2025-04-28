from cleanlab.datalab.datalab import Datalab
from cleanlab.datalab.internal.adapter.imagelab import create_imagelab
from cleanlab.datalab.internal.adapter.constants import DEFAULT_CLEANVISION_ISSUES


class TestImagelabAdapater:
    def test_create_imagelab(self, image_dataset):
        imagelab = create_imagelab(image_dataset, "image")
        assert imagelab is not None
        assert hasattr(imagelab, "issues")
        assert hasattr(imagelab, "issue_summary")
        assert hasattr(imagelab, "info")

    def test_imagelab_serialization(self, tmp_path, synthetic_image_dataset):
        imagelab = Datalab(data=synthetic_image_dataset, image_key="image")
        imagelab.find_issues()
        imagelab.save(tmp_path, force=True)
        imagelab.load(tmp_path)

    def test_imagelab_default_issue_types(self):
        default_issues = DEFAULT_CLEANVISION_ISSUES
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

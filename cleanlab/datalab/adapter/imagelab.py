"""Module for adapting Imagelab to the Datalab API.

It uses the Adapter pattern so that Imagelab becomes compatible
with the DataIssues, IssueFinder, Reporter classes.

"""

from typing import Optional

from cleanvision.imagelab import Imagelab
from datasets.arrow_dataset import Dataset

from cleanlab.datalab.issue_finder import IssueFinder
from cleanlab.datalab.report import Reporter


def create_imagelab(dataset: Dataset, image_key: str) -> Optional[Imagelab]:
    imagelab = None
    if not image_key:
        return imagelab
    try:
        from cleanvision.imagelab import Imagelab
        from datasets.arrow_dataset import Dataset

        if isinstance(dataset, Dataset):
            imagelab = Imagelab(hf_dataset=dataset, image_key=image_key, verbosity=0)
        else:
            raise ValueError(
                "Only huggingface datasets are supported for cleanvision checks from cleanlab as of now"
            )

    except ImportError:
        raise ImportError(
            "Cannot import datasets or cleanvision package. Please install them and try again, or just install cleanlab with "
            "all optional dependencies via: `pip install cleanlab[all]`"
        )
    return imagelab


# This should only be called in the `Datalab.report` method, or by some helper function that handles all Reporter adapters.
class ImagelabReporterAdapter(Reporter):
    # Should these be __init__(self, *args, imagelab, **kwargs)? Calling super() with *args, **kwargs?
    def __init__(self, imagelab: Imagelab) -> None:
        self.imagelab = imagelab

    def report(self, num_examples: int, verbosity: Optional[int] = None) -> None:
        if not self.imagelab.issue_summary.empty:
            print("\n")
            self.imagelab.report(num_images=num_examples)


# How do we let `Datalab` call this in `Datalab.find_issues`?
class ImagelabIssueFinderAdapter(IssueFinder):
    # How should we initialize this?
    def __init__(self, datalab, verbosity, *args, **kwargs):
        self.datalab = datalab
        self.imagelab = self.datalab.imagelab
        self.verbosity = verbosity
        self.data_issues = self.datalab.data_issues

    def get_available_issue_types(self, issue_types, **kwargs):
        if issue_types is None:
            issue_types_copy = {
                issue_type: {} for issue_type in self.imagelab.list_default_issue_types()
            }
        else:
            if "image_issue_types" not in issue_types:
                return None
            else:
                issue_types_copy = issue_types["image_issue_types"].copy()

        # Remove imagelab near/exact duplicate checks
        if "is_near_duplicate_issue" in self.datalab.issues.columns:
            issue_types_copy.pop("near_duplicates")
            issue_types_copy.pop("exact_duplicates")
        return issue_types_copy

    def find_issues(self, issue_types, **kwargs) -> None:
        issue_types_copy = self.get_available_issue_types(issue_types)
        if not issue_types_copy:
            return
        try:
            if self.verbosity:
                print(f'\nFinding {", ".join(issue_types_copy.keys())} images ...')

            self.imagelab.find_issues(issue_types=issue_types_copy)

            self.data_issues.collect_statistics(self.imagelab)
            self.data_issues.collect_issues_from_imagelab(self.imagelab)
            if self.verbosity:
                print(
                    f"Image specific audit complete. {self.imagelab.issue_summary['num_images'].sum()} image issues found in the dataset."
                )

        except Exception as e:
            print(f"Error in checking for image issues: {e}")
            # failed_managers.append(self.imagelab)

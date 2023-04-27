"""Module for adapting Imagelab to the Datalab API.

It uses the Adapter pattern so that Imagelab becomes compatible
with the DataIssues, IssueFinder, Reporter classes.

"""

import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix

from cleanlab.datalab.data import Data
from cleanlab.datalab.data_issues import DataIssues
from cleanlab.datalab.issue_finder import IssueFinder
from cleanlab.datalab.report import Reporter


if TYPE_CHECKING:  # pragma: no cover
    from datasets.arrow_dataset import Dataset
    from cleanvision.imagelab import Imagelab


def create_imagelab(dataset: "Dataset", image_key: str) -> Optional["Imagelab"]:
    """Creates Imagelab instance for running CleanVision checks. CleanVision checks are only supported for
    huggingface datasets as of now.

    Parameters
    ----------
    dataset: datasets.Dataset
        Huggingface dataset used by Imagelab
    image_key: str
        key for image feature in the huggingface dataset

    Returns
    -------
    Imagelab
    """
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


class ImagelabDataIssuesAdapter(DataIssues):
    """
    Class that collects and stores information and statistics on issues found in a dataset.

    Parameters
    ----------
    data :
        The data object for which the issues are being collected.

    Parameters
    ----------
    issues : pd.DataFrame
        Stores information about each individual issue found in the data,
        on a per-example basis.
    issue_summary : pd.DataFrame
        Summarizes the overall statistics for each issue type.
    info : dict
        A dictionary that contains information and statistics about the data and each issue type.
    """

    def __init__(self, data: Data) -> None:
        super().__init__(data)

    def collect_issues_from_imagelab(self, imagelab: "Imagelab") -> None:
        """
        Collect results from Imagelab and update datalab.issues and datalab.issue_summary

        Parameters
        ----------
        imagelab: Imagelab
            Imagelab instance that run all the checks for image issue types
        """
        self._update_issues(imagelab)

        common_rows = list(
            set(imagelab.issue_summary["issue_type"]) & set(self.issue_summary["issue_type"])
        )
        if common_rows:
            warnings.warn(
                f"Overwriting {common_rows} rows in self.issue_summary from issue manager {imagelab}."
            )
        self.issue_summary = self.issue_summary[~self.issue_summary["issue_type"].isin(common_rows)]
        imagelab_summary_copy = imagelab.issue_summary.copy()
        imagelab_summary_copy.rename({"num_images": "num_issues"}, axis=1, inplace=True)
        self.issue_summary = pd.concat(
            [self.issue_summary, imagelab_summary_copy], axis=0, ignore_index=True
        )
        for issue_type in imagelab.info.keys():
            if issue_type == "statistics":
                continue
            self._update_issue_info(issue_type, imagelab.info[issue_type])


class ImagelabReporterAdapter(Reporter):
    def __init__(
        self,
        data_issues: "DataIssues",
        imagelab,
        verbosity: int = 1,
        include_description: bool = True,
    ):
        super().__init__(data_issues, verbosity, include_description)
        self.imagelab = imagelab

    def report(self, num_examples: int, verbosity: Optional[int] = None) -> None:
        super().report(num_examples)
        if not self.imagelab.issue_summary.empty:
            print("\n")
            self.imagelab.report(num_images=num_examples)


# How do we let `Datalab` call this in `Datalab.find_issues`?
class ImagelabIssueFinderAdapter(IssueFinder):
    # How should we initialize this?
    def __init__(self, datalab, verbosity):
        super().__init__(datalab, verbosity)
        self.imagelab = self.datalab._imagelab

    def _get_imagelab_issue_types(self, issue_types, **kwargs):
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

    def find_issues(
        self,
        *,
        pred_probs: Optional[np.ndarray] = None,
        features: Optional[npt.NDArray] = None,
        knn_graph: Optional[csr_matrix] = None,
        issue_types: Optional[Dict[str, Any]] = None,
    ) -> None:
        datalab_issue_types = (
            {k: v for k, v in issue_types.items() if k != "image_issue_types"}
            if issue_types
            else issue_types
        )
        super().find_issues(
            pred_probs=pred_probs,
            features=features,
            knn_graph=knn_graph,
            issue_types=datalab_issue_types,
        )

        issue_types_copy = self._get_imagelab_issue_types(issue_types)
        if not issue_types_copy:
            return
        try:
            if self.verbosity:
                print(f'\nFinding {", ".join(issue_types_copy.keys())} images ...')

            self.imagelab.find_issues(issue_types=issue_types_copy)

            self.datalab.data_issues.collect_statistics(self.imagelab)
            self.datalab.data_issues.collect_issues_from_imagelab(self.imagelab)
            if self.verbosity:
                print(
                    f"Image issues audit complete. {self.imagelab.issue_summary['num_images'].sum()} image issues found in the dataset."
                )

        except Exception as e:
            print(f"Error in checking for image issues: {e}")

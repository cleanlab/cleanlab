"""An internal wrapper around the Imagelab class from the CleanVision package to incorporate it into Datalab.
This allows low-quality images to be detected alongside other issues in computer vision datasets.
The methods/classes in this module are just intended for internal use.
"""

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, cast, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix

from cleanlab.datalab.internal.adapter.constants import (
    DEFAULT_CLEANVISION_ISSUES,
    IMAGELAB_ISSUES_MAX_PREVALENCE,
    SPURIOUS_CORRELATION_ISSUE,
)
from cleanlab.datalab.internal.data import Data
from cleanlab.datalab.internal.data_issues import DataIssues, _InfoStrategy
from cleanlab.datalab.internal.issue_finder import IssueFinder
from cleanlab.datalab.internal.report import Reporter
from cleanlab.datalab.internal.task import Task
from cleanlab.datalab.internal.spurious_correlation import SpuriousCorrelations

if TYPE_CHECKING:  # pragma: no cover
    from cleanvision import Imagelab
    from datasets.arrow_dataset import Dataset


def create_imagelab(dataset: "Dataset", image_key: Optional[str]) -> Optional["Imagelab"]:
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
        from cleanvision import Imagelab
        from datasets.arrow_dataset import Dataset

        if isinstance(dataset, Dataset):
            imagelab = Imagelab(hf_dataset=dataset, image_key=image_key)
        else:
            raise ValueError(
                "For now, only huggingface datasets are supported for running cleanvision checks inside cleanlab. You can easily convert most datasets to the huggingface dataset format."
            )

    except ImportError:
        raise ImportError(
            "Cannot import required image packages. Please install them via: `pip install cleanlab[image]` or just install cleanlab with "
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
    strategy :
        Strategy used for processing info dictionaries.

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

    def __init__(self, data: Data, strategy: Type[_InfoStrategy]) -> None:
        super().__init__(data, strategy)

    def _update_issues_imagelab(self, imagelab: "Imagelab", overlapping_issues: List[str]) -> None:
        overwrite_columns = [f"is_{issue_type}_issue" for issue_type in overlapping_issues]
        overwrite_columns.extend([f"{issue_type}_score" for issue_type in overlapping_issues])

        if overwrite_columns:
            warnings.warn(
                f"Overwriting columns {overwrite_columns} in self.issues with "
                f"columns from imagelab."
            )
            self.issues.drop(columns=overwrite_columns, inplace=True)
        new_columnns = list(set(imagelab.issues.columns).difference(self.issues.columns))
        self.issues = self.issues.join(imagelab.issues[new_columnns], how="outer")

    def filter_based_on_max_prevalence(self, issue_summary: pd.DataFrame, max_num: int):
        removed_issues = issue_summary[issue_summary["num_images"] > max_num]["issue_type"].tolist()
        if len(removed_issues) > 0:
            print(
                f"Removing {', '.join(removed_issues)} from potential issues in the dataset as it exceeds max_prevalence={IMAGELAB_ISSUES_MAX_PREVALENCE}"
            )
        return issue_summary[issue_summary["num_images"] <= max_num].copy()

    def collect_issues_from_imagelab(self, imagelab: "Imagelab", issue_types: List[str]) -> None:
        """
        Collect results from Imagelab and update datalab.issues and datalab.issue_summary

        Parameters
        ----------
        imagelab: Imagelab
            Imagelab instance that run all the checks for image issue types
        """
        overlapping_issues = list(set(self.issue_summary["issue_type"]) & set(issue_types))
        self._update_issues_imagelab(imagelab, overlapping_issues)

        if overlapping_issues:
            warnings.warn(
                f"Overwriting {overlapping_issues} rows in self.issue_summary from imagelab."
            )
        self.issue_summary = self.issue_summary[
            ~self.issue_summary["issue_type"].isin(overlapping_issues)
        ]
        imagelab_summary_copy = imagelab.issue_summary.copy()
        imagelab_summary_copy = self.filter_based_on_max_prevalence(
            imagelab_summary_copy, int(IMAGELAB_ISSUES_MAX_PREVALENCE * len(self.issues))
        )

        imagelab_summary_copy.rename({"num_images": "num_issues"}, axis=1, inplace=True)
        self.issue_summary = pd.concat(
            [self.issue_summary, imagelab_summary_copy], axis=0, ignore_index=True
        )
        for issue_type in issue_types:
            self._update_issue_info(issue_type, imagelab.info[issue_type])

    def get_info(self, issue_name: Optional[str] = None) -> Dict[str, Any]:
        # Extend method for fetching info about spurious correlations
        if issue_name != "spurious_correlations":
            return super().get_info(issue_name)

        correlations_info = self.info.get("spurious_correlations", {})
        if not correlations_info:
            raise ValueError(
                "Spurious correlations have not been calculated. Run find_issues() first."
            )
        return correlations_info


class CorrelationVisualizer:
    """Class to visualize images corresponding to the extreme (minimum and maximum) individual
    scores for each of the detected correlated properties.
    """

    def __init__(self):
        # Wrapper for VizManager that's from the optional cleanvision dependency
        try:
            from cleanvision.utils.viz_manager import VizManager

            self.viz_manager = VizManager
        except ImportError:
            raise ImportError(
                "cleanvision is required for correlation visualization. Please install it to use this feature."
            )

    def visualize(
        self, images: List, title_info: Dict, ncols: int = 2, cell_size: tuple = (2, 2)
    ) -> None:
        self.viz_manager.individual_images(
            images=images,
            title_info=title_info,
            ncols=ncols,
            cell_size=cell_size,
        )


class CorrelationReporter:
    """Class to report spurious correlations between image features and class labels detected in the data.

    If no spurious correlations are found, the class will not report anything.
    """

    def __init__(self, data_issues: "DataIssues", imagelab: "Imagelab"):
        self.imagelab: "Imagelab" = imagelab
        self.data_issues = data_issues
        self.threshold = data_issues.get_info("spurious_correlations").get("threshold")
        if not self.threshold:
            raise ValueError(
                "Spurious correlations have not been calculated. Run find_issues() first."
            )
        self.visualizer = CorrelationVisualizer()

    def report(self) -> None:
        """Reports spurious correlations between image features and class labels detected in the data,
        if any are found.
        """
        correlated_properties = self._get_correlated_properties()
        if not correlated_properties:
            return

        self._print_correlation_summary()
        correlations_df = cast(
            pd.DataFrame, self.data_issues.get_info("spurious_correlations").get("correlations_df")
        )
        filtered_correlations_df = self._get_filtered_correlated_properties(
            correlations_df, correlated_properties
        )
        print(filtered_correlations_df.to_string(index=False) + "\n")

        self._visualize_extremes(correlated_properties, self.data_issues)

    def _print_correlation_summary(self) -> None:
        print("\n\n")
        report_correlation_header = "Summary of (potentially spurious) correlations between image properties and class labels detected in the data:\n\n"
        report_correlation_metric = "Lower scores below correspond to images properties that are more strongly correlated with the class labels.\n\n"
        print(report_correlation_header + report_correlation_metric)

    def _visualize_extremes(
        self, correlated_properties: List[str], data_issues: "DataIssues"
    ) -> None:
        report_extremal_images = "Here are the images corresponding to the extreme (minimum and maximum) individual scores for each of the detected correlated properties:\n\n"
        print(report_extremal_images)
        issues = data_issues.get_issues()
        correlated_indices = {
            prop: [issues[prop].idxmin(), issues[prop].idxmax()] for prop in correlated_properties
        }
        self._visualize(correlated_indices, issues)

    def _visualize(self, correlated_indices: Dict[str, List[Any]], issues: pd.DataFrame) -> None:
        for prop, image_ids in correlated_indices.items():
            print(
                f"{'Images with minimum and maximum individual scores for ' + prop.replace('_score', '') + ' issue:'}\n"
            )
            title_info = {"scores": [f"score: {issues.loc[id, prop]:.4f}" for id in image_ids]}
            self.visualizer.visualize(
                images=[self.imagelab._dataset[id] for id in image_ids],
                title_info=title_info,
            )

    def _get_correlated_properties(self) -> List[str]:
        correlations_df = self.data_issues.get_info("spurious_correlations").get("correlations_df")
        if correlations_df is None or correlations_df.empty:
            return []
        return correlations_df.query("score < @self.threshold")["property"].tolist()

    def _get_filtered_correlated_properties(
        self, correlations_df: pd.DataFrame, correlated_properties: List[str]
    ) -> pd.DataFrame:
        query_str = "property in @correlated_properties"
        filtered_correlations_df = correlations_df.query(query_str)
        filtered_correlations_df.loc[:, "property"] = filtered_correlations_df["property"].apply(
            lambda x: x.replace("_score", "")
        )
        return filtered_correlations_df


class ImagelabReporterAdapter(Reporter):
    def __init__(
        self,
        data_issues: "DataIssues",
        imagelab: "Imagelab",
        task: Task,
        verbosity: int = 1,
        include_description: bool = True,
        show_summary_score: bool = False,
        show_all_issues: bool = False,
    ):
        super().__init__(
            data_issues=data_issues,
            task=task,
            verbosity=verbosity,
            include_description=include_description,
            show_summary_score=show_summary_score,
            show_all_issues=show_all_issues,
        )
        self.imagelab = imagelab
        self.correlation_reporter: Optional[CorrelationReporter] = None
        try:
            self.correlation_reporter = CorrelationReporter(data_issues, imagelab)
        except:
            # Spurious correlations have not been calculated
            self.correlation_reporter = None

    def report(self, num_examples: int) -> None:
        super().report(num_examples)
        self._report_imagelab(num_examples)

        # Only report spurious correlations if they've been calculated & detected
        if self.correlation_reporter is not None:
            self.correlation_reporter.report()

    def _report_imagelab(self, num_examples):
        print("\n\n")
        self.imagelab.report(
            num_images=num_examples,
            max_prevalence=IMAGELAB_ISSUES_MAX_PREVALENCE,
            print_summary=False,
            verbosity=0,
            show_id=True,
        )


class ImagelabIssueFinderAdapter(IssueFinder):
    def __init__(self, datalab, task, verbosity):
        super().__init__(datalab, task, verbosity)
        self.imagelab = self.datalab._imagelab

    def _get_imagelab_issue_types(self, issue_types, **kwargs):
        if issue_types is None:
            return DEFAULT_CLEANVISION_ISSUES

        if "image_issue_types" not in issue_types:
            return None

        issue_types_copy = {}
        for issue_type, params in issue_types["image_issue_types"].items():
            if not params:
                issue_types_copy[issue_type] = DEFAULT_CLEANVISION_ISSUES[issue_type]
            else:
                issue_types_copy[issue_type] = params

        return issue_types_copy

    def find_issues(
        self,
        *,
        pred_probs: Optional[np.ndarray] = None,
        features: Optional[npt.NDArray] = None,
        knn_graph: Optional[csr_matrix] = None,
        issue_types: Optional[Dict[str, Any]] = None,
    ) -> None:
        issue_types_to_ignore_in_datalab = ["image_issue_types", "spurious_correlations"]
        datalab_issue_types = (
            {k: v for k, v in issue_types.items() if k not in issue_types_to_ignore_in_datalab}
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
        if issue_types_copy:
            try:
                if self.verbosity:
                    print(f'Finding {", ".join(issue_types_copy.keys())} images ...')

                self.imagelab.find_issues(issue_types=issue_types_copy, verbose=False)

                self.datalab.data_issues.collect_statistics(self.imagelab)
                self.datalab.data_issues.collect_issues_from_imagelab(
                    self.imagelab, issue_types_copy.keys()
                )
            except Exception as e:
                print(f"Error in checking for image issues: {e}")

        # if issue_types is neither 'None' nor empty dictionary (non-trivial) but
        # there is no mention of 'spurious_correlations', we return.
        if issue_types and "spurious_correlations" not in issue_types:
            return

        # Check if all vision issue scores are computed
        imagelab_columns = self.imagelab.issues.columns.tolist()
        if all(
            default_cleanvision_issue + "_score" not in imagelab_columns
            for default_cleanvision_issue in DEFAULT_CLEANVISION_ISSUES.keys()
        ):
            print("Skipping spurious correlations check: Image property scores not available.")
            print(
                "To include this check, run find_issues() without parameters to compute all scores."
            )
            return

        # Spurious correlation part must be run
        print("Finding spurious correlation issues in the dataset ...")

        # the else part of the following must contain 'spurious_correlations' key
        spurious_correlation_issue_types = (
            SPURIOUS_CORRELATION_ISSUE["spurious_correlations"]
            if not issue_types
            else issue_types["spurious_correlations"]
        )

        # If threshold is not expicitly given (e.g. lab.find_issues("issue_types={"spurious_correlations": {}"))
        # we extract the default value from SPURIOUS_CORRELATION_ISSUE
        spurious_correlation_issue_threshold = spurious_correlation_issue_types.get(
            "threshold", SPURIOUS_CORRELATION_ISSUE["spurious_correlations"]["threshold"]
        )

        try:
            if self.datalab.has_labels:
                self.datalab.data_issues.info["spurious_correlations"] = (
                    handle_spurious_correlations(
                        imagelab_issues=self.imagelab.issues,
                        labels=self.datalab.labels,
                        threshold=spurious_correlation_issue_threshold,
                    )
                )
        except Exception as e:
            print(f"Error in checking for spurious correlations: {e}")


def handle_spurious_correlations(
    *,
    imagelab_issues: pd.DataFrame,
    labels: Union[np.ndarray, List[List[int]]],
    threshold: float,
    **_,
) -> Dict[str, Any]:
    imagelab_columns = imagelab_issues.columns.tolist()

    score_columns = [col for col in imagelab_columns if col.endswith("_score")]
    correlations_df = SpuriousCorrelations(
        data=imagelab_issues[score_columns], labels=labels
    ).calculate_correlations()
    return {
        "correlations_df": correlations_df,
        "threshold": threshold,
    }

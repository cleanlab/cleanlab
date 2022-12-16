from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from cleanlab.classification import CleanLearning

import pandas as pd
import numpy as np

from cleanlab.internal.validation import assert_valid_inputs

if TYPE_CHECKING:
    from cleanlab.experimental.datalab.datalab import Datalab


class IssueManager(ABC):
    """Base class for managing data issues of a particular type in a Datalab.

    For each example in a dataset, the IssueManager for a particular type of issue should compute:
    - A numeric severity score between 0 and 1, with values near 0 indicating severe instances of the issue.
    - A boolean `is_issue` value, which is True if we believe this example suffers from the issue in question.
      `is_issue` may be determined by thresholding the severity score (with an a priori determined reasonable threshold value),
      or via some other means (e.g. Confident Learning for flagging label issues).

    The IssueManager should also report:
    - A global value between 0 and 1 summarizing how severe this issue is in the dataset overall
      (e.g. the average severity across all examples in dataset or count of examples where `is_issue=True`).
    - Other interesting `info` about the issue and examples in the dataset,
      and statistics estimated from current dataset that may be reused to score this issue in future data.
      For example, `info` for label issues could contain the:
      confident_thresholds, confident_joint, predicted label for each example, etc.
      Another example is for (near)-duplicate detection issue, where `info` could contain:
      which set of examples in the dataset are all (nearly) identical.
    """

    def __init__(self, datalab: Datalab):
        self.datalab = datalab
        self.info: Optional[dict] = None
        self.issues: Optional[pd.DataFrame] = None
        self.summary: Optional[pd.DataFrame] = None

    def __repr__(self):
        class_name = self.__class__.__name__
        return class_name

    @abstractproperty
    def issue_key(self) -> str:
        """Returns a key that is used to store issue summary results about this Lab."""
        raise NotImplementedError

    @abstractmethod
    def find_issues(self, /, *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Finds occurrences of this particular issue in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def collect_info(self, /, *args, **kwargs) -> None:
        """Collects data for the info attribute of the Datalab."""
        raise NotImplementedError

    def get_summary(self):
        assert self.issues is not None
        assert self.info is not None
        pass


class HealthIssueManager(IssueManager):
    @property
    def issue_key(self) -> str:
        return "health"

    def find_issues(self, pred_probs: np.ndarray, **kwargs) -> dict:
        health_summary = self._health_summary(pred_probs=pred_probs, **kwargs)
        self.collect_info(summary=health_summary)
        return health_summary

    def collect_info(self, summary: Dict[str, Any], **kwargs) -> None:
        self.datalab.results = summary["overall_label_health_score"]
        for key in self.info_keys:
            if key == "summary":
                self.datalab.info[key] = summary
        self.datalab.info["summary"] = summary

    def _health_summary(self, pred_probs, summary_kwargs=None, **kwargs) -> dict:
        """Returns a short summary of the health of this Lab."""
        from cleanlab.dataset import health_summary

        # Validate input
        self._validate_pred_probs(pred_probs)

        if summary_kwargs is None:
            summary_kwargs = {}

        kwargs_copy = kwargs.copy()
        for k in kwargs_copy:
            if k not in [
                "asymmetric",
                "class_names",
                "num_examples",
                "joint",
                "confident_joint",
                "multi_label",
                "verbose",
            ]:
                del kwargs[k]

        summary_kwargs.update(kwargs)

        class_names = list(self.datalab._label_map.values())
        summary = health_summary(
            self.datalab._labels, pred_probs, class_names=class_names, **summary_kwargs
        )
        return summary

    def _validate_pred_probs(self, pred_probs) -> None:
        assert_valid_inputs(X=None, y=self.datalab._labels, pred_probs=pred_probs)


class LabelIssueManager(IssueManager):
    """Manages label issues in a Datalab.

    Parameters
    ----------
    datalab :
        A Datalab instance.

    clean_learning_kwargs :
        Keyword arguments to pass to the CleanLearning constructor.
    """

    def __init__(
        self,
        datalab: Datalab,
        clean_learning_kwargs: Optional[dict] = None,
    ):
        super().__init__(datalab)
        self.cl = CleanLearning(**(clean_learning_kwargs or {}))

    @property
    def issue_key(self) -> str:
        return "label"

    @property
    def info_keys(self) -> List[str]:
        return ["given_label", "predicted_label"]

    def find_issues(
        self, pred_probs: np.ndarray, model=None, **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        if pred_probs is None and model is not None:
            raise NotImplementedError("TODO: We assume pred_probs is provided.")

        # Check info
        if self.datalab.info is not None:
            # kwargs_copy = kwargs.copy()
            pass

        # Find examples with label issues
        self.issues = self.cl.find_label_issues(labels=self.datalab._labels, pred_probs=pred_probs)

        _health_summary = self._get_health_summary(pred_probs=pred_probs, **kwargs)

        # Get a summarized dataframe of the label issues
        self.summary = self.get_summary(health_summary=_health_summary)

        # Collect info about the label issues
        self.info = self.collect_info(issues=self.issues, health_summary=_health_summary)

        # Drop drop column from issues that are in the info
        self.issues = self.issues.drop(columns=self.info_keys)

        return self.issues, self.summary, self.info

    def _get_health_summary(self, pred_probs, summary_kwargs=None, **kwargs) -> dict:
        """Returns a short summary of the health of this Lab."""
        from cleanlab.dataset import health_summary

        # Validate input
        self._validate_pred_probs(pred_probs)

        if summary_kwargs is None:
            summary_kwargs = {}

        kwargs_copy = kwargs.copy()
        for k in kwargs_copy:
            if k not in [
                "asymmetric",
                "class_names",
                "num_examples",
                "joint",
                "confident_joint",
                "multi_label",
                "verbose",
            ]:
                del kwargs[k]

        summary_kwargs.update(kwargs)

        class_names = list(self.datalab._label_map.values())
        summary = health_summary(
            self.datalab._labels, pred_probs, class_names=class_names, **summary_kwargs
        )
        return summary

    def collect_info(self, issues: pd.DataFrame, health_summary: dict) -> dict:
        issues_info = {
            "num_label_issues": sum(issues["is_label_issue"]),
            "average_label_quality": issues["label_quality"].mean(),
            "given_label": issues["given_label"].tolist(),
            "predicted_label": issues["predicted_label"].tolist(),
        }

        health_summary_info = {
            "confident_joint": health_summary["joint"],
            "classes_by_label_quality": health_summary["classes_by_label_quality"],
            "overlapping_classes": health_summary["overlapping_classes"],
        }

        cl_info = {}
        for k in self.cl.__dict__:
            if k not in ["py", "noise_matrix", "inverse_noise_matrix", "confident_joint"]:
                continue
            cl_info[k] = self.cl.__dict__[k]

        info_dict = {
            **issues_info,
            **health_summary_info,
            **cl_info,
        }

        return info_dict

    def get_summary(self, health_summary: dict) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "issue_type": [self.issue_key],
                "score": [health_summary["overall_label_health_score"]],
            },
        )
        return df

    def _validate_pred_probs(self, pred_probs) -> None:
        assert_valid_inputs(X=None, y=self.datalab._labels, pred_probs=pred_probs)

    @property
    def verbosity_levels(self) -> Dict[int, Any]:
        return {
            0: "Foo",
            1: "Bar",
            2: "Baz",
        }

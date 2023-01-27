from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
from cleanlab.classification import CleanLearning
from cleanlab.internal.validation import assert_valid_inputs
from cleanlab.experimental.datalab.issue_manager import IssueManager

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab.experimental.datalab.datalab import Datalab


class LabelIssueManager(IssueManager):
    """Manages label issues in a Datalab.

    Parameters
    ----------
    datalab :
        A Datalab instance.

    clean_learning_kwargs :
        Keyword arguments to pass to the CleanLearning constructor.
    """

    issue_name: str = "label"

    def __init__(
        self,
        datalab: Datalab,
        clean_learning_kwargs: Optional[Dict[str, Any]] = None,
        health_summary_parameters: Optional[Dict[str, Any]] = None,
        **_,
    ):
        super().__init__(datalab)
        self.cl = CleanLearning(**(clean_learning_kwargs or {}))
        self.health_summary_parameters: Dict[str, Any] = health_summary_parameters or {}
        self.reset()

    def reset(self) -> None:
        """Reset the attributes of this manager based on the available datalab info
        and the keyword arguments stored as instance attributes.

        This allows the builder to use pre-computed info from the datalab to speed up
        some computations in the `find_issues` method.
        """
        if not self.health_summary_parameters:
            self.health_summary_parameters = {
                "labels": self.datalab._labels,
                "asymmetric": self.datalab.info["data"].get("asymmetric", None),
                "class_names": list(self.datalab._label_map.values()),
                "num_examples": self.datalab.info["data"].get("num_examples"),
                "joint": self.datalab.info["data"].get("joint", None),
                "confident_joint": self.datalab.info["data"].get("confident_joint", None),
                "multi_label": self.datalab.info["data"].get("multi_label", None),
            }
        self.health_summary_parameters = {
            k: v for k, v in self.health_summary_parameters.items() if v is not None
        }

    def find_issues(
        self,
        pred_probs: np.ndarray,
        model=None,
        health_summary_kwargs: Optional[Dict[str, Any]] = None,
        **_,
    ) -> None:
        if pred_probs is None and model is not None:
            raise NotImplementedError("TODO: We assume pred_probs is provided.")

        self.health_summary_parameters.update({"pred_probs": pred_probs})
        # Find examples with label issues
        self.issues = self.cl.find_label_issues(labels=self.datalab._labels, pred_probs=pred_probs)
        self.issues.rename(columns={"label_quality": self.issue_score_key}, inplace=True)

        summary_dict = self.get_health_summary(
            pred_probs=pred_probs, **(health_summary_kwargs or {})
        )

        # Get a summarized dataframe of the label issues
        self.summary = self.get_summary(score=summary_dict["overall_label_health_score"])

        # Collect info about the label issues
        self.info = self.collect_info(issues=self.issues, summary_dict=summary_dict)

        # Drop drop column from issues that are in the info
        self.issues = self.issues.drop(columns=["given_label", "predicted_label"])

    def get_health_summary(self, pred_probs, **kwargs) -> dict:
        """Returns a short summary of the health of this Lab."""
        from cleanlab.dataset import health_summary

        # Validate input
        self._validate_pred_probs(pred_probs)

        summary_kwargs = self._get_summary_parameters(pred_probs, **kwargs)
        summary = health_summary(**summary_kwargs)
        return summary

    def _get_summary_parameters(self, pred_probs, **kwargs) -> Dict["str", Any]:
        """Collects a set of input parameters for the health summary function based on
        any info available in the datalab.

        Parameters
        ----------
        pred_probs :
            The predicted probabilities for each example.

        kwargs :
            Keyword arguments to pass to the health summary function.

        Returns
        -------
        summary_parameters :
            A dictionary of parameters to pass to the health summary function.
        """
        if "confident_joint" in self.health_summary_parameters:
            summary_parameters = {
                "confident_joint": self.health_summary_parameters["confident_joint"]
            }
        elif all([x in self.health_summary_parameters for x in ["joint", "num_examples"]]):
            summary_parameters = {
                k: self.health_summary_parameters[k] for k in ["joint", "num_examples"]
            }
        else:
            summary_parameters = {
                "pred_probs": pred_probs,
                "labels": self.datalab._labels,
            }

        summary_parameters["class_names"] = self.health_summary_parameters["class_names"]

        for k in ["asymmetric", "verbose"]:
            # Start with the health_summary_parameters, then override with kwargs
            if k in self.health_summary_parameters:
                summary_parameters[k] = self.health_summary_parameters[k]
            if k in kwargs:
                summary_parameters[k] = kwargs[k]
        return summary_parameters

    def collect_info(self, issues: pd.DataFrame, summary_dict: dict) -> dict:
        issues_info = {
            "num_label_issues": sum(issues[f"is_{self.issue_name}_issue"]),
            "average_label_quality": issues[self.issue_score_key].mean(),
            "given_label": issues["given_label"].tolist(),
            "predicted_label": issues["predicted_label"].tolist(),
        }

        health_summary_info = {
            "confident_joint": summary_dict["joint"],
            "classes_by_label_quality": summary_dict["classes_by_label_quality"],
            "overlapping_classes": summary_dict["overlapping_classes"],
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

    def _validate_pred_probs(self, pred_probs) -> None:
        assert_valid_inputs(X=None, y=self.datalab._labels, pred_probs=pred_probs)

    @property
    def verbosity_levels(self) -> Dict[int, Any]:
        return {
            0: {},
            1: {"info": ["confident_joint"]},
            2: {"issue": ["given_label", "predicted_label"]},
            3: {"info": ["classes_by_label_quality", "overlapping_classes"]},
        }

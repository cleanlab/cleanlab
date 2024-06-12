from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from cleanlab import Datalab
from cleanlab.datalab.internal.report import Reporter
from cleanlab.datalab.internal.task import Task


class TestReporter:
    @pytest.fixture
    def lab(self):
        N = 30
        K = 2
        X = np.random.rand(N, K)
        y = np.random.randint(0, K, size=N)
        pred_probs = np.random.rand(N, K)
        lab = Datalab(data={"y": y}, label_name="y")
        lab.find_issues(features=X, pred_probs=pred_probs)
        return lab

    @pytest.fixture
    def data_issues(self, lab):
        return lab.data_issues

    @pytest.fixture
    def reporter(self, data_issues):
        return Reporter(data_issues=data_issues, task=Task.CLASSIFICATION)

    def test_init(self, reporter, data_issues):
        assert reporter.data_issues == data_issues
        assert reporter.verbosity == 1
        assert reporter.include_description == True
        assert reporter.show_summary_score == False

        another_reporter = Reporter(data_issues=data_issues, task=Task.CLASSIFICATION, verbosity=2)
        assert another_reporter.verbosity == 2

    def test_report(self, reporter):
        """Test that the report method works. It just wraps the get_report method in a print
        statement."""
        mock_get_report = Mock()

        with patch("builtins.print") as mock_print:  # type: ignore
            with patch.object(reporter, "get_report", mock_get_report):
                reporter.report(num_examples=3)
            mock_get_report.assert_called_with(num_examples=3)
        mock_print.assert_called_with(mock_get_report.return_value)

    @pytest.mark.parametrize("include_description", [True, False])
    def test_get_report(self, reporter, data_issues, include_description, monkeypatch):
        """Test that the report method works. Assuming we have two issue managers, each should add
        their section to the report."""

        mock_issue_manager = Mock()
        mock_issue_manager.issue_name = "foo"
        mock_issue_manager.report.return_value = "foo report"

        class MockIssueManagerFactory:
            @staticmethod
            def from_str(*args, **kwargs):
                return mock_issue_manager

        monkeypatch.setattr(
            "cleanlab.datalab.internal.report._IssueManagerFactory", MockIssueManagerFactory
        )
        mock_issues = pd.DataFrame(
            {
                "is_foo_issue": [False, True, False, False, False],
                "foo_score": [0.6, 0.2, 0.7, 0.7, 0.8],
            }
        )
        monkeypatch.setattr(data_issues, "issues", mock_issues)

        mock_issue_summary = pd.DataFrame(
            {
                "issue_type": ["foo"],
                "score": [0.6],
                "num_issues": [1],
            }
        )

        mock_info = {"foo": {"bar": "baz"}}

        monkeypatch.setattr(data_issues, "issue_summary", mock_issue_summary)

        reporter = Reporter(
            data_issues=data_issues,
            task=Task.CLASSIFICATION,
            verbosity=0,
            include_description=include_description,
        )
        monkeypatch.setattr(data_issues, "issues", mock_issues, raising=False)
        monkeypatch.setattr(data_issues, "info", mock_info, raising=False)

        monkeypatch.setattr(
            reporter, "_write_summary", lambda *args, **kwargs: "Here is a lab summary\n\n"
        )
        report = reporter.get_report(num_examples=3)
        expected_report = "\n\n".join(["Here is a lab summary", "foo report"])
        assert report == expected_report

    @pytest.mark.parametrize(
        "show_all_issues, expected_report",
        [
            (True, "Here is a lab summary\n\nfoo report\n\n\nbar report"),
            (False, "Here is a lab summary\n\nfoo report"),
        ],
    )
    def test_show_all_issues(
        self, reporter, data_issues, monkeypatch, show_all_issues, expected_report
    ):
        """Test that the report method works. Assuming we have two issue managers, each should add
        their section to the report."""

        mock_issue_manager_foo = Mock()
        mock_issue_manager_foo.issue_name = "foo"
        mock_issue_manager_foo.report.return_value = "foo report"

        mock_issue_manager_bar = Mock()
        mock_issue_manager_bar.issue_name = "bar"
        mock_issue_manager_bar.report.return_value = "bar report"

        class MockIssueManagerFactory:
            @staticmethod
            def from_str(*args, **kwargs):
                name = kwargs["issue_type"]
                issue_managers = {
                    "foo": mock_issue_manager_foo,
                    "bar": mock_issue_manager_bar,
                }
                issue_manager = issue_managers.get(name)
                if issue_manager is None:
                    raise ValueError(f"Unknown issue manager name: {name}")
                return issue_manager

        monkeypatch.setattr(
            "cleanlab.datalab.internal.report._IssueManagerFactory", MockIssueManagerFactory
        )
        mock_issues = pd.DataFrame(
            {
                "is_foo_issue": [False, True, False, False, False],
                "foo_score": [0.6, 0.2, 0.7, 0.7, 0.8],
                "is_bar_issue": [False, False, False, False, False],
                "bar_score": [0.7, 0.9, 0.8, 0.8, 0.8],
            }
        )
        monkeypatch.setattr(data_issues, "issues", mock_issues)

        # "bar" issue may be omitted in report, unless show_all_issues is True
        mock_issue_summary = pd.DataFrame(
            {
                "issue_type": ["foo", "bar"],
                "score": [0.6, 0.8],
                "num_issues": [1, 0],
            }
        )

        mock_info = {
            "foo": {"foobar": "baz"},
            "bar": {"barfoo": "bazbar"},
        }

        monkeypatch.setattr(data_issues, "issue_summary", mock_issue_summary)

        reporter = Reporter(
            data_issues=data_issues,
            task="classification",
            verbosity=0,
            include_description=False,
            show_all_issues=show_all_issues,
        )
        monkeypatch.setattr(data_issues, "issues", mock_issues, raising=False)
        monkeypatch.setattr(data_issues, "info", mock_info, raising=False)

        monkeypatch.setattr(
            reporter, "_write_summary", lambda *args, **kwargs: "Here is a lab summary\n\n"
        )
        report = reporter.get_report(num_examples=3)
        assert report == expected_report

    summary = pd.DataFrame(
        {
            "issue_type": ["foo", "bar"],
            "score": [0.6, 0.8],
            "num_issues": [1, 0],
        }
    )

    expected_filtered_summary = pd.DataFrame(
        {
            "issue_type": ["foo"],
            "score": [0.6],
            "num_issues": [1],
        }
    )

    def test_summary_with_score(self, reporter, data_issues, monkeypatch):
        """Test that the _write_summary method returns the expected output when show_summary_score is True.

        It should include the score column in the summary and a note about what the score means.
        """
        mock_statistics = {"num_examples": 100, "num_classes": 5}
        monkeypatch.setattr(data_issues, "get_info", lambda *args, **kwargs: mock_statistics)

        expected_output = (
            "Dataset Information: num_examples: 100, num_classes: 5\n\n"
            + "Here is a summary of various issues found in your data:\n\n"
            + self.expected_filtered_summary.to_string(index=False)
            + "\n\n"
            + "(Note: A lower score indicates a more severe issue across all examples in the dataset.)\n\n"
            + "Learn about each issue: https://docs.cleanlab.ai/stable/cleanlab/datalab/guide/issue_type_description.html\n"
            + "See which examples in your dataset exhibit each issue via: `datalab.get_issues(<ISSUE_NAME>)`\n\n"
            + "Data indices corresponding to top examples of each issue are shown below.\n\n\n"
        )

        reporter.show_summary_score = True
        assert reporter._write_summary(self.summary) == expected_output

    def test_summary_without_score(self, reporter, data_issues, monkeypatch):
        mock_statistics = {"num_examples": 100, "num_classes": 5}
        monkeypatch.setattr(data_issues, "get_info", lambda *args, **kwargs: mock_statistics)

        expected_output = (
            "Dataset Information: num_examples: 100, num_classes: 5\n\n"
            + "Here is a summary of various issues found in your data:\n\n"
            + self.expected_filtered_summary.drop(columns=["score"]).to_string(index=False)
            + "\n\n"
            + "Learn about each issue: https://docs.cleanlab.ai/stable/cleanlab/datalab/guide/issue_type_description.html\n"
            + "See which examples in your dataset exhibit each issue via: `datalab.get_issues(<ISSUE_NAME>)`\n\n"
            + "Data indices corresponding to top examples of each issue are shown below.\n\n\n"
        )

        reporter.show_summary_score = False
        assert reporter._write_summary(self.summary) == expected_output

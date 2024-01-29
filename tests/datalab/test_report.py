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

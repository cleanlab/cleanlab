import numpy as np
import pandas as pd
import pytest

from cleanlab.datalab.internal.issue_manager.placeholder import PlaceholderIssueManager

SEED = 42


class TestPlaceholderIssueManager:
    @pytest.fixture
    def clean_features(self):
        np.random.seed(SEED)
        features = np.random.uniform(10, 80, size=(200, 3))
        return features

    @pytest.fixture
    def features_with_placeholder(self):
        np.random.seed(SEED)
        features = np.random.uniform(10, 80, size=(200, 2))
        features[::4, 0] = -99
        features[1::5, 1] = -99
        return features

    @pytest.fixture
    def issue_manager(self, lab):
        return PlaceholderIssueManager(datalab=lab)

    def test_init(self, lab, issue_manager):
        assert issue_manager.datalab == lab

    def test_find_issues_clean_data(self, issue_manager, clean_features):
        issue_manager.find_issues(features=clean_features)
        assert not issue_manager.issues["is_placeholder_issue"].any()
        assert issue_manager.summary["score"][0] == pytest.approx(1.0, abs=1e-7)
        assert issue_manager.info["placeholder_by_column"] == {}

    def test_find_issues_detects_negative_sentinel(self, issue_manager, features_with_placeholder):
        issue_manager.find_issues(features=features_with_placeholder)
        issues = issue_manager.issues

        assert issues["is_placeholder_issue"].sum() > 0
        assert issues["placeholder_score"].mean() < 1.0
        assert any(
            any(np.isclose(v, -99) for v in vals)
            for vals in issue_manager.info["placeholder_by_column"].values()
        )

    def test_legitimate_negative_values_not_flagged(self, issue_manager):
        np.random.seed(SEED)
        features = np.random.uniform(-50, -10, size=(200, 2))
        issue_manager.find_issues(features=features)
        assert not issue_manager.issues["is_placeholder_issue"].any()

    def test_find_issues_with_dataframe(self, issue_manager, features_with_placeholder):
        df = pd.DataFrame(features_with_placeholder, columns=["age", "income"])
        issue_manager.find_issues(features=df)
        assert "age" in issue_manager.info["placeholder_by_column"]

    def test_report(self, issue_manager, features_with_placeholder):
        issue_manager.find_issues(features=features_with_placeholder)
        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
        )
        assert isinstance(report, str)
        assert "placeholder issues" in report

    def test_requires_features(self, issue_manager):
        with pytest.raises(ValueError, match="features must be provided"):
            issue_manager.find_issues()

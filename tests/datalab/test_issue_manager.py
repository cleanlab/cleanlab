import numpy as np
import pandas as pd
import pytest

from cleanlab.datalab.issue_manager import IssueManager
from cleanlab.datalab.factory import REGISTRY, register


class TestCustomIssueManager:
    @pytest.mark.parametrize(
        "score",
        [0, 0.5, 1],
        ids=["zero", "positive_float", "one"],
    )
    def test_make_summary_with_score(self, custom_issue_manager, score):
        summary = custom_issue_manager.make_summary(score=score)

        expected_summary = pd.DataFrame(
            {
                "issue_type": [custom_issue_manager.issue_name],
                "score": [score],
            }
        )
        assert pd.testing.assert_frame_equal(summary, expected_summary) is None

    @pytest.mark.parametrize(
        "score",
        [-0.3, 1.5, np.nan, np.inf, -np.inf],
        ids=["negative_float", "greater_than_one", "nan", "inf", "negative_inf"],
    )
    def test_make_summary_invalid_score(self, custom_issue_manager, score):
        with pytest.raises(ValueError):
            custom_issue_manager.make_summary(score=score)


def test_register_custom_issue_manager(monkeypatch):
    import io
    import sys

    assert "foo" not in REGISTRY

    @register
    class Foo(IssueManager):
        issue_name = "foo"

        def find_issues(self):
            pass

    assert "foo" in REGISTRY
    assert REGISTRY["foo"] == Foo

    # Reregistering should overwrite the existing class, put print a warning

    monkeypatch.setattr("sys.stdout", io.StringIO())

    @register
    class NewFoo(IssueManager):
        issue_name = "foo"

        def find_issues(self):
            pass

    assert "foo" in REGISTRY
    assert REGISTRY["foo"] == NewFoo
    assert all(
        [
            text in sys.stdout.getvalue()
            for text in ["Warning: Overwriting existing issue manager foo with ", "NewFoo"]
        ]
    ), "Should print a warning"

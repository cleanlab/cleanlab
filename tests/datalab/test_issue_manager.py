import numpy as np
import pandas as pd
import pytest

from cleanlab.datalab.internal.issue_manager import IssueManager
from cleanlab.datalab.internal.issue_manager_factory import (
    REGISTRY,
    register,
)
from cleanlab.datalab.internal.task import Task


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

    class Foo(IssueManager):
        issue_name = "foo"

        def find_issues(self):
            pass

    Foo = register(Foo)

    assert REGISTRY[Task.CLASSIFICATION].get("foo") == Foo

    # Reregistering should overwrite the existing class, put print a warning

    monkeypatch.setattr("sys.stdout", io.StringIO())

    class NewFoo(IssueManager):
        issue_name = "foo"

        def find_issues(self):
            pass

    NewFoo = register(NewFoo)

    assert REGISTRY[Task.CLASSIFICATION].get("foo") == NewFoo
    assert all(
        [
            text in sys.stdout.getvalue()
            for text in [
                "Warning: Overwriting existing issue manager foo with ",
                "NewFoo",
                " for task classification.",
            ]
        ]
    ), "Should print a warning"

    # Reregistering for task should overwrite the existing class, put print a warning
    class NewerFoo(IssueManager):
        issue_name = "label"

        def find_issues(self):
            pass

    NewerFoo = register(NewerFoo, task="classification")

    assert REGISTRY[Task.CLASSIFICATION].get("label") == NewerFoo
    assert all(
        [
            text in sys.stdout.getvalue()
            for text in [
                "Warning: Overwriting existing issue manager label with ",
                "NewerFoo",
                " for task classification.",
            ]
        ]
    ), "Should print a warning"

    # Registering any issue manager for another task is permitted
    class Bar(IssueManager):
        issue_name = "bar"

        def find_issues(self):
            pass

    Bar = register(Bar, task="regression")

    assert REGISTRY[Task.REGRESSION].get("bar") == Bar

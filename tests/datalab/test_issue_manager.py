import numpy as np
import pandas as pd
import pytest

from cleanlab.datalab.issue_manager import IssueManager
from cleanlab.datalab.factory import REGISTRY, register


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

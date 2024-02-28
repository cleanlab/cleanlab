import pytest

from cleanlab.datalab.internal.issue_manager_factory import register, REGISTRY
from cleanlab import Datalab
from cleanlab.datalab.internal.issue_manager.issue_manager import IssueManager
from cleanlab.datalab.internal.task import Task


@pytest.fixture
def registry():
    return REGISTRY


def test_list_possible_issue_types(registry):
    lab = Datalab(data=[], label_name=None)
    issue_types = lab.list_possible_issue_types()
    assert isinstance(issue_types, list)
    possible_issues = [
        "outlier",
        "near_duplicate",
        "non_iid",
        "label",
        "class_imbalance",
        "underperforming_group",
        "data_valuation",
        "null",
    ]
    assert set(issue_types) == set(possible_issues)

    test_key = "test_for_list_possible_issue_types"

    class TestIssueManager(IssueManager):
        issue_name = test_key

    TestIssueManager = register(TestIssueManager)

    issue_types = lab.list_possible_issue_types()
    assert set(issue_types) == set(
        possible_issues + [test_key]
    ), "New issue type should be added to the list"

    # Clean up
    del registry[Task.CLASSIFICATION][test_key]

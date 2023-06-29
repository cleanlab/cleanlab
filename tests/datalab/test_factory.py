import pytest

from cleanlab.datalab.factory import register, REGISTRY
from cleanlab import Datalab
from cleanlab.datalab.issue_manager.issue_manager import IssueManager


@pytest.fixture
def registry():
    return REGISTRY


def test_list_possible_issue_types(registry):
    issue_types = Datalab.list_possible_issue_types()
    assert isinstance(issue_types, list)
    defaults = ["label", "outlier", "near_duplicate", "non_iid"]
    assert set(issue_types) == set(defaults)

    test_key = "test_for_list_possible_issue_types"

    @register
    class TestIssueManager(IssueManager):
        issue_name = test_key

    issue_types = Datalab.list_possible_issue_types()
    assert set(issue_types) == set(
        defaults + [test_key]
    ), "New issue type should be added to the list"

    # Clean up
    del registry[test_key]

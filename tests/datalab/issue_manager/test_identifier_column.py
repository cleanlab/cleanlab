import numpy as np
import pytest

from cleanlab.datalab.internal.issue_manager.identifier_column import IdentifierIssueManager

SEED = 42


class TestIdentifierIssueManager:
    @pytest.fixture
    def embeddings(self):
        np.random.seed(SEED)
        embedding_array = np.random.random((4, 3))
        return embedding_array

    @pytest.fixture
    def embeddings_with_id(self):
        np.random.seed(SEED)
        embedding_array = np.random.random((4, 3))
        embedding_array[:, 0] = np.arange(4)
        return embedding_array

    @pytest.fixture
    def issue_manager(self, lab):
        return IdentifierIssueManager(datalab=lab)

    def test_init(self, lab, issue_manager):
        assert issue_manager.datalab == lab

    def test_find_issues(self, issue_manager, embeddings):
        np.random.seed(SEED)
        issue_manager.find_issues(features=embeddings)
        summary_sort, info_sort = (
            issue_manager.summary,
            issue_manager.info,
        )
        assert summary_sort["issue_type"][0] == "identifier_column"
        assert summary_sort["score"][0] == 1
        assert info_sort.get("identifier_column") == []

    def test_find_issues_with_id(self, issue_manager, embeddings_with_id):
        np.random.seed(SEED)
        issue_manager.find_issues(features=embeddings_with_id)
        summary_sort, info_sort = (
            issue_manager.summary,
            issue_manager.info,
        )
        assert summary_sort["issue_type"][0] == "identifier_column"
        assert summary_sort["score"][0] == 0
        assert info_sort.get("identifier_column") == [0]

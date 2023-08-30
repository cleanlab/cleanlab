import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite
from hypothesis.extra.numpy import arrays


from cleanlab import Datalab
from cleanlab.datalab.internal.issue_manager.duplicate import NearDuplicateIssueManager

SEED = 42


@composite
def embeddings_strategy(draw):
    shape_strategy = st.tuples(
        st.integers(min_value=3, max_value=20), st.integers(min_value=2, max_value=2)
    )
    element_strategy = st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    )
    embeddings = draw(
        arrays(
            dtype=np.float64,
            shape=shape_strategy,
            elements=element_strategy,
            unique=True,
        )
    )
    return embeddings


class TestNearDuplicateIssueManager:
    @pytest.fixture
    def embeddings(self, lab):
        np.random.seed(SEED)
        embeddings_array = 0.5 + 0.1 * np.random.rand(lab.get_info("statistics")["num_examples"], 2)
        embeddings_array[4, :] = (
            embeddings_array[3, :] + np.random.rand(embeddings_array.shape[1]) * 0.001
        )
        return {"embedding": embeddings_array}

    @pytest.fixture
    def issue_manager(self, lab, embeddings, monkeypatch):
        mock_data = lab.data.from_dict({**lab.data.to_dict(), **embeddings})
        monkeypatch.setattr(lab, "data", mock_data)
        return NearDuplicateIssueManager(
            datalab=lab,
            metric="euclidean",
            k=2,
        )

    def test_init(self, lab, issue_manager):
        assert issue_manager.datalab == lab
        assert issue_manager.metric == "euclidean"
        assert issue_manager.k == 2
        assert issue_manager.threshold == 0.13

        issue_manager = NearDuplicateIssueManager(
            datalab=lab,
            threshold=0.1,
        )
        assert issue_manager.threshold == 0.1

    def test_find_issues(self, issue_manager, embeddings):
        issue_manager.find_issues(features=embeddings["embedding"])
        issues, summary, info = issue_manager.issues, issue_manager.summary, issue_manager.info
        expected_issue_mask = np.array([False] * 3 + [True] * 2)
        assert np.all(
            issues["is_near_duplicate_issue"] == expected_issue_mask
        ), "Issue mask should be correct"
        assert summary["issue_type"][0] == "near_duplicate"
        assert summary["score"][0] == pytest.approx(expected=0.03122489, abs=1e-7)

        assert (
            info.get("near_duplicate_sets", None) is not None
        ), "Should have sets of near duplicates"

        new_issue_manager = NearDuplicateIssueManager(
            datalab=issue_manager.datalab,
            metric="euclidean",
            k=2,
            threshold=0.1,
        )
        new_issue_manager.find_issues(features=embeddings["embedding"])

    def test_report(self, issue_manager, embeddings):
        issue_manager.find_issues(features=embeddings["embedding"])
        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
        )
        assert isinstance(report, str)
        assert (
            "------------------ near_duplicate issues -------------------\n\n"
            "Number of examples with this issue:"
        ) in report

        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
            verbosity=3,
        )
        assert "Additional Information: " in report

    @given(embeddings=embeddings_strategy())
    @settings(deadline=800)
    def test_near_duplicate_sets(self, embeddings):
        data = {"metadata": ["" for _ in range(len(embeddings))]}
        lab = Datalab(data)
        issue_manager = NearDuplicateIssueManager(
            datalab=lab,
            metric="euclidean",
            k=2,
        )
        embeddings = np.array(embeddings)
        issue_manager.find_issues(features=embeddings)
        near_duplicate_sets = issue_manager.info["near_duplicate_sets"]
        issues = issue_manager.issues["is_near_duplicate_issue"]

        # Test: Near duplicates are symmetric
        all_symmetric = all(
            i in near_duplicate_sets[j]
            for i, near_duplicates in enumerate(near_duplicate_sets)
            for j in near_duplicates
        )
        assert all_symmetric, "Some near duplicate sets are not symmetric"

        # Test: Near duplicate sets for issues
        assert all(
            len(near_duplicate_set) == 0
            for i, near_duplicate_set in enumerate(near_duplicate_sets)
            if not issues[i]
        ), "Non-issue examples should not have near duplicate sets"
        assert all(
            len(near_duplicate_set) > 0
            for i, near_duplicate_set in enumerate(near_duplicate_sets)
            if issues[i]
        ), "Issue examples should have near duplicate sets"

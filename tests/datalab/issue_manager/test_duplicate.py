import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st
from hypothesis.strategies import composite
from hypothesis.extra.numpy import arrays


from cleanlab import Datalab
from cleanlab.datalab.internal.issue_manager.duplicate import NearDuplicateIssueManager

from .conftest import knn_graph_strategy

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
        assert summary["score"][0] == pytest.approx(expected=0.4734458, abs=1e-7)

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

    def test_scores_of_examples_with_issues_are_smaller_than_those_without(
        self, issue_manager, embeddings
    ):
        # TODO: Turn this into a property-based test
        issue_manager.find_issues(features=embeddings["embedding"])
        is_issue = issue_manager.issues["is_near_duplicate_issue"]
        scores = issue_manager.issues["near_duplicate_score"]
        max_issue_score = np.max(scores[is_issue])
        min_non_issue_score = np.min(scores[~is_issue])
        assert max_issue_score < min_non_issue_score

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
        all_non_issues_have_empty_near_duplicate_sets = all(
            len(near_duplicate_set) == 0
            for i, near_duplicate_set in enumerate(near_duplicate_sets)
            if not issues[i]
        )
        assert (
            all_non_issues_have_empty_near_duplicate_sets
        ), "Non-issue examples should not have near duplicate sets"
        all_issues_have_non_empty_near_duplicate_sets = all(
            len(near_duplicate_set) > 0
            for i, near_duplicate_set in enumerate(near_duplicate_sets)
            if issues[i]
        )
        assert (
            all_issues_have_non_empty_near_duplicate_sets
        ), "Issue examples should have near duplicate sets"


def build_issue_manager(
    draw, num_samples_strategy, k_neighbors_strategy, with_issues=False, threshold=None
):
    """Create a random knn_graph with the given number of samples and k neighbors.
    Run the NearDuplicateIssueManager on the knn_graph and return the issue manager.
    A threshold can be provided to control the number of issues for small graphs.
    A with_issues flag can be provided to control whether the issue manager should have issues.
    """

    if with_issues:
        knn_graph = draw(
            knn_graph_strategy(num_samples=num_samples_strategy, k_neighbors=k_neighbors_strategy)
        )
    else:
        knn_graph = draw(
            knn_graph_strategy(
                num_samples=num_samples_strategy, k_neighbors=k_neighbors_strategy, min_distance=0.1
            )
        )

    lab = Datalab(data={})
    inputs = {"datalab": lab, "threshold": threshold}
    inputs = {k: v for k, v in inputs.items() if v is not None}
    issue_manager = NearDuplicateIssueManager(**inputs)
    issue_manager.find_issues(knn_graph=knn_graph)
    issues = issue_manager.issues["is_near_duplicate_issue"]

    if with_issues:
        assume(any(issues))
    else:
        assume(not any(issues))
    return issue_manager


@st.composite
def no_issue_issue_manager_strategy(draw):
    """Strategy for generating NearDuplicateIssueManagers with no issues."""
    return build_issue_manager(
        draw,
        st.integers(min_value=10, max_value=50),
        st.integers(min_value=2, max_value=5),
        with_issues=False,
        threshold=0.0001,
    )


@st.composite
def issue_manager_with_issues_strategy(draw):
    """Strategy for generating NearDuplicateIssueManagers with issues."""
    return build_issue_manager(
        draw,
        st.integers(min_value=10, max_value=20),
        st.integers(min_value=2, max_value=5),
        with_issues=True,
        threshold=0.9,
    )


class TestNearDuplicateSets:
    """Property-based tests properties of near duplicate sets found in a knn graph."""

    @pytest.mark.slow
    @given(issue_manager=no_issue_issue_manager_strategy())
    @settings(
        deadline=800, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large]
    )
    def test_near_duplicate_sets_empty_if_no_issue_next(self, issue_manager):
        near_duplicate_sets = issue_manager.info["near_duplicate_sets"]
        assert all(len(near_duplicate_set) == 0 for near_duplicate_set in near_duplicate_sets)

    @given(issue_manager=issue_manager_with_issues_strategy())
    @settings(deadline=800, max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_symmetric_and_flagged_consistency(self, issue_manager):
        near_duplicate_sets = issue_manager.info["near_duplicate_sets"]
        issues = issue_manager.issues["is_near_duplicate_issue"]

        # Test symmetry: If A is in near_duplicate_set of B, then B should be in near_duplicate_set of A.
        for i, near_duplicates in enumerate(near_duplicate_sets):
            for j in near_duplicates:
                assert (
                    i in near_duplicate_sets[j]
                ), f"Example {j} is in near_duplicate_set of {i}, but not vice versa"

        # Test consistency of flags with near_duplicate_sets
        for i, near_duplicate_set in enumerate(near_duplicate_sets):
            if issues[i]:
                # Near duplicate sets of flagged examples should not be empty
                assert (
                    len(near_duplicate_set) > 0
                ), f"Near duplicate set of flagged example {i} is empty"

                # Check if all examples in the near_duplicate_set of a flagged example are also flagged
                flagged_in_set = [issues[j] for j in near_duplicate_set]
                assert all(
                    flagged_in_set
                ), f"Example {i} is flagged as near_duplicate but some examples in its near_duplicate_set are not flagged"

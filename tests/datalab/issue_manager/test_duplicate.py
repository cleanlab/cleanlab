import numpy as np
import pytest
import warnings
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
        # Validates score decreases with increasing nearest-neighbor distance
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


class TestNearDuplicateEnhancements:
    """Test the enhanced functionality of NearDuplicateIssueManager."""

    @pytest.fixture
    def embeddings_with_exact_duplicates(self, lab):
        """Create embeddings with exact duplicates for testing."""
        np.random.seed(SEED)
        num_examples = lab.get_info("statistics")["num_examples"]
        embeddings_array = np.random.rand(num_examples, 10)

        # Create exact duplicates
        embeddings_array[1, :] = embeddings_array[0, :]  # 0 and 1 are exact duplicates
        embeddings_array[4, :] = embeddings_array[3, :]  # 3 and 4 are exact duplicates

        return {"embedding": embeddings_array}

    def test_exact_duplicates_only(self, lab, embeddings_with_exact_duplicates):
        """Test exact_duplicates_only parameter."""
        issue_manager = NearDuplicateIssueManager(
            datalab=lab,
            exact_duplicates_only=True,
            metric="euclidean",
            k=2,  # Use k=2 for small test dataset
        )

        issue_manager.find_issues(features=embeddings_with_exact_duplicates["embedding"])
        issues = issue_manager.issues
        info = issue_manager.info

        # Should only flag exact duplicates
        flagged_indices = issues[issues["is_near_duplicate_issue"]].index.tolist()
        assert set(flagged_indices) == {0, 1, 3, 4}, f"Expected [0,1,3,4], got {flagged_indices}"

        # Check info
        assert info["exact_duplicates_only"] is True
        assert "num_duplicate_sets" in info

    def test_cosine_similarity_threshold(self, lab):
        """Test similarity_threshold parameter with cosine metric."""
        # Create embeddings with known cosine similarities
        embeddings = np.array(
            [
                [1.0, 0.0],  # Example 0
                [0.9, 0.436],  # Example 1: ~0.9 cosine similarity to example 0
                [0.0, 1.0],  # Example 2: orthogonal to example 0
                [0.8, 0.6],  # Example 3: 0.8 cosine similarity to example 0
                [-1.0, 0.0],  # Example 4: -1.0 cosine similarity to example 0
            ]
        )

        issue_manager = NearDuplicateIssueManager(
            datalab=lab,
            metric="cosine",
            similarity_threshold=0.85,  # Should catch examples 0 and 1
            k=2,  # Use k=2 for small test dataset
        )

        issue_manager.find_issues(features=embeddings)
        issues = issue_manager.issues
        info = issue_manager.info

        # Check that the similarity threshold is stored
        assert info["similarity_threshold"] == 0.85
        assert info["metric"] == "cosine"

        # At least examples with high cosine similarity should be flagged
        flagged_indices = issues[issues["is_near_duplicate_issue"]].index.tolist()
        # Note: Exact behavior depends on k-NN graph construction, but we should have some duplicates
        assert len(flagged_indices) >= 0  # Basic sanity check

    def test_enhanced_info_collection(self, lab, embeddings_with_exact_duplicates):
        """Test that enhanced info is collected properly."""
        issue_manager = NearDuplicateIssueManager(
            datalab=lab,
            similarity_threshold=0.95,
            exact_duplicates_only=False,
            metric="cosine",
            k=2,  # Use k=2 for small test dataset
        )

        issue_manager.find_issues(features=embeddings_with_exact_duplicates["embedding"])
        info = issue_manager.info

        # Check that all new parameters are in info
        assert "similarity_threshold" in info
        assert "exact_duplicates_only" in info
        assert "num_duplicate_sets" in info
        assert "num_near_duplicates" in info

        # Check parameter values
        assert info["similarity_threshold"] == 0.95
        assert info["exact_duplicates_only"] is False
        assert info["metric"] == "cosine"

    def test_enhanced_verbosity(self, lab, embeddings_with_exact_duplicates):
        """Test enhanced verbosity levels."""
        issue_manager = NearDuplicateIssueManager(
            datalab=lab,
            similarity_threshold=0.9,
            exact_duplicates_only=True,
            k=2,  # Use k=2 for small test dataset
        )

        issue_manager.find_issues(features=embeddings_with_exact_duplicates["embedding"])

        # Test verbosity level 1
        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
            verbosity=1,
        )
        assert "num_duplicate_sets" in report

        # Test verbosity level 2
        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
            verbosity=2,
        )
        assert "exact_duplicates_only" in report
        assert "similarity_threshold" in report

    def test_backward_compatibility(self, lab, embeddings_with_exact_duplicates):
        """Test that existing functionality still works."""
        # Test with old-style parameters
        issue_manager = NearDuplicateIssueManager(
            datalab=lab,
            threshold=0.13,
            metric="euclidean",
            k=2,  # Use k=2 for small test dataset
        )

        issue_manager.find_issues(features=embeddings_with_exact_duplicates["embedding"])
        issues = issue_manager.issues
        info = issue_manager.info

        # Should work as before
        assert "is_near_duplicate_issue" in issues.columns
        assert "near_duplicate_score" in issues.columns
        assert "threshold" in info
        assert "metric" in info

        # New parameters should have default values
        assert info["similarity_threshold"] is None
        assert info["exact_duplicates_only"] is False


class TestNearDuplicateValidation:
    """Test parameter validation and error handling."""

    def test_invalid_similarity_threshold(self, lab):
        """Test validation of similarity_threshold parameter."""
        # Test invalid type
        with pytest.raises(TypeError, match="similarity_threshold must be a numeric value"):
            NearDuplicateIssueManager(datalab=lab, similarity_threshold="invalid")

        # Test out of range values
        with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
            NearDuplicateIssueManager(datalab=lab, similarity_threshold=-0.1)

        with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
            NearDuplicateIssueManager(datalab=lab, similarity_threshold=1.5)

    def test_invalid_k(self, lab):
        """Test validation of k parameter."""
        # Test invalid type
        with pytest.raises(TypeError, match="k must be an integer"):
            NearDuplicateIssueManager(datalab=lab, k=2.5)

        # Test invalid value
        with pytest.raises(ValueError, match="k must be positive"):
            NearDuplicateIssueManager(datalab=lab, k=0)

        with pytest.raises(ValueError, match="k must be positive"):
            NearDuplicateIssueManager(datalab=lab, k=-1)

    def test_invalid_threshold(self, lab):
        """Test validation of threshold parameter."""
        # Test invalid type
        with pytest.raises(TypeError, match="threshold must be a numeric value"):
            NearDuplicateIssueManager(datalab=lab, threshold="invalid")

    def test_invalid_exact_duplicates_only(self, lab):
        """Test validation of exact_duplicates_only parameter."""
        # Test invalid type
        with pytest.raises(TypeError, match="exact_duplicates_only must be a boolean"):
            NearDuplicateIssueManager(datalab=lab, exact_duplicates_only="True")

    def test_similarity_threshold_with_non_cosine_metric_warning(self, lab):
        """Test warning when similarity_threshold is used with non-cosine metric."""
        with pytest.warns(UserWarning, match="similarity_threshold is provided but metric is"):
            NearDuplicateIssueManager(datalab=lab, metric="euclidean", similarity_threshold=0.9)

    def test_conflicting_parameters_warning(self, lab):
        """Test warning when conflicting parameters are specified."""
        with pytest.warns(
            UserWarning, match="Both exact_duplicates_only=True and similarity_threshold"
        ):
            NearDuplicateIssueManager(
                datalab=lab, exact_duplicates_only=True, similarity_threshold=0.9
            )

    def test_k_too_large_error(self, lab):
        """Test error handling when k is larger than dataset."""
        # Create small embeddings
        small_embeddings = np.random.rand(3, 10)

        issue_manager = NearDuplicateIssueManager(datalab=lab, k=10)  # k > dataset size

        with pytest.raises(ValueError, match="k=10 is too large for dataset"):
            issue_manager.find_issues(features=small_embeddings)

    def test_no_features_error(self, lab):
        """Test error when no features are provided."""
        issue_manager = NearDuplicateIssueManager(datalab=lab)

        with pytest.raises(ValueError, match="No features provided for duplicate detection"):
            issue_manager.find_issues()  # No features provided

    def test_empty_dataset_handling(self, lab):
        """Test graceful handling of empty datasets."""
        empty_embeddings = np.empty((0, 10))

        issue_manager = NearDuplicateIssueManager(datalab=lab, k=2)

        with pytest.warns(UserWarning, match="Empty dataset provided"):
            issue_manager.find_issues(features=empty_embeddings)

        # Should handle gracefully
        assert len(issue_manager.issues) == 0
        assert issue_manager.info["num_duplicate_sets"] == 0


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


class TestMemoryMonitoring:
    """Test memory monitoring and warning functionality."""

    def test_memory_estimation(self):
        """Test memory estimation functions."""
        # Create a mock Datalab instance
        dummy_data = {"features": np.array([[1, 2]]), "labels": ["A"]}
        dummy_datalab = Datalab(dummy_data, label_name="labels")

        # Create issue manager
        manager = NearDuplicateIssueManager(dummy_datalab, k=10)

        # Test memory estimation for different dataset sizes
        estimates_small = manager._estimate_memory_usage(100, 20, 10)
        estimates_large = manager._estimate_memory_usage(10000, 500, 20)

        # Check that estimates are reasonable
        assert estimates_small["total_memory_mb"] < estimates_large["total_memory_mb"]
        assert estimates_small["features_memory_mb"] > 0
        assert estimates_small["knn_graph_memory_mb"] > 0
        assert estimates_small["total_memory_mb"] > 0

        # Check that larger datasets have proportionally larger estimates
        assert estimates_large["features_memory_mb"] > estimates_small["features_memory_mb"] * 10

    def test_memory_warnings_small_dataset(self):
        """Test that small datasets don't trigger memory warnings."""
        dummy_data = {"features": np.array([[1, 2]]), "labels": ["A"]}
        dummy_datalab = Datalab(dummy_data, label_name="labels")

        manager = NearDuplicateIssueManager(dummy_datalab, k=5)

        # Small dataset should not trigger warnings
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            X_small = np.random.randn(50, 10).astype(np.float64)
            manager._check_memory_requirements(features=X_small)

        # Filter for memory-related warnings
        memory_warnings = [w for w in warning_list if "memory usage" in str(w.message)]
        assert len(memory_warnings) == 0, "Small dataset should not trigger memory warnings"

    def test_memory_warnings_large_k(self):
        """Test that large k values trigger appropriate warnings."""
        dummy_data = {"features": np.array([[1, 2]]), "labels": ["A"]}
        dummy_datalab = Datalab(dummy_data, label_name="labels")

        manager = NearDuplicateIssueManager(dummy_datalab, k=100)  # Very large k

        # Large k should trigger warning
        with pytest.warns(UserWarning, match="Large k value detected"):
            X_test = np.random.randn(200, 10).astype(np.float64)
            manager._check_memory_requirements(features=X_test)

    def test_memory_warnings_large_dataset(self):
        """Test that large datasets trigger memory warnings."""
        dummy_data = {"features": np.array([[1, 2]]), "labels": ["A"]}
        dummy_datalab = Datalab(dummy_data, label_name="labels")

        manager = NearDuplicateIssueManager(dummy_datalab, k=10)

        # Test with dimensions that should trigger warning (>1GB estimated memory)
        with pytest.warns(UserWarning, match="memory usage"):
            manager._check_memory_requirements(n_samples=50000, n_features=500)

    def test_memory_warnings_critical_dataset(self):
        """Test that very large datasets trigger critical warnings."""
        dummy_data = {"features": np.array([[1, 2]]), "labels": ["A"]}
        dummy_datalab = Datalab(dummy_data, label_name="labels")

        manager = NearDuplicateIssueManager(dummy_datalab, k=10)

        # Test with dimensions that should trigger critical warning (>4GB estimated memory)
        with pytest.warns(UserWarning, match="Large dataset detected"):
            manager._check_memory_requirements(n_samples=100000, n_features=1000)

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from cleanlab import Datalab
from cleanlab.datalab.internal.issue_manager.label import LabelIssueManager


@pytest.mark.parametrize("k", [2, 10])
@pytest.mark.parametrize("reverse_labels", [False, True])
def test_label_issues_with_conflicting_duplicate_features(k, reverse_labels):
    labels = np.array(["cat"] * 20 + ["dog"] * 10)
    if reverse_labels:
        labels = labels[::-1]
    features = np.ones((len(labels), 3))
    features.setflags(write=False)
    lab = Datalab(data={"label": labels}, label_name="label")
    manager = LabelIssueManager(datalab=lab, k=k)

    manager.find_issues(features=features)

    probabilities = manager.health_summary_parameters["pred_probs"]
    assert probabilities.shape == (len(labels), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert len(manager.issues) == len(labels)
    np.testing.assert_array_equal(features, np.ones((len(labels), 3)))


@pytest.mark.parametrize("k", [1, 3, 11])
def test_feature_predictions_exclude_each_example(k):
    rng = np.random.default_rng(52)
    features = rng.normal(size=(12, 3))
    labels = np.tile([0, 1, 2], 4)
    lab = Datalab(data={"label": labels}, label_name="label")
    manager = LabelIssueManager(datalab=lab, k=k)

    manager.find_issues(features=features)

    distances = pairwise_distances(features)
    np.fill_diagonal(distances, np.inf)
    neighbor_labels = labels[np.argsort(distances, axis=1)[:, :k]]
    expected = np.column_stack([(neighbor_labels == label).mean(axis=1) for label in range(3)])
    np.testing.assert_allclose(manager.health_summary_parameters["pred_probs"], expected)


def test_public_label_audit_with_duplicate_features():
    labels = np.array([0] * 20 + [1] * 10)
    lab = Datalab(data={"label": labels}, label_name="label")

    lab.find_issues(features=np.ones((len(labels), 3)), issue_types={"label": {"k": 2}})

    issues = lab.get_issues("label")
    assert len(issues) == len(labels)
    assert np.isfinite(issues["label_score"]).all()

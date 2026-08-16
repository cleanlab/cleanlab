import numpy as np
import pandas as pd
import pytest

from cleanlab.datalab.datalab import Datalab


def test_datalab_minimal_tabular_workflow():
    """Verify standard tabular Datalab workflow executes with only cleanlab[datalab] dependencies."""
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    n_classes = 3

    # Generate synthetic tabular dataset
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, size=n_samples)
    data = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    data["label"] = y

    # Generate plausible prediction probabilities
    logits = np.random.randn(n_samples, n_classes)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    pred_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # 1. Initialize Datalab
    lab = Datalab(data=data, label_name="label")
    assert lab.has_labels
    assert len(lab.data) == n_samples

    # 2. Run issue finding with features and pred_probs
    lab.find_issues(pred_probs=pred_probs, features=X)

    # 3. Verify core issue types are detected and populated
    issue_summary = lab.get_issue_summary()
    assert isinstance(issue_summary, pd.DataFrame)
    assert len(issue_summary) > 0

    label_issues = lab.get_issues("label")
    assert isinstance(label_issues, pd.DataFrame)
    assert len(label_issues) == n_samples
    assert "is_label_issue" in label_issues.columns

    outlier_issues = lab.get_issues("outlier")
    assert isinstance(outlier_issues, pd.DataFrame)
    assert len(outlier_issues) == n_samples

    near_duplicate_issues = lab.get_issues("near_duplicate")
    assert isinstance(near_duplicate_issues, pd.DataFrame)
    assert len(near_duplicate_issues) == n_samples

    # 4. Verify report generation runs cleanly without crashing
    report_output = lab.report()
    assert report_output is None or isinstance(report_output, str)

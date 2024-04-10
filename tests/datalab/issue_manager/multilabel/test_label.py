import numpy as np
import pandas as pd
import pytest

from cleanlab import Datalab
from cleanlab.datalab.internal.issue_manager.multilabel.label import MultilabelIssueManager
from cleanlab.internal.multilabel_utils import onehot2int


class TestLabelIssueManager:
    @pytest.fixture
    def data(self):
        # True labels of multilabel dataset
        np.random.seed(10)
        true_y = np.array(
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 0]]
        )
        pred_probs = np.full_like(true_y, fill_value=0.1, dtype=float)
        pred_probs[true_y == 1] = 0.9
        # Flip labels of some rows to add noise
        candidate_rows = np.where(true_y.sum(axis=1) < 3)[0]
        noisy_rows = np.random.choice(candidate_rows, 2, replace=False)
        noisy_y = true_y.copy()
        noisy_y[noisy_rows] = 1 - true_y[noisy_rows]
        labels = onehot2int(noisy_y)
        return {"labels": labels, "pred_probs": pred_probs}

    @pytest.fixture
    def issue_manager(self, data):
        labels = data["labels"]
        lab = Datalab({"labels": labels}, task="multilabel", label_name="labels")
        return MultilabelIssueManager(datalab=lab)

    def test_find_issues(self, data, issue_manager):
        """Test that the find_issues method works."""
        pred_probs = data["pred_probs"]
        issue_manager.find_issues(pred_probs=pred_probs)
        issues, summary, info = issue_manager.issues, issue_manager.summary, issue_manager.info
        assert isinstance(issues, pd.DataFrame), "Issues should be a dataframe"
        assert isinstance(summary, pd.DataFrame), "Summary should be a dataframe"
        assert summary["issue_type"].values[0] == "label"
        assert issues.index[issues["is_label_issue"]].tolist() == [3, 6]
        assert pytest.approx(summary["score"].values[0], abs=1e-3) == 0.6714
        assert isinstance(info, dict), "Info should be a dict"

        issue_manager.find_issues(pred_probs=pred_probs, frac_noise=0.5)
        issues = issue_manager.issues
        assert issues.index[issues["is_label_issue"]].tolist() == [3]

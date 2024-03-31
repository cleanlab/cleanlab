import numpy as np
import pytest

from cleanlab.internal.multiannotator_utils import assert_valid_inputs_multiannotator


def test_assert_valid_inputs_multiannotator_warnings():
    not_agree_labels = np.array([[2, 1, np.nan, 0], [1, 0, 2, np.nan]])
    with pytest.warns(UserWarning, match="do not agree"):
        assert_valid_inputs_multiannotator(not_agree_labels)

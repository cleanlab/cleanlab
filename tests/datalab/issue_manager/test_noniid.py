import numpy as np
import pytest
from cleanlab.datalab.issue_manager.noniid import simplified_kolmogorov_smirnov_test


@pytest.mark.parametrize(
    "neighbor_histogram, non_neighbor_histogram, expected_statistic",
    [
        # Test with equal histograms
        (
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            0.0,
        ),
        # Test with maximum difference in the first bin
        (
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.25, 0.5],
            1.0,
        ),
        # Test with maximum difference in the last bin
        (
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.25, 0.25, 0.0],
            0.25,
        ),
        # Test with arbitrary histograms
        (
            [0.2, 0.3, 0.4, 0.1],
            [0.1, 0.4, 0.25, 0.3],
            0.15,  # (0.2 -> 0.5 -> *0.9* -> 1.0) vs (0.1 -> 0.5 -> *0.75* -> 1.05
        ),
    ],
    ids=[
        "equal_histograms",
        "maximum_difference_in_first_bin",
        "maximum_difference_in_last_bin",
        "arbitrary_histograms",
    ],
)
def test_simplified_kolmogorov_smirnov_test(
    neighbor_histogram, non_neighbor_histogram, expected_statistic
):
    nh = np.array(neighbor_histogram)
    nnh = np.array(non_neighbor_histogram)
    statistic = simplified_kolmogorov_smirnov_test(nh, nnh)
    np.testing.assert_almost_equal(statistic, expected_statistic)

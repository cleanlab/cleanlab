import pytest
import numpy as np

from cleanlab.internal.numerics import softmax


class TestSoftmax:
    def test_basic_softmax(self):
        input_arr = np.array([1.0, 2.0, 3.0])
        output = softmax(input_arr)
        expected_output = np.array([0.09003057, 0.24472847, 0.66524096])
        assert np.isclose(np.sum(output), 1.0)
        assert np.allclose(output, expected_output)

    def test_temperature_effect(self):
        input_arr = np.array([1.0, 2.0, 3.0])
        output_high_temp = softmax(input_arr, temperature=5.0)
        output_low_temp = softmax(input_arr, temperature=0.1)

        expected_high_temp = np.array([0.2693075, 0.32893292, 0.40175958])
        expected_low_temp = np.array([2.06106005e-09, 4.53978686e-05, 9.99954600e-01])

        assert np.allclose(output_high_temp, expected_high_temp)
        assert np.allclose(output_low_temp, expected_low_temp)

    def test_axis(self):
        input_arr = np.array(
            [
                [1, 2, 3],  # unit step
                [4, 5, 6],  # unit step
                [7, 8, 10],  # non-unit step
            ]
        )
        output = softmax(input_arr, axis=1)

        expected_output = np.array(
            [
                [0.09003057, 0.24472847, 0.66524096],  # unit step
                [0.09003057, 0.24472847, 0.66524096],  # unit step
                [0.04201007, 0.1141952, 0.84379473],  # non-unit step
            ]
        )

        assert np.allclose(output, expected_output)

    @pytest.mark.parametrize(
        "input_arr, expected_output",
        [
            (np.array([1.0, 2.0, 3.0]) + 1000, np.array([0.09003057, 0.24472847, 0.66524096])),
            (np.array([1e3, 2e3, 3e3]), np.array([0, 0, 1])),
        ],
    )
    def test_shift(self, input_arr, expected_output):
        # Without shift, softmax overflows and gets a RuntimeWarning, but just returns nan
        with pytest.warns(RuntimeWarning):
            output_no_shift = softmax(input_arr, shift=False)
        assert np.isnan(output_no_shift).all()

        output_shift = softmax(input_arr, shift=True)
        assert np.allclose(output_shift, expected_output)

    @pytest.mark.parametrize(
        "input_arr, expected_output",
        [
            (np.array([0, -np.inf, -np.inf]), np.array([1.0, 0.0, 0.0])),
            (np.array([-np.inf, 0, 1]), np.array([0.0, 0.26894142, 0.73105858])),
        ],
    )
    def test_special_values(self, input_arr, expected_output):
        output = softmax(input_arr)
        assert np.allclose(output, expected_output)

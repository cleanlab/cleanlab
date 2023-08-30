from typing import Optional
import numpy as np


def softmax(
    x: np.ndarray, temperature: float = 1.0, axis: Optional[int] = None, shift: bool = False
) -> np.ndarray:
    """Softmax function.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    temperature : float
        Temperature of the softmax function.

    axis : Optional[int]
        Axis to apply the softmax function. If None, the softmax function is
        applied to all elements of the input array.

    shift : bool
        Whether to shift the input array before applying the softmax function.
        This is useful to avoid numerical issues when the input array contains
        large values, that could result in overflows when applying the exponential
        function.

    Returns
    -------
    np.ndarray
        Softmax function applied to the input array.
    """
    x = x / temperature
    if shift:
        x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

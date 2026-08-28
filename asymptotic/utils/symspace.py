import numpy as np
from typing import Optional, Union

def symlog_transform(
    x: float | np.ndarray,
    linthresh: float,
    linscale: float,
    base: float,
) -> Union[float, np.ndarray]:
    """
    Apply the symmetric logarithm transform to the input value(s).

    Parameters
    ----------
    x : float or ndarray
        Input value(s) to transform.
    linthresh : float
        The range within which the mapping is linear (from -linthresh to linthresh).
    linscale : float
        Stretches the linear range relative to the logarithmic range.
    base : float
        The base of the logarithm.

    Returns
    -------
    y : float or ndarray
        Transformed value(s).
    """
    x = np.asarray(x)
    sign = np.sign(x)
    abs_x = np.abs(x)
    y = np.empty_like(x, dtype=np.float64)
    mask = abs_x <= linthresh
    y[mask] = linscale * x[mask] / linthresh
    y[~mask] = sign[~mask] * (
        linscale + np.log(abs_x[~mask] / linthresh) / np.log(base)
    )
    return y

def symlog_inverse(
    y: float | np.ndarray,
    linthresh: float,
    linscale: float,
    base: float,
) -> Union[float, np.ndarray]:
    """
    Apply the inverse symmetric logarithm transform to the input value(s).

    Parameters
    ----------
    y : float or ndarray
        Input value(s) to inverse transform.
    linthresh : float
        The range within which the mapping is linear.
    linscale : float
        Stretches the linear range relative to the logarithmic range.
    base : float
        The base of the logarithm.

    Returns
    -------
    x : float or ndarray
        Inverse transformed value(s).
    """
    y = np.asarray(y)
    sign = np.sign(y)
    abs_y = np.abs(y)
    x = np.empty_like(y, dtype=np.float64)
    mask = abs_y <= linscale
    x[mask] = y[mask] * linthresh / linscale
    x[~mask] = sign[~mask] * linthresh * base ** (abs_y[~mask] - linscale)
    return x

def symlogspace(
    start: float,
    stop: float,
    num: int = 50,
    linthresh: float = 1.0,
    linscale: float = 1.0,
    base: float = 10.0,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    """
    Return numbers spaced evenly on a symmetric log scale.

    Parameters
    ----------
    start : float
        The starting value of the sequence.
    stop : float
        The final value of the sequence.
    num : int, optional
        Number of samples to generate. Default is 50.
    linthresh : float, optional
        The range within which the mapping is linear (from -linthresh to linthresh).
    linscale : float, optional
        Stretches the linear range relative to the logarithmic range. Default is 1.0.
    base : float, optional
        The base of the logarithm. Default is 10.0.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    samples : ndarray
        Numbers spaced evenly on a symmetric log scale.

    Examples
    --------
    >>> symlogspace(-100, 100, num=5)
    array([-100.        ,   -3.16227766,    0.        ,    3.16227766,
           100.        ])
    """
    y_start = symlog_transform(start, linthresh, linscale, base)
    y_stop = symlog_transform(stop, linthresh, linscale, base)
    y = np.linspace(y_start, y_stop, num=num, endpoint=True)
    x = symlog_inverse(y, linthresh, linscale, base)
    if dtype is not None:
        x = x.astype(dtype)
    return x

def symgeomspace(
    start: float,
    stop: float,
    num: int = 50,
    linthresh: float = 1.0,
    linscale: float = 1.0,
    base: float = 10.0,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    """
    Return numbers spaced evenly on a symmetric geometric progression.

    Parameters
    ----------
    start : float
        The starting value of the sequence.
    stop : float
        The final value of the sequence.
    num : int, optional
        Number of samples to generate. Default is 50.
    linthresh : float, optional
        The range within which the mapping is linear (from -linthresh to linthresh).
    linscale : float, optional
        Stretches the linear range relative to the logarithmic range. Default is 1.0.
        Must be positive.
    base : float, optional
        The base of the logarithm. Default is 10.0. Must be positive.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    samples : ndarray
        Numbers spaced evenly on a symmetric geometric progression.

    Examples
    --------
    >>> symgeomspace(-100, 100, num=5)
    array([-100.        ,   -3.16227766,    0.        ,    3.16227766,
           100.        ])
    """
    if num < 2:
        return np.array([start], dtype=dtype)

    # Apply symmetric logarithmic transform
    y_start = symlog_transform(start, linthresh, linscale, base)
    y_stop = symlog_transform(stop, linthresh, linscale, base)

    # Shift y values to be strictly positive
    min_y = min(y_start, y_stop)
    shift = -min_y + 1 if min_y <= 0 else 0
    y_start_shifted = y_start + shift
    y_stop_shifted = y_stop + shift

    # Use geomspace in the shifted space
    y_shifted = np.geomspace(
        y_start_shifted, y_stop_shifted, num=num, endpoint=True
    )

    # Shift back and apply inverse transform
    y = y_shifted - shift
    x = symlog_inverse(y, linthresh, linscale, base)

    if dtype is not None:
        x = x.astype(dtype)

    return x
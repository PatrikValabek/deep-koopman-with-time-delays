from typing import Literal

import numpy as np
import pandas as pd


def normalize(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def hankel(
    X: np.ndarray | pd.DataFrame,
    hn: int,
    return_partial: bool | Literal["copy"] = "copy",
) -> np.ndarray | pd.DataFrame:
    """Create a Hankel matrix from a given input array.

    Args:
        X (np.ndarray): The input array.
        hn (int): The number of columns in the Hankel matrix.
        cut_rollover (bool, optional): Whether to cut the rollover part of the Hankel matrix. Defaults to True.

    Returns:
        np.ndarray: The Hankel matrix.

    TODO:
        - [ ] Add support for 2D arrays.

    Example:
    >>> X = np.array([1., 2., 3., 4., 5.])
    >>> hankel(X, 3)
    array([[1., 1., 1.],
           [1., 1., 2.],
           [1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.]])
    >>> hankel(X, 3, return_partial=False)
    array([[1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.]])
    >>> X = np.array([[1., 2., 3., 4., 5.], [9., 8., 7., 6., 5.]]).T
    >>> hankel(X, 3, return_partial=True)
    array([[nan, nan, nan, nan,  1.,  9.],
           [nan, nan,  1.,  9.,  2.,  8.],
           [ 1.,  9.,  2.,  8.,  3.,  7.],
           [ 2.,  8.,  3.,  7.,  4.,  6.],
           [ 3.,  7.,  4.,  6.,  5.,  5.]])
    """
    if isinstance(X, pd.DataFrame):
        feature_names_in_ = X.columns
        X = X.values
    else:
        feature_names_in_ = None
    if len(X.shape) > 1:
        n = X.shape[1]
    else:
        n = 1
    if hn <= 1:
        return X
    hX = np.empty((X.shape[0], hn * n))
    X = np.roll(X, hn - 1, axis=0)
    for i in range(0, hn * n, n):
        hX[:, i : i + n] = X if len(X.shape) > 1 else X.reshape(-1, 1)
        if return_partial == "copy" and i / n < hn - 1:
            hX[: hn - int(i / n) - 1, i : i + n] = hX[
                hn - int(i / n) - 1, i : i + n
            ]
        elif return_partial and i / n < hn - 1:
            hX[: hn - int(i / n) - 1, i : i + n] = np.nan
        X = np.roll(X, -1, axis=0)
    if not return_partial:
        hX = hX[hn - 1 :]
    if feature_names_in_ is not None:
        return pd.DataFrame(
            hX,
            columns=[f"{f}_{i}" for i in range(hn) for f in feature_names_in_],
        )
    return hX

import itertools
from typing import Literal

import numpy as np
import pandas as pd


def normalize(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def hankel(
    X: np.ndarray | pd.DataFrame,
    hn: int,
    step: int = 1,
    return_partial: bool | Literal["copy"] = "copy",
) -> np.ndarray | pd.DataFrame:
    """Create a Hankel matrix from a given input array.

    Args:
        X (np.ndarray): The input array.
        hn (int): The number of columns in the Hankel matrix.
        step (int, optional): The step size for the delays. Defaults to 1.
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
    >>> X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [9.0, 8.0, 7.0, 6.0, 5.0]]).T
    >>> hankel(X, 3, 2, return_partial=True)
    array([[nan, nan, nan, nan,  3.,  7.],
           [nan, nan,  2.,  8.,  4.,  6.],
           [ 1.,  9.,  3.,  7.,  5.,  5.],
           [ 2.,  8.,  4.,  6.,  1.,  9.],
           [ 3.,  7.,  5.,  5.,  2.,  8.]])
    """
    if hn <= 1:
        return X

    if isinstance(X, pd.DataFrame):
        feature_names_in_ = X.columns
        index_in_ = X.index
        X = X.values
    else:
        feature_names_in_ = None

    if len(X.shape) > 1:
        n = X.shape[1]
    else:
        n = 1

    hX = np.empty((X.shape[0], hn * n))
    # Roll forth so that the last hankel columns are the start of the array
    X = np.roll(X, hn - 1, axis=0)
    for i in range(0, hn * n, n):
        hX[:, i : i + n] = X if len(X.shape) > 1 else X.reshape(-1, 1)
        if return_partial == "copy" and i / n < hn - 1:
            hX[: hn - int(i / n) - 1, i : i + n] = hX[
                hn - int(i / n) - 1, i : i + n
            ]
        elif return_partial and i / n < hn - 1:
            hX[: hn - int(i / n) - 1, i : i + n] = np.nan
        X = np.roll(X, -step, axis=0)
    if not return_partial:
        hX = hX[hn - 1 :]
    if feature_names_in_ is not None:
        return pd.DataFrame(
            hX,
            columns=[f"{f}_{i}" for i in range(hn) for f in feature_names_in_],
            index=index_in_,
        )
    return hX


def polynomial_extension(df: np.ndarray | pd.DataFrame, degree):
    if isinstance(df, pd.DataFrame):
        poly_features = pd.DataFrame()

        # Iterate over the combinations of columns up to the specified degree
        for d in range(2, degree + 1):
            for combination in itertools.combinations_with_replacement(
                df.columns, d
            ):
                col_name = "*".join(str(c) for c in combination)

                poly_features[col_name] = df[list(combination)].prod(axis=1)
    else:
        poly_features = np.empty((df.shape[0], 0))
        for d in range(2, degree + 1):
            for combination in itertools.combinations_with_replacement(
                range(df.shape[1]), d
            ):
                col_name = "*".join(str(c) for c in combination)
                poly_features = np.hstack(
                    (
                        poly_features,
                        df[:, list(combination)].prod(axis=1).reshape(-1, 1),
                    )
                )
    return poly_features

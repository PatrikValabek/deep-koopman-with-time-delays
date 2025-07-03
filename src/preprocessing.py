import itertools
from typing import Literal, overload

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.utils.validation import check_is_fitted

ErrorHandling = Literal["raise", "warn", "print", "ignore"]


# Create a wrapper around StandardScaler that preserves DataFrame structure
class StandardScaler(SklearnStandardScaler):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.values
        return super().fit(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.values
            transformed_array = super().fit_transform(X, y, **fit_params)
            return pd.DataFrame(
                transformed_array, index=X.index, columns=X.columns
            )
        return super().fit_transform(X, y, **fit_params)

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            transformed_array = super().transform(X)
            return pd.DataFrame(
                transformed_array, index=X.index, columns=X.columns
            )
        return super().transform(X)


# Create a wrapper around StandardScaler that preserves DataFrame structure
class MinMaxScaler(SklearnMinMaxScaler):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.values
        return super().fit(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.values
            transformed_array = super().fit_transform(X, y, **fit_params)
            return pd.DataFrame(
                transformed_array, index=X.index, columns=X.columns
            )
        return super().fit_transform(X, y, **fit_params)

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            transformed_array = super().transform(X)
            return pd.DataFrame(
                transformed_array, index=X.index, columns=X.columns
            )
        return super().transform(X)


class Hankel(BaseEstimator, TransformerMixin):
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
    >>> hankel(X, 3, return_partial=True)
    array([[nan, nan,  1.],
           [nan,  1.,  2.],
           [ 1.,  2.,  3.],
           [ 2.,  3.,  4.],
           [ 3.,  4.,  5.]])
    >>> hankel(X, 3, return_partial="copy")
    array([[1., 1., 1.],
           [1., 1., 2.],
           [1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.]])
    >>> X = np.array([[1., 2., 3., 4., 5.], [9., 8., 7., 6., 5.]]).T
    >>> hankel(X, 3, return_partial=False)
    array([[1., 9., 2., 8., 3., 7.],
           [2., 8., 3., 7., 4., 6.],
           [3., 7., 4., 6., 5., 5.]])
    >>> X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [9.0, 8.0, 7.0, 6.0, 5.0]]).T
    >>> hankel(X, 3, 2, return_partial=False)
    array([[1., 9., 3., 7., 5., 5.],
           [2., 8., 4., 6., 1., 9.],
           [3., 7., 5., 5., 2., 8.]])
    """

    def __init__(
        self,
        hn: int = 1,
        step: int = 1,
        return_partial: bool | Literal["copy"] = "copy",
    ):
        self.hn = hn
        self.step = step
        self.return_partial = return_partial

        self.index_in_: pd.Index
        self.feature_names_in_: list[str]

    def fit(self, X, y=None):
        """Fit the transformer.

        This is a no-op for this transformer.

        Args:
            X: The input data.
            y: Ignored.

        Returns:
            self
        """
        return self

    @overload
    def transform(self, X: np.ndarray) -> np.ndarray: ...
    @overload
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...

    def transform(self, X):
        """Transform X into a Hankel matrix.

        Args:
            X: The input data.

        Returns:
            The Hankel matrix.
        """
        if self.hn <= 1:
            return X

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns
            self.index_in_ = X.index
            X = X.values
        else:
            self.feature_names_in_ = None

        if len(X.shape) > 1:
            n = X.shape[1]
        else:
            n = 1

        hX = np.empty((X.shape[0], self.hn * n))
        # Roll forth so that the last hankel columns are the start of the array
        X = np.roll(X, self.hn - 1, axis=0)
        for i in range(0, self.hn * n, n):
            hX[:, i : i + n] = X if len(X.shape) > 1 else X.reshape(-1, 1)
            if self.return_partial == "copy" and i / n < self.hn - 1:
                hX[: self.hn - int(i / n) - 1, i : i + n] = hX[
                    self.hn - int(i / n) - 1, i : i + n
                ]
            elif self.return_partial and i / n < self.hn - 1:
                hX[: self.hn - int(i / n) - 1, i : i + n] = np.nan
            X = np.roll(X, -self.step, axis=0)
        if not self.return_partial:
            hX = hX[self.hn - 1 :]
        if self.feature_names_in_ is not None:
            return pd.DataFrame(
                hX,
                columns=[
                    f"{f} ({i})"
                    for i in range(self.hn)
                    for f in self.feature_names_in_
                ],
                index=self.index_in_[-len(hX) :]
                if len(hX) < len(self.index_in_)
                else self.index_in_,
            )
        return hX

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Inverse transform the Hankel matrix.

        Args:
            X: The Hankel matrix.

        Returns:
            The original matrix.

        Examples:
            >>> import numpy as np
            >>> from .preprocessing import Hankel
            >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            >>> h = Hankel(hn=2)
            >>> hX = h.transform(X)
            >>> hX
            array([[1., 2., 1., 2.],
                   [1., 2., 3., 4.],
                   [3., 4., 5., 6.],
                   [5., 6., 7., 8.]])
            >>> h = Hankel(hn=2, return_partial=False)
            >>> hX = h.transform(X)
            >>> hX
            array([[1., 2., 3., 4.],
                   [3., 4., 5., 6.],
                   [5., 6., 7., 8.]])
            >>> h.inverse_transform(hX)
            array([[1., 2.],
                   [3., 4.],
                   [5., 6.],
                   [7., 8.]])

            >>> import pandas as pd
            >>> X_df = pd.DataFrame(X, columns=['A', 'B'])
            >>> hX_df = h.transform(X_df)
            >>> hX_df.columns
            Index(['A (0)', 'B (0)', 'A (1)', 'B (1)'], dtype='object')
            >>> h.inverse_transform(hX_df)
                 A    B
            0  1.0  2.0
            1  3.0  4.0
            2  5.0  6.0
            3  7.0  8.0
        """
        if self.hn <= 1:
            return X

        if isinstance(X, pd.DataFrame):
            X = X.values

        n = X.shape[1] // self.hn
        rows = (
            X.shape[0] + self.hn - 1 if not self.return_partial else X.shape[0]
        )

        if not self.return_partial:
            # Initialize the original matrix
            original = np.zeros((rows, n))

            # For each column in the original matrix
            for i in range(n):
                # Extract the corresponding columns from the Hankel matrix
                hankel_cols = X[:, i::n]

                # Reconstruct the original column
                # If we didn't return partial data, we need to add rows at the beginning
                for j in range(self.hn):
                    original[j : rows - (self.hn - j - 1), i] += hankel_cols[
                        :, j
                    ]
            # Normalize by the number of times each element was added
            counts = np.zeros((rows, n))
            for j in range(self.hn):
                if not self.return_partial:
                    counts[j : rows - (self.hn - j - 1), :] += 1
                else:
                    valid_rows = rows - j
                    counts[j : j + valid_rows, :] += 1

            original = original / counts
        else:
            original = X[:, -n:]

        if self.feature_names_in_ is not None:
            return pd.DataFrame(
                original,
                columns=self.feature_names_in_,
                index=self.index_in_ if rows == len(self.index_in_) else None,
            )
        return original

    def get_feature_names_out(self, input_features=None):
        input_features_ = (
            self.feature_names_in_
            if input_features is None
            else input_features
        )
        if self.hn <= 1:
            return input_features_
        return [f"{f} ({i})" for i in range(self.hn) for f in input_features_]


def hankel(
    X: np.ndarray | pd.DataFrame,
    hn: int,
    step: int = 1,
    return_partial: bool | Literal["copy"] = "copy",
) -> np.ndarray | pd.DataFrame:
    """Create a Hankel matrix from a given input array.

    This is a convenience function that uses the Hankel transformer.

    Args:
        X (np.ndarray): The input array.
        hn (int): The number of columns in the Hankel matrix.
        step (int, optional): The step size for the delays. Defaults to 1.
        return_partial (bool | Literal["copy"], optional): How to handle partial data. Defaults to "copy".

    Returns:
        np.ndarray | pd.DataFrame: The Hankel matrix.
    """
    return Hankel(hn=hn, step=step, return_partial=return_partial).transform(X)


@overload
def time_shift(
    X: np.ndarray, shift: int, step: int = 1, cut_rollover: bool = True
) -> np.ndarray: ...
@overload
def time_shift(
    X: pd.DataFrame, shift: int, step: int = 1, cut_rollover: bool = True
) -> pd.DataFrame: ...


def time_shift(
    X: np.ndarray | pd.DataFrame,
    shift: int,
    step: int = 1,
    cut_rollover: bool = True,
) -> np.ndarray | pd.DataFrame:
    """Shift the columns of a matrix by a given number of steps.

    Args:
        X (np.array): The input matrix.
        shift (int): The number of steps to shift the columns.

    Returns:
        np.array: The shifted matrix.

    Example:
    >>> X = np.array([1., 2., 3., 4., 5.])
    >>> time_shift(X, 3)
    array([[1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.]])
    >>> X = np.array([1., 2., 3., 4., 5.])
    >>> time_shift(X, 3, cut_rollover=False)
    array([[1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.],
           [4., 5., 1.],
           [5., 1., 2.]])
    >>> X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [9.0, 8.0, 7.0, 6.0, 5.0]]).T
    >>> time_shift(X, 3)
    array([[1., 9., 2., 8., 3., 7.],
           [2., 8., 3., 7., 4., 6.],
           [3., 7., 4., 6., 5., 5.]])
    >>> time_shift(X, 3, 2, cut_rollover=False)
    array([[1., 9., 3., 7., 5., 5.],
           [2., 8., 4., 6., 1., 9.],
           [3., 7., 5., 5., 2., 8.],
           [4., 6., 1., 9., 3., 7.],
           [5., 5., 2., 8., 4., 6.]])
    """
    if shift <= 1:
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

    hX = np.empty((X.shape[0], shift * n))
    for i in range(0, shift * n, n):
        hX[:, i : i + n] = X if len(X.shape) > 1 else X.reshape(-1, 1)
        X = np.roll(X, -step, axis=0)
    if cut_rollover:
        hX = hX[: -shift + 1]
    if feature_names_in_ is not None:
        return pd.DataFrame(
            hX,
            columns=[
                f"{f}_{i}" for i in range(shift) for f in feature_names_in_
            ],
            index=index_in_[shift - 1 :],
        )
    return hX


class PolynomialFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer that creates polynomial features from input data.

    This transformer creates polynomial features up to the specified degree.
    For example, if the input features are [a, b] and degree=2, the transformer
    will create features [a*a, a*b, b*b].

    Parameters
    ----------
    degree : int, default=2
        The degree of polynomial features to create.

    Attributes
    ----------
    feature_names_out_ : list
        Names of the output features.
    n_features_in_ : int
        Number of features in the input data.
    feature_names_in_ : list or None
        Names of the input features if available.
    """

    def __init__(self, degree=2, interaction: bool = True):
        self.degree = degree
        self.interaction = interaction
        self.feature_names_out_ = None
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Fit the transformer by storing input feature names.

        Parameters
        ----------
        X : array-like or pandas DataFrame of shape (n_samples, n_features)
            The input data.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = [
                f"x_{i}" for i in range(self.n_features_in_)
            ]

        # Generate feature names for output
        self.feature_names_out_ = self.get_feature_names_out(
            self.feature_names_in_
        )
        return self

    def transform(self, X):
        """
        Create polynomial features from the input data.

        Parameters
        ----------
        X : array-like or pandas DataFrame of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_poly : pandas DataFrame or numpy array
            The input data with polynomial features added.

        Examples
        --------
        >>> X = np.array([[1, 3], [2, 4]])
        >>> transformer = PolynomialFeatures(degree=2, interaction=True)
        >>> transformer.fit_transform(X)
        array([[ 1.,  3.,  1.,  3.,  9.],
               [ 2.,  4.,  4.,  8., 16.]])
        >>> transformer = PolynomialFeatures(degree=2, interaction=False)
        >>> transformer.fit_transform(X)
        array([[ 1.,  3.,  1.,  9.],
               [ 2.,  4.,  4., 16.]])
        >>> X = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> transformer = PolynomialFeatures(degree=2, interaction=True)
        >>> transformer.fit_transform(X)
           a  b  a*a  a*b  b*b
        0  1  3    1    3    9
        1  2  4    4    8   16
        >>> transformer = PolynomialFeatures(degree=2, interaction=False)
        >>> transformer.fit_transform(X)
           a  b  a*a  b*b
        0  1  3    1    9
        1  2  4    4   16
        """
        check_is_fitted(self, ["feature_names_out_"])

        if isinstance(X, pd.DataFrame):
            # Start with original columns
            poly_features = X.copy()

            # Iterate over the combinations of columns up to the specified degree
            for d in range(2, self.degree + 1):
                if self.interaction:
                    # Include all combinations when interaction is True
                    for col_comb in itertools.combinations_with_replacement(
                        X.columns, d
                    ):
                        col_name = "*".join(col_comb)
                        poly_features[col_name] = X[list(col_comb)].prod(
                            axis=1
                        )
                else:
                    # Only include self-multiplications when interaction is False
                    for col in X.columns:
                        col_name = "*".join([col] * d)
                        poly_features[col_name] = X[col] ** d

            return poly_features
        else:
            # Start with original columns
            poly_features = np.empty((
                X.shape[0],
                len(self.feature_names_out_),
            ))

            poly_features[:, : X.shape[1]] = X
            col_idx = X.shape[1]

            for d in range(2, self.degree + 1):
                if self.interaction:
                    # Include all combinations when interaction is True
                    for col_comb in itertools.combinations_with_replacement(
                        range(X.shape[1]), d
                    ):
                        poly_features[:, col_idx] = X[:, list(col_comb)].prod(
                            axis=1
                        )
                        col_idx += 1
                else:
                    # Only include self-multiplications when interaction is False
                    for col in range(X.shape[1]):
                        poly_features[:, col_idx] = X[:, col] ** d
                        col_idx += 1

            return poly_features

    def inverse_transform(self, X):
        """
        Inverse transform for polynomial features when interaction=False.
        For interaction=True, the transformation is not invertible.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_out)
            The polynomial features.

        Returns
        -------
        X_inv : array-like of shape (n_samples, n_features_in)
            The inverse transformed features.
            If interaction=True, raises NotImplementedError.

        Examples
        --------
        >>> transformer = PolynomialFeatures(degree=2, interaction=False)
        >>> X = np.array([[1, 3], [2, 4]])
        >>> transformer.fit(X)
        PolynomialFeatures(...)
        >>> X_poly = transformer.transform(X)
        >>> X_inv = transformer.inverse_transform(X_poly)
        >>> np.allclose(X, X_inv)
        True
        """
        check_is_fitted(self, ["feature_names_out_"])

        if self.interaction:
            raise NotImplementedError(
                "Inverse transform is not implemented for polynomial features with interaction=True"
            )

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Initialize output array
        n_samples = X.shape[0]
        n_features = len(self.feature_names_in_)
        X_inv = np.zeros((n_samples, n_features))

        # For non-interaction case, we can simply take the d-th root of the d-th power terms
        for i, feature in enumerate(self.feature_names_in_):
            # Original feature is in the first n_features columns
            X_inv[:, i] = X[:, i]

            # Handle higher degree terms
            for d in range(2, self.degree + 1):
                col_name = "*".join([feature] * d)
                if col_name in self.feature_names_out_:
                    col_idx = np.where(self.feature_names_out_ == col_name)[0][
                        0
                    ]
                    # Take the d-th root of the d-th power term
                    X_inv[:, i] = np.power(X[:, col_idx], 1 / d)

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_inv, columns=self.feature_names_in_)
        return X_inv

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features. If None, then feature names from the constructor are used.

        Returns
        -------
        feature_names_out : list of str
            Output feature names.

        Examples
        --------
        >>> X = np.array([[1, 3], [2, 4]])
        >>> transformer = PolynomialFeatures(degree=2, interaction=True)
        >>> transformer.fit_transform(X)
        array([[ 1.,  3.,  1.,  3.,  9.],
               [ 2.,  4.,  4.,  8., 16.]])
        >>> transformer.get_feature_names_out().tolist()
        ['x_0', 'x_1', 'x_0^2', 'x_0 x_1', 'x_1^2']
        """
        check_is_fitted(self, ["feature_names_out_"])

        if input_features is not None:
            # Regenerate feature names based on provided input_features
            feature_names_out = [*input_features]
            for d in range(2, self.degree + 1):
                if self.interaction:
                    # Generate all polynomial terms
                    for col_comb in itertools.combinations_with_replacement(
                        input_features, d
                    ):
                        # Count occurrences of each feature in the combination
                        counts = {}
                        for col in col_comb:
                            counts[col] = counts.get(col, 0) + 1

                        # Build feature name with powers
                        name_parts = []
                        for col, count in counts.items():
                            if count == 1:
                                name_parts.append(col)
                            else:
                                name_parts.append(f"{col}^{count}")
                        feature_names_out.append(" ".join(name_parts))
                else:
                    # Generate no interaction terms
                    for col in input_features:
                        feature_names_out.append(f"{col}^{d}")
            return np.array(feature_names_out)

        return self.feature_names_out_

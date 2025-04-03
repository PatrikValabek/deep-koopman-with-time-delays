"""Dynamic Mode Decomposition (DMD) in scikkit-learn API.

This module contains the implementation of the Online DMD, Windowed DMD,
and DMD with Control algorithm. It is based on the paper by Zhang et al.
[^1] and implementation of authors available at [GitHub](https://github.com/haozhg/odmd).
However, this implementation provides a more flexible interface aligned with
River API covers and separates update and revert methods in Windowed DMD.

References:
    [^1]: Schmid, P. (2022). Dynamic Mode Decomposition and Its Variants. 54(1), pp.225-254. doi:[10.1146/annurev-fluid-030121-015835](https://doi.org/10.1146/annurev-fluid-030121-015835).
"""

from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, check_is_fitted
from sklearn.utils.validation import check_array, validate_data

from .utils import get_default_rank


def _sort_svd(U, S, Vt):
    """Sort the singular value decomposition in descending order.

    As sparse SVD does not guarantee the order of the singular values, we
    need to sort the singular value decomposition in descending order.
    """
    sort_idx = np.argsort(S)[::-1]
    if not np.array_equal(sort_idx, range(len(S))):
        S = S[sort_idx]
        U = U[:, sort_idx]
        Vt = Vt[sort_idx, :]
    return U, S, Vt


def _truncate_svd(U, S, Vt, n_components):
    """Truncate the singular value decomposition to the n components.

    Full SVD returns the full matrices U, S, and V in correct order. If the
    result acqisition is faster than sparse SVD, we combine the results of
    full SVD with truncation.
    """
    U = U[:, :n_components]
    S = S[:n_components]
    Vt = Vt[:n_components, :]
    return U, S, Vt


class DMD(BaseEstimator):
    """Class for Dynamic Mode Decomposition (DMD) model.

    Args:
        r: Number of modes to keep. If 0 (default), optimal rank is computed. If -1, all modes are kept.

    Attributes:
        m: Number of features (variables).
        n: Number of time steps (snapshots).
        feature_names_in_: list of feature names. Used for pd.DataFrame inputs.
        Lambda: Eigenvalues of the Koopman matrix.
        Phi: Eigenfunctions of the Koopman operator (Modal structures)
        A_bar: Low-rank approximation of the Koopman operator (Rayleigh quotient matrix).
        A: Koopman operator.
        C: Discrete temporal dynamics matrix (Vandermonde matrix).
        xi: Amlitudes of the singular values of the input matrix.
        _Y: Data snaphot from time step 2 to n (for xi comp.).

    References:
        [^1]: Schmid, P. (2022). Dynamic Mode Decomposition and Its Variants. 54(1), pp.225-254. doi:[10.1146/annurev-fluid-030121-015835](https://doi.org/10.1146/annurev-fluid-030121-015835).
    """

    def __init__(self, r: int = 0):
        self.r = r
        self.n_features_in_: int
        self.n_samples_: int
        self.feature_names_in_: list[str]
        self.Lambda: np.ndarray
        self._Phi: np.ndarray | None = None
        self._A_bar: np.ndarray | None = None
        self._A: np.ndarray | None = None
        self._Y: np.ndarray

        # Properties to be reset at each update
        self._eig: tuple[np.ndarray, np.ndarray] | None = None
        self._modes: np.ndarray | None = None
        self._xi: np.ndarray | None = None
        self._poles: np.ndarray | None = None

    @property
    def A(self) -> np.ndarray:
        if self._A is None:
            self._A = (
                self._Y.T
                @ self._v
                @ self._s_inv
                @ self._ut[:, : self.n_features_in_]
            )

            # Ensure stability by scaling eigenvalues if needed
            eigenvalues, eigenvectors = np.linalg.eig(self._A)
            unstable_indices = np.abs(eigenvalues) > 1.0

            if np.any(unstable_indices):
                # Scale unstable eigenvalues to have magnitude 1 (on the unit circle)
                eigenvalues[unstable_indices] = eigenvalues[
                    unstable_indices
                ] / np.abs(eigenvalues[unstable_indices])
                # Reconstruct A with stabilized eigenvalues
                # Ensure A is real (might have small imaginary components due to numerical errors)
                self._A = np.real(
                    eigenvectors
                    @ np.diag(eigenvalues)
                    @ np.linalg.inv(eigenvectors)
                )

        return self._A  # type: ignore

    @property
    def A_bar(self) -> np.ndarray:
        if self._A_bar is None:
            # Compute the low-rank approximation of Koopman matrix
            self._A_bar = (
                self._ut[:, : self.n_features_in_]
                @ self._Y.T
                @ self._v
                @ self._s_inv
            )
            # In case of DMDwC we are only interested in part corresponding to states
            self._A_bar = self._A_bar[
                : self.n_features_in_, : self.n_features_in_
            ]
        return self._A_bar

    @property
    def C(self) -> np.ndarray:
        """Compute Discrete temporal dynamics matrix (Vandermonde matrix)."""
        return np.vander(self.Lambda, self.n_samples_, increasing=True)

    @property
    def eig(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute and return DMD eigenvalues and DMD modes at current step"""
        if self._eig is None:
            Lambda, Phi = np.linalg.eig(self.A_bar)

            sort_idx = np.argsort(Lambda)[::-1]
            if not np.array_equal(sort_idx, range(len(Lambda))):
                Lambda = Lambda[sort_idx]
                Phi = Phi[:, sort_idx]
            self._eig = Lambda, Phi
        return self._eig

    @property
    def modes(self) -> np.ndarray:
        """Reconstruct high dimensional discrete-time DMD modes"""
        if self._modes is None:
            _, Phi_comp = self.eig
            if self.r < self.n_features_in_:
                # Exact DMD modes (Tu et al. (2016))
                # self._Y.T @ self._svd._Vt.T is increasingly more computationally expensive without rolling
                # self._modes = (
                #     self._Y.T
                #     @ self._svd._Vt.T  # sign may change if sparse SVD is used
                #     @  self._s_inv
                #     @ Phi_comp  # sign may change if sparse EIG is used
                # )

                # Projected DMD modes (Schmid (2010)) - faster, not guaranteed
                # self._modes = self._svd._U @ Phi_comp
                # This regularization works much better than the above
                #  if high variance in svs of X
                self._modes = (
                    self._u[: self.n_features_in_] @ self._s_inv @ Phi_comp
                )
            else:
                self._modes = Phi_comp
        return self._modes

    @property
    def Phi(self) -> np.ndarray:
        if self._Phi is None:
            # Compute the coefficient matrix
            # self._Phi = (
            #     Y
            #     @ self._v[:, : self.n_features_in_]
            #     @  self._s_inv[: self.n_features_in_, : self.n_features_in_]
            #     @ W
            # )
            # self._Phi = u_ @ W
            self._Phi = (
                self._u[: self.n_features_in_, : self.n_features_in_]
                @ self._s_inv[: self.n_features_in_, : self.n_features_in_]
                @ self.eig[1]
            )
        return self._Phi

    @property
    def xi(self) -> np.ndarray:
        if self._xi is None:
            # r = self.r if self.r > 0 else self.n_features_in_
            # _, Phi = self.eig
            # xi = self._Phi.conj().T @ self._Y @ np.linalg.pinv(self.C)

            # from scipy.optimize import minimize

            # def objective_function(x):
            #     return np.linalg.norm(
            #         self._Y[:r] - Phi @ np.diag(x) @ self.C, "fro"
            #     ) + 0.5 * np.linalg.norm(x, 1)

            # # Minimize the objective function
            # xi = minimize(objective_function, np.ones(r)).x
            xi = np.linalg.lstsq(
                self.modes,
                self._Y.T,
                rcond=None,
            )[0]
            _xi = np.abs(xi)
            self._xi = _xi
            return _xi
        return self._xi

    @property
    def poles(self) -> np.ndarray:
        """Compute and return DMD poles."""
        if self._poles is None:
            self._poles = np.linalg.eigvals(self.A)
        return self._poles

    def is_stable(self) -> bool:
        """Check stability of A matrix."""
        eigenvalues = self.poles
        return bool(np.all(np.abs(eigenvalues) <= 1))

    def _fit(self, X: np.ndarray, Y=None):
        # Perform singular value decomposition on X
        if self.r == 0:
            self.r = get_default_rank(X)
        elif self.r == -1:
            self.r = self.n_features_in_
        # # Truncate the singular value matrices
        X = np.asarray(X, dtype=np.float64)
        self._v, _s, self._ut = np.linalg.svd(X, full_matrices=False)
        self._v, _s, self._ut = _truncate_svd(self._v, _s, self._ut, self.r)
        self._u = self._ut.conj().T
        _vt = self._v.conj().T
        self._s_inv = np.diag(np.reciprocal(_s))

        # This initializes the variables which would be initialized on the first call of predict
        self.A_bar
        self.A
        self._Phi

    def fit(self, X: pd.DataFrame | np.ndarray, Y=None) -> "DMD":
        """
        Fit the DMD model to the input X.

        Args:
            X: Input X matrix of shape (n, m), where m is the number of variables and n is the number of time steps.
            Y: The output snapshot matrix of shape (n, m).

        """
        # Build X matrices
        X_ = validate_data(
            self,
            X,
            reset=True,
            **dict(ensure_2d=True, ensure_min_samples=2),
        )
        self._Y = X_[1:, :]
        X_ = X_[:-1, :]
        self.n_samples_ = self._Y.shape[0]

        self._fit(X_)

        return self

    def predict(
        self,
        x: np.ndarray,
        forecast: int = 1,
    ) -> np.ndarray:
        """
        Predict future values using the trained DMD model.

        Args:
        x: numpy.ndarray of shape (m,)
        forecast: int
            Number of steps to predict into the future.

        Returns:
            predictions: Predicted data matrix for the specified number of prediction steps.
        """
        check_is_fitted(self)
        if x.ndim != 2:
            if self.n_features_in_ == 1:
                x = x.reshape(-1, 1)
            else:
                x = x.reshape(1, -1)
        x = validate_data(self, x, reset=False)

        if x.shape[0] != 1:
            forecast = x.shape[0]

        mat = np.zeros((forecast + 1, self.n_features_in_))
        mat[0, :] = x[0, :]
        for s in range(1, forecast + 1):
            mat[s, :] = np.real(self.A @ mat[s - 1, :])
        return mat[1:, :]


class DMDc(DMD):
    """Dynamic Mode Decomposition with Control (DMDc).

    DMDc extends the standard DMD algorithm to incorporate control inputs, allowing
    for the identification of systems with external forcing or control.

    Args:
        p: Number of modes to keep for the state dynamics. If 0 (default), optimal rank is computed.
           If -1, all modes are kept.
        q: Number of modes to keep for the control input. If 0 (default), optimal rank is computed.
           If -1, all modes are kept.
        B: Control matrix. If None (default), it will be computed during fitting.

    Attributes:
        n_inputs_in_: Number of control input variables.
        input_names_in_: List of control input feature names (when using pandas DataFrame).
        B: Control matrix that maps control inputs to state changes.

    References:
        Proctor, J. L., Brunton, S. L., & Kutz, J. N. (2016). Dynamic mode decomposition
        with control. SIAM Journal on Applied Dynamical Systems, 15(1), 142-161.
        doi:10.1137/15M1013857
    """

    def __init__(
        self, p: int = 0, q: int = 0, B: Union[np.ndarray, None] = None
    ):
        super().__init__()
        self.p = p
        self.q = q
        self._B: np.ndarray | None = B
        self.n_inputs_in_: int = 0
        self.input_names_in_: list[str]
        self.known_B: bool = self.B is not None

    @property
    def B(self) -> np.ndarray | None:
        if self._B is None:
            if self.n_inputs_in_ == 0:
                return None
            else:
                self._B = (
                    self._Y.T
                    @ self._v
                    @ self._s_inv
                    @ self._ut[:, -self.n_inputs_in_ :]
                )
        return self._B

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        Y=None,
        U: Union[pd.DataFrame, np.ndarray, None] = None,
    ):
        if U is None:
            super().__init__(self.p + self.q)
            return super().fit(X)

        X_: np.ndarray = validate_data(
            self,
            X,
            reset=True,
            **dict(ensure_2d=True, ensure_min_samples=2),
        )
        if isinstance(U, pd.DataFrame):
            self.input_names_in_ = U.columns.tolist()
        U_: np.ndarray = check_array(U, ensure_2d=True, ensure_min_samples=2)

        if X_.shape[0] != U_.shape[0]:
            raise ValueError(
                "X and u must have the same number of time steps.\n"
                f"Got X: {X_.shape[0]}, U: {U_.shape[0]} instead"
            )

        Y = X_[1:, :]
        X_ = X_[:-1, :]
        U_ = U_[:-1, :]

        if not self.known_B:
            X_ = np.hstack((X_, U_))
            self._Y = Y
        else:
            # Subtract the effect of actuation
            self._Y = Y - self.B * U_

        self.n_samples_, self.n_inputs_in_ = U_.shape

        if self.p == 0:
            self.p = get_default_rank(self._Y)
        elif self.p == -1:
            self.p = self.n_features_in_
        if self.q == 0:
            self.q = get_default_rank(U_)
        elif self.q == -1:
            self.q = self.n_inputs_in_
        self.r = self.p + self.q
        self._fit(X_)

        self.B

        return self

    def predict(
        self,
        x: pd.DataFrame | np.ndarray,
        forecast: int = 1,
        U: Union[np.ndarray, None] = None,
    ) -> np.ndarray:
        """
        Predict future values using the trained DMD model.

        Args:
        - forecast: int
            Number of steps to predict into the future.

        Returns:
        - predictions: numpy.ndarray
            Predicted data matrix for the specified number of prediction steps.
        """
        check_is_fitted(self)
        if isinstance(x, np.ndarray) and x.ndim != 2:
            if self.n_features_in_ == 1:
                x = x.reshape(-1, 1)
            else:
                x = x.reshape(1, -1)
        x_ = validate_data(self, x, reset=False)

        if U is None:
            return super().predict(x_, forecast)

        U_ = check_array(U, ensure_2d=True)

        if forecast != 1 and U_.shape[0] != forecast:
            raise ValueError(
                "u must have forecast number of time steps.\n"
                f"Got U: {U_.shape[0]}, forecast: {forecast} instead."
            )

        mat = np.zeros((forecast + 1, self.n_features_in_))
        mat[0, :] = x_[0, :]
        for s in range(1, forecast + 1):
            action = np.real(self.B @ U_[s - 1, :])
            mat[s, :] = np.real(self.A @ mat[s - 1, :]) + action
        return mat[1:, :]


class DMDcRegressor(RegressorMixin, DMDc):
    """DMDc variant that follows scikit-learn's regressor interface.

    This class wraps DMDc to accept X as control inputs (U) and y as state variables (X)
    to match scikit-learn's API conventions.
    """

    def __init__(
        self, p: int = 0, q: int = 0, B: Union[np.ndarray, None] = None
    ):
        super().__init__(p=p, q=q, B=B)

    def fit(self, X, y):
        """Fit DMDc model following scikit-learn convention.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training control inputs (U in DMDc terminology)
        y : array-like of shape (n_samples, n_targets)
            Training state variables (X in DMDc terminology)

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return super().fit(y, U=X)

    def predict(self, X, y0=None):
        """Predict using DMDc model following scikit-learn convention.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Control inputs to predict for (U in DMDc terminology)
        y0 : array-like of shape (1, n_targets)
            Initial state variables to predict from (X in DMDc terminology). Used in multi-fidelity modeling to initialize the prediction from high-fidelity data points.

        Returns
        -------
        y_pred : array-like of shape (n_samples, n_targets)
            Predicted state variables
        """
        fitted_feature_names = getattr(self, "feature_names_in_", None)
        if y0 is None:
            if fitted_feature_names is not None:
                y0_init = pd.DataFrame(
                    self._Y[-1:], columns=fitted_feature_names
                )
            else:
                y0_init = self._Y[-1:]
        else:
            y0_init = y0
        return super().predict(y0_init, forecast=X.shape[0], U=X)

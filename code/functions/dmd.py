"""Dynamic Mode Decomposition (DMD) in scikkit-learn API.

This module contains the implementation of the Online DMD, Windowed DMD,
and DMD with Control algorithm. It is based on the paper by Zhang et al.
[^1] and implementation of authors available at [GitHub](https://github.com/haozhg/odmd).
However, this implementation provides a more flexible interface aligned with
River API covers and separates update and revert methods in Windowed DMD.

TODO:

    - [ ] Align design with (n, m) convention (currently (m, n)).

References:
    [^1]: Schmid, P. (2022). Dynamic Mode Decomposition and Its Variants. 54(1), pp.225-254. doi:[10.1146/annurev-fluid-030121-015835](https://doi.org/10.1146/annurev-fluid-030121-015835).
"""

from typing import Union

import numpy as np


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


class DMD:
    """Class for Dynamic Mode Decomposition (DMD) model.

    Args:
        r: Number of modes to keep. If 0 (default), all modes are kept.

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
        self.m: int
        self.n: int
        self.feature_names_in_: list[str]
        self.Lambda: np.ndarray
        self.Phi: np.ndarray
        self.A_bar: np.ndarray
        self.A: np.ndarray
        self._Y: np.ndarray

        # Properties to be reset at each update
        self._eig: tuple[np.ndarray, np.ndarray] | None = None
        self._modes: np.ndarray | None = None
        self._xi: np.ndarray | None = None

    @property
    def C(self) -> np.ndarray:
        """Compute Discrete temporal dynamics matrix (Vandermonde matrix)."""
        return np.vander(self.Lambda, self.n, increasing=True)

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
            if self.r < self.m:
                # Exact DMD modes (Tu et al. (2016))
                # self._Y.T @ self._svd._Vt.T is increasingly more computationally expensive without rolling
                # self._modes = (
                #     self._Y.T
                #     @ self._svd._Vt.T  # sign may change if sparse SVD is used
                #     @ np.diag(1 / self._svd._S)
                #     @ Phi_comp  # sign may change if sparse EIG is used
                # )

                # Projected DMD modes (Schmid (2010)) - faster, not guaranteed
                # self._modes = self._svd._U @ Phi_comp
                # This regularization works much better than the above
                #  if high variance in svs of X
                self._modes = (
                    self._u[: self.m] @ np.diag(1 / self._s) @ Phi_comp
                )
            else:
                self._modes = Phi_comp
        return self._modes

    @property
    def xi(self) -> np.ndarray:
        if self._xi is None:
            # r = self.r if self.r > 0 else self.m
            # _, Phi = self.eig
            # xi = self.Phi.conj().T @ self._Y @ np.linalg.pinv(self.C)

            # from scipy.optimize import minimize

            # def objective_function(x):
            #     return np.linalg.norm(
            #         self._Y[:r] - Phi @ np.diag(x) @ self.C, "fro"
            #     ) + 0.5 * np.linalg.norm(x, 1)

            # # Minimize the objective function
            # xi = minimize(objective_function, np.ones(r)).x
            xi = np.linalg.lstsq(
                self.modes,
                self._Y.T[0],
                rcond=None,
            )[0]
            self._xi = np.abs(xi)
        return self._xi

    def _fit(self, X: np.ndarray, Y: np.ndarray):
        # Perform singular value decomposition on X
        r = self.r if self.r > 0 else self.m
        # # Truncate the singular value matrices
        self._u, self._s, vt_ = np.linalg.svd(X, full_matrices=False)
        self._u, self._s, vt_ = _truncate_svd(self._u, self._s, vt_, r)
        self._u = self._u
        ut_ = self._u.conj().T
        self._v = vt_.conj().T
        s_inv = np.diag(np.reciprocal(self._s))
        # Compute the low-rank approximation of Koopman matrix
        self.A_bar = ut_[:, : self.m] @ Y @ self._v @ s_inv
        # In case of DMDwC we are only interested in part corresponding to states
        self.A_bar = self.A_bar[: self.m, : self.m]

        # Perform eigenvalue decomposition on A
        self.Lambda, W = self.eig

        # Compute the coefficient matrix
        # self.Phi = Y @ self._v[:, : self.m] @ s_inv[: self.m, : self.m] @ W
        # self.Phi = u_ @ W
        self.Phi = self._u[: self.m, : self.m] @ s_inv[: self.m, : self.m] @ W
        self.A = Y @ self._v @ s_inv @ ut_

    def fit(self, X: np.ndarray, Y: Union[np.ndarray, None] = None):
        """
        Fit the DMD model to the input X.

        Args:
            X: Input X matrix of shape (n, m), where m is the number of variables and n is the number of time steps.
            Y: The output snapshot matrix of shape (n, m).

        """
        # Build X matrices
        if Y is None:
            Y = X[1:, :]
            X = X[:-1, :]
        X = X.T  # PATCH#1: Match (m, n) implementation
        Y = Y.T  # PATCH#1: Match (m, n) implementation

        self._Y = Y

        self.m, self.n = self._Y.shape

        self._fit(X, self._Y)

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
        if self.A is None or self.m is None:
            raise RuntimeError("Fit the model before making predictions.")

        mat = np.zeros((forecast + 1, self.m))
        mat[0, :] = x
        for s in range(1, forecast + 1):
            mat[s, :] = (self.A @ mat[s - 1, :]).real
        return mat[1:, :]


class DMDc(DMD):
    def __init__(
        self, p: int = 0, q: int = 0, B: Union[np.ndarray, None] = None
    ):
        super().__init__(p + q if p + q > 0 else 0)
        self.p = p
        self.q = q
        self.B = B
        self.known_B = B is not None
        self.l: int

    def fit(
        self,
        X: np.ndarray,
        Y: Union[np.ndarray, None] = None,
        U: Union[np.ndarray, None] = None,
    ):
        if U is None:
            super().fit(X, Y)
            return
        U_ = U.copy()
        if Y is None:
            Y = X[1:, :]
            X = X[:-1, :]
            U_ = U_[:-1, :]
        if not self.known_B:
            X = np.hstack((X, U_))

        if X.shape[0] != U_.shape[0]:
            raise ValueError(
                "X and u must have the same number of time steps.\n"
                f"X: {X.shape[0]}, u: {U_.shape[0]}"
            )

        X = X.T  # PATCH#1: Match (m, n) implementation
        U_ = U_.T  # PATCH#1: Match (m, n) implementation
        Y = Y.T  # PATCH#1: Match (m, n) implementation

        if not self.known_B:
            self._Y = Y
        else:
            # Subtract the effect of actuation
            self._Y = Y - self.B * U_

        self.l = U_.shape[0]
        self.m, self.n = Y.shape
        self.p = self.p if self.p > 0 else self.m
        self.q = self.q if self.q > 0 else self.l
        self.r = self.p + self.q

        self._fit(X, self._Y)
        if not self.known_B:
            # split K into state transition matrix and control matrix
            self.B = self.A[: self.m, -self.l :]
            self.A = self.A[: self.m, : -self.l]

    def predict(
        self,
        x: np.ndarray,
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
        if U is None:
            mat = super().predict(x, forecast)
            return mat
        if self.A is None or self.m is None:
            raise RuntimeError("Fit the model before making predictions.")
        if forecast != 1 and U.shape[0] != forecast:
            raise ValueError(
                "u must have forecast number of time steps.\n"
                f"u: {U.shape[0]}, forecast: {forecast}"
            )

        mat = np.zeros((forecast + 1, self.m))
        mat[0, :] = x
        for s in range(1, forecast + 1):
            action = (self.B @ U[s - 1, :]).real
            mat[s, :] = (self.A @ mat[s - 1, :]).real + action
        return mat[1:, :]

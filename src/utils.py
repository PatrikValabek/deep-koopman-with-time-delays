import re

import numpy as np


def normalize_string(s):
    return re.sub(r"[^a-zA-Z0-9\s]", "", s)


def get_default_rank(X, noise_variance: float | None = None):
    """Get default rank for the given data matrix

    Args:
        X (np.ndarray): Data matrix

    Returns:
        int: Default rank

    References:
        [1] Gavish, M., and Donoho L. D. (2014). The Optimal Hard Threshold for Singular Values is 4/sqrt(3). IEEE Transactions on Information Theory 60.8 (2014): 5040-5053. doi:[10.1109/TIT.2014.2323359](https://doi.org/10.1109/TIT.2014.2323359).
    """
    n, m = X.shape
    beta = m / n
    s = np.linalg.svd(X.T, compute_uv=False)
    if noise_variance is None:
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        tau = omega * np.median(s)
    else:
        lambda_opt = np.sqrt(
            2 * (beta + 1)
            + (8 * beta) / ((beta + 1) + np.sqrt(beta**2 + 14 * beta + 1))
        )

        tau = lambda_opt * np.sqrt(n * noise_variance)
    r = sum(s > tau)
    return r

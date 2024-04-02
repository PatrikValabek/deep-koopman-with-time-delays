"""DMD with Control vs Deep Recurrent Koopman (DeReK) comparison for non-linear systems"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.dmd import DMDwC  # noqa: E402
from functions.preprocessing import hankel  # noqa: E402

results_path = "results/.dmdc/"
if not os.path.exists(results_path):
    os.makedirs(results_path)


def sse_row_wise(Y: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    return np.sum((Y - Y_pred) ** 2, axis=1)


def plot_prediction(
    Y: np.ndarray | pd.DataFrame,
    Y_pred: np.ndarray | pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        ax = plt.subplot()
    ax.plot(Y, "-", color="tab:blue")
    ax.plot(Y_pred, "-", color="tab:orange")
    legend_elements = [
        Line2D([0], [0], linestyle="-", color="tab:blue", label="Original"),
        Line2D([0], [0], linestyle="-", color="tab:orange", label="Predicted"),
    ]
    ax.legend(handles=legend_elements)
    return ax


if __name__ == "__main__":
    # Import data
    hn = 1
    train_data = pd.read_pickle("data/train_sim.pkl")
    X: np.ndarray = train_data["X"]
    X = hankel(X, hn)[:-1]
    U: np.ndarray = train_data["U"]
    U = hankel(U, hn)[:-1]
    # We want Y_k to be X_{k+1}
    Y: np.ndarray = train_data["Y"][hn:]
    test_data = pd.read_pickle("data/test_sim.pkl")
    X_t: np.ndarray = test_data["X"]
    X_t = hankel(X_t, hn)[:-1]
    U_t: np.ndarray = test_data["U"]
    U_t = hankel(U_t, hn)[:-1]
    # We want Y_k to be X_{k+1}
    Y_t: np.ndarray = test_data["Y"][hn:]

    # Set parameters
    pred_step = len(Y_t)
    r = 0

    # Perform DMD
    model = DMDwC(r=r)
    model.fit(X, U=U)
    Y_pred = model.predict(X_t[0, :], U=U_t, forecast=pred_step)[
        :, -Y.shape[1] :
    ]

    # Print SSE for each variable
    print("SSE for each variable:")
    mse = np.mean(sse_row_wise(Y_t, Y_pred))
    print("MSE: ", mse)
    sse = np.sum(sse_row_wise(Y_t, Y_pred))
    print("SSE: ", sse)

    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(16, 9))
    axs[0].set_title(
        f"DMD multi step prediction (SSE: {sse:.2f}; MSE: {mse:.2f})"
    )
    axs[0] = plot_prediction(Y_t, Y_pred, ax=axs[0])
    axs[0].plot(
        sse_row_wise(Y_t, Y_pred), color="tab:gray", alpha=0.5, label="SSE"
    )
    axs[0].grid()

    # Perform DMD as one-step ahead prediction
    Y_pred = np.zeros((pred_step, Y.shape[1]))
    for i in range(pred_step):
        y_pred = model.predict(
            X_t[i, :], U=U_t[i, :].reshape(1, -1), forecast=1
        ).reshape(-1)[-Y.shape[1] :]
        Y_pred[i, :] = y_pred

    print("SSE for each variable:")
    mse = np.mean(sse_row_wise(Y_t, Y_pred))
    print("MSE: ", mse)
    sse = np.sum(sse_row_wise(Y_t, Y_pred))
    print("SSE: ", sse)

    # Plot results
    axs[1].set_title(
        f"DMD single step prediction (SSE: {sse:.2f}; MSE: {mse:.2f})"
    )
    axs[1] = plot_prediction(Y_t, Y_pred, ax=axs[1])
    axs[1].plot(
        sse_row_wise(Y_t, Y_pred), color="tab:gray", alpha=0.5, label="SSE"
    )
    axs[1].grid()

    axs[2].set_title("Actuation")
    axs[2].plot(U_t[:, -1])
    axs[2].grid()

    fig.tight_layout()
    fig.savefig(f"{results_path}dmdc_prediction-hn{hn}-r{r}.png")

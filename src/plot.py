from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

line_styles = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 5)),
    (0, (3, 5, 1, 5)),
    (0, (3, 10, 1, 10)),
    (0, (5, 1)),
    (0, (5, 10)),
]

# Combine default color cycle with the line styles
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
line_cycler = cycler("color", colors) + cycler("linestyle", line_styles)

# Apply the combined cycler to the axes
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern",
        "font.serif": "Computer Modern",
        "axes.prop_cycle": line_cycler,
        "axes.grid": True,
    }
)


def set_size(
    width: float
    | int
    | Literal["article", "ieee", "thesis", "beamer"] = 307.28987,
    fraction=1.0,
    subplots=(1, 1),
):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the height which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "article":
        width_pt = 390.0
    elif width == "ieee":
        width_pt = 252.0
    elif width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = (
        fig_width_in * golden_ratio * ((subplots[0] * fraction) / subplots[1])
    )

    return (fig_width_in * 1.2, fig_height_in * 1.2)


def plot_results(U_t, *Ys, labels=None):
    """
    Plot the results of the eDMD model predictions and the input flow rate.

    Parameters:
    *Ys (tuple of ndarrays): True water level values and predicted water level values.
    U_t (ndarray): Input flow rate values.
    """
    fig, axs = plt.subplots(
        3, 1, figsize=set_size("ieee", subplots=(3, 1)), sharex=True
    )
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    for ys in Ys:
        for i, (ax, y) in enumerate(zip(axs, ys.T), start=1):
            ax.plot(y)
            ax.set_ylabel(f"Water Level in Tank {i} " + "$\\mathrm{(m)}$")
            ax.ticklabel_format(style="sci", axis="y", scilimits=(2, 1))

    axs[2].plot(U_t)
    axs[2].set_ylabel("Input Flow Rate $\\mathrm{(m^3~s^{-1})}$")
    axs[2].set_xlabel("Time (-)")
    axs[2].ticklabel_format(style="sci", axis="y", scilimits=(2, 1))

    axs[0].legend(labels)

    fig.tight_layout()

    return fig


def plot_eigs(
    eigs,
    amplitudes: np.ndarray | None = None,
    ax=None,
):
    # Sort eigenvalues, modes, and dynamics according to amplitude magnitude.
    if amplitudes is not None:
        mode_order = np.argsort(-amplitudes)
        lead_eigs = eigs[mode_order]
        amplitudes = amplitudes[mode_order]
    else:
        lead_eigs = eigs

    dt = 10
    disc_eigs = lead_eigs
    cont_eigs = np.log(disc_eigs) / dt

    # Get the actual rank used for the DMD fit.
    rank = len(disc_eigs)
    index_modes_cc = []
    index_modes = (0, 1)
    for idx in index_modes:
        eig = cont_eigs[idx]
        if eig.conj() not in cont_eigs:
            index_modes_cc.append((idx, idx))
        elif idx not in np.array(index_modes_cc):
            index_modes_cc.append((idx, list(cont_eigs).index(eig.conj())))
    other_eigs = np.setdiff1d(np.arange(rank), np.array(index_modes_cc))

    max_eig_ms = 10
    circle_color = "tab:blue"
    rank_color = "tab:orange"
    main_colors = ["tab:red", "tab:green", "tab:purple"]

    if ax is None:
        _, eig_ax = plt.subplots(1, 1, figsize=set_size("ieee"))
    else:
        eig_ax = ax
    if isinstance(eig_ax, np.ndarray):
        eig_ax = eig_ax[0]
    # PLOTS 2-3: Plot the eigenvalues (discrete-time and continuous-time).
    # Scale marker sizes to reflect their associated amplitude.
    if amplitudes is not None:
        ms_vals = max_eig_ms * np.sqrt(amplitudes / amplitudes[0])
    else:
        ms_vals = max_eig_ms / 2 * np.ones(rank)
    # PLOT 2: Plot the discrete-time eigenvalues on the unit circle.
    eig_ax.axvline(x=0, c="k", lw=1)
    eig_ax.axhline(y=0, c="k", lw=1)
    eig_ax.axis("equal")
    t = np.linspace(0, 2 * np.pi, 100)
    eig_ax.plot(np.cos(t), np.sin(t), c=circle_color, ls="--")

    # Now plot the eigenvalues and record the colors used for each main index.
    mode_colors = {}

    # Plot the main indices and their complex conjugate.
    for i, indices in enumerate(index_modes_cc):
        for idx in indices:
            eig_ax.plot(
                disc_eigs[idx].real,
                disc_eigs[idx].imag,
                "x",
                c=main_colors[i],
                ms=ms_vals[idx],
            )
            mode_colors[idx] = main_colors[i]
    # Plot all other DMD eigenvalues.
    for idx in other_eigs:
        eig_ax.plot(
            disc_eigs[idx].real,
            disc_eigs[idx].imag,
            "x",
            c=rank_color,
            ms=ms_vals[idx],
        )

    return eig_ax

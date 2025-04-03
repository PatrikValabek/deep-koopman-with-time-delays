from typing import TypedDict

import numpy as np
import torch
from neuromancer.dataset import DictDataset
from torch.utils.data import DataLoader

from .models import TwoTank


class TwoTankData(TypedDict):
    Y: np.ndarray
    X: np.ndarray
    U: np.ndarray
    Time: np.ndarray


def _forward_shift(data: np.ndarray, delay: int) -> np.ndarray:
    """
    Forward shift the data by the given delay.

    Args:
        data (np.ndarray): Input data to shift.
        delay (int): Delay to shift the data by.

    Returns:
        np.ndarray: Shifted data.

    Examples:
    >>> U = np.array([1, 2, 3, 4, 5])
    >>> _forward_shift(U, 1)
    array([2, 3, 4, 5, 5])
    """
    data[:-delay] = data[delay:]
    return data


def add_noise(
    data: TwoTankData, noise_level: float = 0.1, seed: int = 42
) -> TwoTankData:
    """Add Gaussian noise to the input data.

    Args:
        data (np.ndarray): Input data to add noise to.
        noise_level (float, optional): Standard deviation of the Gaussian noise. Defaults to 0.1.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        np.ndarray: Noisy data with non-negative values.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level, size=data["X"].shape)
    noisy_data: TwoTankData = {k: v.copy() for k, v in data.items()}  # type: ignore
    noisy_data["X"] += noise
    noisy_data["X"][noisy_data["X"] < 0] = 0
    noisy_data["Y"] += noise
    noisy_data["Y"][noisy_data["Y"] < 0] = 0
    return noisy_data


def simulate_two_tank(
    n_samples: int,
    n_sequences: int,
    delay: int = 20,
    noise_variance: float = 0.0,
    seed: int = 42,
) -> tuple[TwoTank, tuple[TwoTankData, ...]]:
    """Simulate the two tank model

    Args:
        model (TwoTank): The two tank model
        nx (int): The number of states
        ny (int): The number of outputs
        nu (int): The number of inputs

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The simulated data
    """
    model = TwoTank()
    responses = []
    for _ in range(n_sequences):
        response = model.simulate(nsim=n_samples, ts=model.ts)
        response["U"] = _forward_shift(response["U"], delay)
        if noise_variance > 0:
            response = add_noise(response, noise_variance, seed)
        responses.append(response)
    return model, tuple(responses)


def get_data_loaders(
    sys: TwoTank,
    data: dict[str, TwoTankData],
    nsim,
    nsteps,
    time,
    bs,
):
    """
    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.system)
    :param ts: (float) step size
    :param bs: (int) batch size

    """
    nbatch = (nsim - time) // nsteps
    length = (nsim // nsteps) * nsteps

    def _create_sequences(data, sequence_length):
        """Covert np.array data to lstm input as moving window sequences"""
        X = []
        for i in range(len(data) - sequence_length):
            X.append(data[i : i + sequence_length])
        return np.array(X)

    def normalize(x, mean, std):
        return (x - mean) / std

    def create_loader(
        sys: TwoTank,
        sim_data: TwoTankData,
        length: int,
        time: int,
        nbatch: int,
        nsteps: int,
        bs: int,
        name: str,
    ) -> DataLoader:
        """Create a PyTorch DataLoader from simulation data.

        Args:
            sim_data: Dictionary containing simulation data with keys 'Y' and 'U'
            length: Total length of the data
            time: Number of lookback steps
            nbatch: Number of batches
            nsteps: Number of prediction steps
            nx: Number of states
            nu: Number of inputs
            bs: Batch size
            name: Name of the dataset ('train' or 'dev')

        Returns:
            DataLoader: PyTorch DataLoader containing the processed data
        """
        nx = sys.nx
        nu = sys.nu

        # Normalize and reshape state data
        X_b = normalize(
            sim_data["Y"][:length],
            sys.stats["Y"]["mean"],
            sys.stats["Y"]["std"],
        )
        X = X_b[time:].reshape(nbatch, nsteps, nx)
        X = torch.tensor(X, dtype=torch.float32)

        # Normalize and reshape input data
        U_b = normalize(
            sim_data["U"][:length],
            sys.stats["U"]["mean"],
            sys.stats["U"]["std"],
        )
        U = U_b[time:].reshape(nbatch, nsteps, nu)
        U = torch.tensor(U, dtype=torch.float32)

        # Create time sequences
        UX_b = np.concatenate((X_b, U_b), axis=1)
        T = _create_sequences(UX_b, time)
        T = T.reshape(nbatch, nsteps, time, nu + nx)
        T = torch.tensor(T, dtype=torch.float32)

        # Create dataset
        data = DictDataset(
            {
                "Y": X,
                "Y0": X[:, 0:1, :],
                "U": U,
                "timeYU0": T[:, 0:1, :, :].reshape(nbatch, time, nu + nx),
                "timeYU": T,
            },
            name=name,
        )

        # Create dataloader
        loader = DataLoader(
            data,
            batch_size=bs,
            collate_fn=data.collate_fn,
            shuffle=True,
        )

        return loader

    return (
        create_loader(
            sys,
            data_,
            length,
            time,
            nbatch,
            nsteps,
            bs,
            name,
        )
        for name, data_ in data.items()
    )

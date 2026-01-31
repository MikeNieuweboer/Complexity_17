"""Plotting functionality for the MCAF data.

Group:        17
Course:       Complex System Simulation

Description:
-----------
Plots the mean and percentiles of the MCAF data in the file with the given stem and
removal type.

AI usage:
--------
> Analyse the @src/plot_mcaf.py file and try to understand its structure for
> future doc comment generation

> Generate the PEP style comment.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from utils import data_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Namespace with attributes ``weights_name`` (str) and ``type`` ("blob"
    or "random").

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "weights_name",
        help="The stem of the filename containing the weights",
        type=str,
    )
    parser.add_argument(
        "type",
        help="The type of removal procedure",
        choices=["blob", "random"],
        type=str,
    )
    return parser.parse_args()


def read_data(path: Path) -> tuple[npt.NDArray, npt.NDArray]:
    """Read MCAF CSV data and return the removed counts and measurements.

    The CSV is expected to have a header row. Each subsequent row must contain an
    integer in the first column (number removed) followed by one or more integer
    measurements for that removal. Example row:

        10, 5, 7, 6, 8

    Parameters
    ----------
    path : Path
        Path to the CSV file.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple ``(removed, data)`` where ``removed`` is a 1-D integer array of
        length N with the number removed for each row and ``data`` is a 2-D
        integer array with shape (N, R) containing R repeated measurements.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed into integers.

    """
    with path.open("r") as file:
        reader = csv.reader(file)
        next(reader)
        input_data = np.array(list(reader), dtype=int)
    removed = input_data[:, 0]
    data = input_data[:, 1:]
    return (removed, data)


def mean_data(data: npt.NDArray) -> npt.NDArray:
    """Return the row-wise mean of the provided 2-D array.

    Parameters
    ----------
    data : numpy.ndarray
        A 2-D array with shape (N, R) where R are repeated measurements.

    Returns
    -------
    numpy.ndarray
        A 1-D array of length N containing the mean of each row.

    """
    return np.mean(data, axis=1)


def std_data(data: npt.NDArray) -> npt.NDArray:
    """Return the row-wise standard deviation of the provided 2-D array.

    Parameters
    ----------
    data : numpy.ndarray
        A 2-D array with shape (N, R).

    Returns
    -------
    numpy.ndarray
        A 1-D array of length N containing the standard deviation of each row.

    """
    return np.std(data, axis=1)


def percentile_data(data: npt.NDArray, q: float) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute symmetric percentile bounds for each row.

    For a percentile ``q`` this returns the ``q``-th and ``(100-q)``-th percentiles
    computed row-wise. For example, ``q=5`` yields the 5th and 95th percentiles.

    Parameters
    ----------
    data : numpy.ndarray
        A 2-D array with shape (N, R).
    q : float
        Percentile value between 0 and 50 (inclusive) used to compute the lower
        bound; the upper bound is computed as ``100 - q``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple ``(lower, upper)`` where each is a 1-D array of length N.

    """
    lower = np.percentile(data, q, axis=1)
    upper = np.percentile(data, 100 - q, axis=1)
    return (lower, upper)


def bounds_data(data: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Return the row-wise minimum and maximum for a 2-D array.

    Parameters
    ----------
    data : numpy.ndarray
        A 2-D array with shape (N, R).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(lower, upper)`` where each is a 1-D array of length N containing the
        minimum and maximum value of each row respectively.

    """
    lower = np.min(data, axis=1)
    upper = np.max(data, axis=1)
    return (lower, upper)


def style_axis(axis: Axes) -> None:
    """Apply common styling to a matplotlib axis.

    Currently this enables the grid for the axis. This helper centralizes
    styling applied to axes created in this module.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The axis to style in-place.

    """
    axis.grid()


def plot_time(
    removed: npt.NDArray,
    mse_data: npt.NDArray,
    count_data: npt.NDArray,
    removal_type: str,
    *,
    axis: Axes | None = None,
) -> None:
    """Plot recovery time statistics on the given axis.

    This plots the row-wise mean and shaded percentile bounds for both the
    MSE-based and count-based recovery time measurements. If ``axis`` is not
    provided a new figure and axis will be created.

    Parameters
    ----------
    removed : numpy.ndarray
        1-D array with the number of removals for each row (x axis).
    mse_data : numpy.ndarray
        2-D array with MSE-based recovery time measurements (rows align with
        ``removed``).
    count_data : numpy.ndarray
        2-D array with count-based recovery time measurements.
    removal_type : str
        Description of the removal procedure used; used in titles/labels.
    axis : matplotlib.axes.Axes, optional
        Axis to draw onto. If None, a new axis is created.

    """
    if axis is None:
        _, axis = plt.subplots(nrows=1)
        style_axis(axis)

    axis.set_title(f"Recovery times with {removal_type} removals")
    axis.set_xlabel("Removals")
    axis.set_ylabel("Recovery time (steps)")

    mean = mean_data(mse_data)
    lower, upper = percentile_data(mse_data, 5)

    axis.plot(removed, mean, label="Mean MSE-based")
    axis.fill_between(
        removed,
        lower,
        upper,
        alpha=0.2,
        label="5th Percentile MSE-based",
    )

    mean = mean_data(count_data)
    lower, upper = percentile_data(count_data, 5)

    axis.plot(removed, mean, label="Mean Count-based")
    axis.fill_between(
        removed,
        lower,
        upper,
        alpha=0.2,
        label="5th Percentile Count-based",
    )

    axis.legend()


def plot_survived(
    removed: npt.NDArray,
    survived: npt.NDArray,
    removal_type: str,
    *,
    axis: Axes | None = None,
) -> None:
    """Plot mean and percentile bounds for cell counts after simulation.

    Plots the row-wise mean and symmetric percentile bounds (e.g. 5th/95th)
    for the provided ``survived`` measurements against ``removed``. If
    ``axis`` is not provided a new figure and axis will be created.

    Parameters
    ----------
    removed : numpy.ndarray
        1-D array with the number of removals for each row (x axis).
    survived : numpy.ndarray
        2-D array with cell count measurements after the simulation.
    removal_type : str
        Description of the removal procedure used; used in titles/labels.
    axis : matplotlib.axes.Axes, optional
        Axis to draw onto. If None, a new axis is created.

    Returns
    -------
    None

    """
    if axis is None:
        _, axis = plt.subplots(nrows=1)
        style_axis(axis)

    axis.set_title(f"Cell count with {removal_type} removals after 4000 steps")
    axis.set_xlabel("Removals")
    axis.set_ylabel("Cell count")

    mean = mean_data(survived)
    lower, upper = percentile_data(survived, 5)
    axis.plot(removed, mean, label="Mean cell count")
    axis.fill_between(
        removed,
        lower,
        upper,
        alpha=0.2,
        label="5th Percentile cell count",
    )
    axis.legend()


def main() -> None:
    """Entry point when run as a script.

    Parses CLI arguments, reads the three expected MCAF CSV files (count, mse,
    survived), and displays two subplots: recovery times and cell counts.

    The expected CSV filenames are constructed as
    ``{type}_{weights_name}_count.csv``, ``{type}_{weights_name}_mse.csv``, and
    ``{type}_{weights_name}_survived.csv`` and are looked up under
    ``data_path / 'MCAF'``.
    """
    mcaf_path = data_path / "MCAF"

    args = parse_args()
    weights_name = args.weights_name  # pyright: ignore[reportAttributeAccessIssue]
    mcaf_type = args.type  # pyright: ignore[reportAttributeAccessIssue]

    count_path = mcaf_path / f"{mcaf_type}_{weights_name}_count.csv"
    mse_path = mcaf_path / f"{mcaf_type}_{weights_name}_mse.csv"
    survived_path = mcaf_path / f"{mcaf_type}_{weights_name}_survived.csv"

    removed, count_data = read_data(count_path)
    _, mse_data = read_data(mse_path)
    _, survived = read_data(survived_path)

    fig = plt.figure()
    axes = fig.subplots(nrows=2)
    style_axis(axes[0])
    style_axis(axes[1])

    plot_time(removed, mse_data, count_data, mcaf_type, axis=axes[0])
    plot_survived(removed, survived, mcaf_type, axis=axes[1])
    plt.show()


if __name__ == "__main__":
    main()

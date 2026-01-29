import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from utils import data_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "weights_name",
        help="The stem of the filename containing the weights",
        type=str,
    )
    parser.add_argument(
        "type",
        help="The type of removal procedure",
        choices=["channel", "channel_mask", "blob", "random"],
        type=str,
    )
    return parser.parse_args()


def read_data(path: Path) -> tuple[npt.NDArray, npt.NDArray]:
    with path.open("r") as file:
        reader = csv.reader(file)
        next(reader)
        input_data = np.array(list(reader), dtype=int)
    removed = input_data[:, 0]
    data = input_data[:, 1:]
    return (removed, data)


def mean_data(data: npt.NDArray) -> npt.NDArray:
    return np.mean(data, axis=1)


def std_data(data: npt.NDArray) -> npt.NDArray:
    return np.std(data, axis=1)


def percentile_data(data: npt.NDArray, q: float) -> tuple[npt.NDArray, npt.NDArray]:
    lower = np.percentile(data, q, axis=1)
    upper = np.percentile(data, 100 - q, axis=1)
    return (lower, upper)


def bounds_data(data) -> tuple[npt.NDArray, npt.NDArray]:
    lower = np.min(data, axis=1)
    upper = np.max(data, axis=1)
    return (lower, upper)


def style_axis(axis: Axes) -> None:
    axis.grid()


def plot_time(
    removed: npt.NDArray,
    mse_data: npt.NDArray,
    count_data: npt.NDArray,
    removal_type: str,
    *,
    axis: Axes | None = None,
) -> None:
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
        removed, lower, upper, alpha=0.2, label="5th Percentile MSE-based"
    )

    mean = mean_data(count_data)
    lower, upper = percentile_data(count_data, 5)

    axis.plot(removed, mean, label="Mean Count-based")
    axis.fill_between(
        removed, lower, upper, alpha=0.2, label="5th Percentile Count-based"
    )

    axis.legend()


def plot_survived(
    removed: npt.NDArray,
    survived: npt.NDArray,
    type: str,
    *,
    axis: Axes | None = None,
) -> None:
    if axis is None:
        _, axis = plt.subplots(nrows=1)
        style_axis(axis)

    axis.set_title(f"Cell count with {type} removals after 4000 steps")
    axis.set_xlabel("Removals")
    axis.set_ylabel("Cell count")

    mean = mean_data(survived)
    lower, upper = percentile_data(survived, 5)
    axis.plot(removed, mean, label="Mean cell count")
    axis.fill_between(
        removed, lower, upper, alpha=0.2, label="5th Percentile cell count"
    )
    axis.legend()


def main() -> None:
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

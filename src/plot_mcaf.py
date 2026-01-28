import argparse
import csv
from pathlib import Path

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

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
    pass


def plot_time(
    removed: npt.NDArray,
    time_data: npt.NDArray,
    type: str,
    *,
    axis: Axes | None = None,
) -> None:
    if axis is None:
        _, axis = plt.subplots(nrows=1)
        style_axis(axis)

    axis.set_title(f"Extinction with {type} removals")
    axis.set_xlabel("Removals")
    axis.set_ylabel("Extinction time (steps)")

    mean = mean_data(time_data)
    # std = std_data(time_data)
    # lower, upper = percentile_data(time_data, 5)
    lower, upper = bounds_data(time_data)

    axis.plot(removed, mean, label="Mean")
    axis.fill_between(removed, lower, upper, alpha=0.2, label="Bounds")

    axis.legend()


def plot_count(
    removed: npt.NDArray,
    count_data: npt.NDArray,
    type: str,
    *,
    axis: Axes | None = None,
) -> None:
    if axis is None:
        _, axis = plt.subplots(nrows=1)
        style_axis(axis)

    axis.set_title("Cell count with {type} removals after 1000 steps")
    axis.set_xlabel("Removals")
    axis.set_ylabel("Cell count")

    mean = mean_data(count_data)
    axis.plot(removed, mean)


def main() -> None:
    mcaf_path = data_path / "MCAF"

    args = parse_args()
    weights_name = args.weights_name  # pyright: ignore[reportAttributeAccessIssue]
    mcaf_type = args.type  # pyright: ignore[reportAttributeAccessIssue]

    count_path = mcaf_path / f"{mcaf_type}_{weights_name}_count.csv"
    time_path = mcaf_path / f"{mcaf_type}_{weights_name}_time.csv"

    removed, count_data = read_data(count_path)
    _, time_data = read_data(time_path)

    plot_time(removed, time_data, mcaf_type)
    plt.show()


if __name__ == "__main__":
    main()

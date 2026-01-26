import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(path: Path) -> None:
    data = None
    with path.open("r") as file:
        reader = csv.reader(file)
        data = np.array(list(reader), dtype=np.float32)

    if data is None:
        return

    x = np.arange(len(data))
    minimum = np.min(data, axis=1)
    mean = np.mean(data, axis=1)

    fig = plt.figure()
    axes = fig.subplots(nrows=1)

    axes.set_ylim(0, 1)
    axes.set_xlabel("Generaion")
    axes.set_ylabel("Loss")

    axes.plot(x, minimum, label="min")
    axes.plot(x, mean, label="mean")

    axes.legend()

    plt.show()


if __name__ == "__main__":
    main(Path("./data/FULL_CIRCLE_BALANCED_BASIC.csv"))

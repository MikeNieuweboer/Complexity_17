import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from grid import Grid


def channel_destruction() -> None:
    pass


def channel_mask_destruction() -> None:
    pass


def blob_destruction() -> None:
    pass


def random_destruction(
    grid: Grid, delay: int, max_testing_time: int, *, seed: int = 43
) -> None:
    # Warm up
    grid = grid.deepcopy()
    empty = grid.empty
    grid_state = grid.run_simulation(delay)

    alpha = grid_state.detach().numpy()[:, :, 0]
    alive = alpha > 0

    total = np.sum(alive)
    indices = np.where(alive)

    survival_times = []
    for ratio in tqdm(np.linspace(0, 1, 50)):
        survival_time = 0
        local_grid = grid.deepcopy()
        gen = np.random.Generator(np.random.PCG64(seed))

        # Generate indices to be removed
        removals = int(total * ratio)
        removed_indices = gen.choice(range(len(indices[0])), removals, replace=False)
        removed_row = indices[0][removed_indices]
        removed_column = indices[1][removed_indices]

        # Remove the random cells
        for r, c in zip(removed_row, removed_column, strict=True):
            local_grid.set_cell_state(c, r, empty)

        while survival_time < max_testing_time and np.any(
            local_grid.state(layer=0) > 0
        ):
            local_grid.step()
            survival_time += 1
        survival_times.append(survival_time)
    print(survival_times)


def load_weights(path: Path) -> tuple[npt.NDArray, npt.NDArray]:
    """Load weights from either an .npz or .pt file.

    Args:
    ----
        path: Path to the weights file (.npz or .pt)

    Returns:
    -------
        Tuple of (hidden_layer_weights, output_layer_weights) as numpy arrays

    Raises:
    ------
        ValueError: If file format is not supported or weights structure is invalid

    """
    weight_count = 2
    if path.suffix == ".npz":
        # Load from NPZ file
        data = np.load(path)
        # NPZ files typically store multiple arrays, get them in order
        arrays = [data[key] for key in sorted(data.files)]
        if len(arrays) < weight_count:
            msg = f"NPZ file must contain at least 2 weight arrays, got {len(arrays)}"
            raise ValueError(msg)
        return tuple(arrays[:weight_count])
    if path.suffix == ".pt":
        # Load from PyTorch file
        state_dict = torch.load(path, weights_only=True)
        # Convert torch tensors to numpy arrays
        weights = []
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            weights.append(tensor.cpu().numpy())

        if len(weights) < weight_count:
            msg = f"PT file must contain at least 2 weight tensors, got {len(weights)}"
            raise ValueError(msg)
        return tuple(weights[:weight_count])
    msg = f"Unsupported file format: {path.suffix}. Expected .npz or .pt"
    raise ValueError(msg)


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "weights",
        help="The path from the running root to the"
        ".npz or .pt file containing the weights.",
        type=str,
    )
    parser.add_argument(
        "delay",
        help="The amount of steps taken before removal occurs.",
        type=int,
    )
    parser.add_argument(
        "type",
        help="The type of removal procedure",
        choices=["channel", "channel_mask", "blob", "random"],
        type=str,
    )
    parser.add_argument(
        "-c",
        "--channel",
        help="Which channel to be targeted when using channel or channel_mask mode",
        type=int,
    )

    return parser.parse_args()  # pyright: ignore[reportReturnType]def main():


def main() -> None:
    seed = 43
    max_testing_time = 100

    args = parse_args()

    weight_path = Path(args.weights)  # pyright: ignore[reportAttributeAccessIssue]
    weights = load_weights(weight_path)

    num_channels = weights[1].shape[1]  # Number of outputs of the NN
    grid = Grid(50, 50, num_channels, seed=seed, weights=weights)

    delay = args.delay  # pyright: ignore[reportAttributeAccessIssue]
    arg_type = args.type  # pyright: ignore[reportAttributeAccessIssue]

    if arg_type == "random":
        random_destruction(grid, delay, max_testing_time)
    elif arg_type == "channel":
        channel_destruction()
    elif arg_type == "blob":
        blob_destruction()
    elif arg_type == "channel_mask":
        channel_mask_destruction()


if __name__ == "__main__":
    main()

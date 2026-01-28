"""Utils for calculating the unofficial MCAF value.

Name:        Mike Nieuweboer
Course:      Complex System Simulation

Description:
------------
Allows for the calculation of the MCAF (Mean Critical Annihilation Fraction, a
factor made up to make it feel as if something new has been invented), which
represents the fraction of cells that need to be annihilated for total
collapse of the system towards the empty zero state.
These fractions can be removed from:
- Random living cells.
- A centralised blob of cells.
- Cells with activity in a certain channel.
- A specified channel itself.

"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from grid import Grid
from utils import data_path, load_target_image, load_weights

removal_iterations = 50
min_count_diff = 30
max_diff_count = 100


def loss(state: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.nn.functional.mse_loss(state[0, :, :], target).detach())


def _simulate_to_end(
    grid: Grid,
    max_time: int,
    target_loss: float,
    target: torch.Tensor,
    target_count: int,
) -> tuple[int, int]:
    mse_time = 0
    count_time = 0

    state = grid.batch_state.squeeze(0)
    reached_mse = False
    reached_count = False
    iteration = 0
    while iteration < max_time:
        iteration += 1
        if not reached_mse:
            reached_mse = loss(state, target) < target_loss
            mse_time += 1
        diff = abs(int(torch.sum(state.detach()[0, :, :] > 0).detach()) - target_count)
        if diff < min_count_diff:
            if not reached_count:
                count_time = iteration
            reached_count = True
        elif diff > max_diff_count:
            reached_count = False
        if not reached_count:
            count_time = iteration

        grid.run_simulation_batch(1)
        state = grid.batch_state.squeeze(0)
    return (mse_time, count_time)


def find_start_flood(grid: Grid, gen: np.random.Generator) -> tuple[int, int]:
    state = grid.batch_state.detach().squeeze(0).numpy()
    indices = np.where(state[0, :, :] > 0)
    removed_index = int(gen.choice(range(len(indices[0])), 1)[0])
    row = int(indices[0][removed_index])
    column = int(indices[1][removed_index])
    return (row, column)


def flood_fill_step(queue: list[tuple[int, int]], grid: Grid, gen: np.random.Generator):
    state = grid.pool_state.squeeze(0)

    if queue == []:
        row, column = find_start_flood(grid, gen)
    else:
        row, column = queue.pop()
        while (state[0, row, column] == 0) and (queue != []):
            row, column = queue.pop()

        if queue == []:
            row, column = find_start_flood(grid, gen)

    grid.set_cell_state(0, column, row, grid.empty)

    # Check all surrounding cells
    for row_offset in range(-1, 2):
        for column_offset in range(-1, 2):
            new_row = row + row_offset
            new_column = column + column_offset

            # If cell is alive
            if state[0, new_row, new_column] > 0:
                queue.insert(0, (new_row, new_column))


def blob_destruction(
    grid: Grid, delay: int, max_time: int, target: torch.Tensor, *, seed: int = 43
) -> tuple[list[list[int]], list[list[int]], list[list[int]], npt.NDArray]:
    iterations = 50

    mse_times = [[] for _ in range(iterations)]
    count_times = [[] for _ in range(iterations)]
    surviving_cells = [[] for _ in range(iterations)]

    gen = np.random.Generator(np.random.PCG64(seed))
    for i in tqdm(range(iterations)):
        grid.clear_and_seed(grid_idx=0)
        grid.set_batch([0])
        grid.load_batch_from_pool()

        local_state = grid.run_simulation(delay).squeeze(0)
        target_loss = loss(grid.batch_state.squeeze(0), target) * 1.1
        target_count = int(
            np.sum(grid.batch_state.detach().squeeze(0).numpy()[0, :, :] > 0)
        )
        queue = [find_start_flood(grid, gen)]

        grid.write_batch_back_to_pool()

        alpha = local_state.detach().numpy()[0, :, :]
        alive = alpha > 0

        total = np.sum(alive)

        removed_arr = (
            np.pow(np.linspace(0, 1, removal_iterations), 2 / 3) * total
        ).astype(int)

        last_removed = 0
        for current_removal in removed_arr:
            removals = current_removal - last_removed
            for _ in range(removals):
                flood_fill_step(queue, grid, gen)
            last_removed += removals

            grid.set_batch([0])
            grid.load_batch_from_pool()
            mse_time, count_time = _simulate_to_end(
                grid, max_time, target_loss, target, target_count
            )
            mse_times[i].append(mse_time)
            count_times[i].append(count_time)

            surviving_cells[i].append(
                np.sum(grid.batch_state.detach().squeeze(0).numpy()[0, :, :] > 0)
            )
    return mse_times, count_times, surviving_cells, removed_arr


def random_destruction(
    grid: Grid, delay: int, max_time: int, target: torch.Tensor, *, seed: int = 43
) -> tuple[list[list[int]], list[list[int]], list[list[int]], npt.NDArray]:
    # Warm up
    grid.clear_and_seed(grid_idx=0)
    grid.set_batch([0])
    grid.load_batch_from_pool()
    empty = grid.empty
    grid_state = grid.run_simulation(delay).squeeze(0)
    target_loss = loss(grid.batch_state.squeeze(0), target) * 1.1
    grid.write_batch_back_to_pool()

    # Select cells to be deleted (which are currently alive.)
    alpha = grid_state.detach().numpy()[0, :, :]
    alive = alpha > 0

    total = int(np.sum(alive))
    indices = np.where(alive)
    iterations = 20

    # Initialise data lists
    mse_times = [[] for _ in range(iterations)]
    count_times = [[] for _ in range(iterations)]
    surviving_cells = [[] for _ in range(iterations)]

    gen = np.random.Generator(np.random.PCG64(seed))
    removed_arr = (np.pow(np.linspace(0, 1, removal_iterations), 2 / 3) * total).astype(
        int
    )
    for i in tqdm(range(iterations)):
        for removals in removed_arr:
            # Copy over the warmed grid to the active batch.
            grid.set_batch([0])
            grid.load_batch_from_pool()

            # Generate indices to be removed
            removed_indices = gen.choice(
                range(len(indices[0])),
                removals,
                replace=False,
            )
            removed_row = indices[0][removed_indices]
            removed_column = indices[1][removed_indices]

            # Remove the random cells
            for r, c in zip(removed_row, removed_column, strict=True):
                grid.set_cell_state_batch(0, c, r, empty)

            # Simulate and gather results
            mse_time, count_time = _simulate_to_end(
                grid,
                max_time,
                target_loss,
                target,
                total,
            )

            mse_times[i].append(mse_time)
            count_times[i].append(count_time)

            survived = np.sum(grid.batch_state.detach().squeeze(0).numpy()[0, :, :] > 0)
            surviving_cells[i].append(survived)

    return mse_times, count_times, surviving_cells, removed_arr


def save_stats(path: Path, removed: npt.NDArray, stats: list[list[int]]):
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["Removed"] + [f"Iteration_{i + 1}" for i in range(len(stats))])

        # Write survival times data
        for idx, remove_count in enumerate(removed):
            row = [remove_count] + [stats[i][idx] for i in range(len(stats))]
            writer.writerow(row)


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "weights",
        help="The path from the running root to the"
        ".npz or .pt file containing the weights.",
        type=str,
    )
    parser.add_argument(
        "target",
        help="The path from the root to thefile containing the target image.",
        type=str,
    )
    parser.add_argument(
        "delay",
        help="The amount of steps taken before removal occurs.",
        type=int,
    )
    parser.add_argument(
        "testing_time",
        help="The amount of steps taken after removal occurs.",
        type=int,
    )
    parser.add_argument(
        "type",
        help="The type of removal procedure",
        choices=["blob", "random"],
        type=str,
    )

    return parser.parse_args()  # pyright: ignore[reportReturnType]def main():


def main() -> None:
    seed = 43

    args = parse_args()

    weight_path = Path(args.weights)  # pyright: ignore[reportAttributeAccessIssue]
    weights = load_weights(weight_path)

    num_channels = weights[1].shape[1]  # Number of outputs of the NN
    grid = Grid(
        1,
        1,
        num_channels,
        50,
        50,
        seed=seed,
        weights=weights,
    )

    delay = args.delay  # pyright: ignore[reportAttributeAccessIssue]
    arg_type = args.type  # pyright: ignore[reportAttributeAccessIssue]
    target_path = Path(args.target)  # pyright: ignore[reportAttributeAccessIssue]
    max_time = args.testing_time  # pyright: ignore[reportAttributeAccessIssue]

    target = load_target_image(target_path, 50)

    if arg_type == "random":
        mse_times, count_times, surviving_cells, removed = random_destruction(
            grid, delay, max_time, target
        )
    elif arg_type == "blob":
        mse_times, count_times, surviving_cells, removed = blob_destruction(
            grid, delay, max_time, target
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown arg_type: {arg_type}")

    # Save results to CSV
    csv_folder = data_path / "MCAF"
    csv_folder.mkdir(parents=True, exist_ok=True)
    survive_path = csv_folder / f"{arg_type}_{weight_path.stem}_survived.csv"
    save_stats(survive_path, removed, surviving_cells)
    survive_path = csv_folder / f"{arg_type}_{weight_path.stem}_mse.csv"
    save_stats(survive_path, removed, mse_times)
    survive_path = csv_folder / f"{arg_type}_{weight_path.stem}_count.csv"
    save_stats(survive_path, removed, count_times)


if __name__ == "__main__":
    main()

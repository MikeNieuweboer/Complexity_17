"""Utils for calculating the unofficial MCAF value.

Name:        Mike Nieuweboer
Course:      Complex System Simulation

Description:
------------
Allows for the gathering of data surrounding the MCAF (Mean Critical
Annihilation Fraction, a factor made up to make it feel as if something new
has been invented), which represents the fraction of cells that need to be
annihilated for total collapse of the system towards the empty zero state. These
fractions can be removed from:
- Random living cells.
- A centralised blob of cells.

To analyse both where this dropoff happens and if there is any critical slowing
down, the cell count after run time and the time until either the cell count or MSE
loss has returned to normal are stored in the <data_path>/MCAF folder.

AI usage:
--------
> Analyse the @src/mcaf.py file to allow for future doc comment generation

> Generate the PEP styled doc comments
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from grid import Grid
from utils import data_path, load_target_image, load_weights

# Uggly hardcoded global variables.
## The amount of steps going from 0 to 100% cell removal.
removal_iterations = 50
## The difference in cell count to be seen as recovered.
min_count_diff = 30
## The difference in cell count to lose recovered state.
max_diff_count = 100


def loss(state: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean-squared error between a batch state and a target image.

    The function expects ``state`` to be a batched tensor where the first
    dimension is the batch index; the function uses the first batch
    (``state[0, :, :]``) for the comparison. ``target`` is expected to be a
    2-D tensor with the same spatial shape as the state slice.

    Args:
    ----
        state: Batched state tensor (B, H, W) or similar; only index 0 is used.
        target: Target image tensor (H, W) to compare against.

    Returns:
    -------
        Float mean-squared error between the state slice and the target.

    """
    return float(torch.nn.functional.mse_loss(state[0, :, :], target).detach())


def _simulate_to_end(
    grid: Grid,
    max_time: int,
    target_loss: float,
    target: torch.Tensor,
    target_count: int,
) -> tuple[int, int]:
    """Advance the grid until ``max_time`` and measure recovery times.

    The simulation is stepped forward up to ``max_time`` iterations. Two
    recovery metrics are recorded:

    - ``mse_time``: iteration count related to when the MSE between the
      current state and ``target`` falls below ``target_loss``.
    - ``count_time``: iteration count related to when the absolute
      difference between the current living-cell count and ``target_count``
      falls below the module-level ``min_count_diff``.

    Note: both return values are integers intended to represent the
    iteration index at which the condition was observed.

    Args:
    ----
        grid: Grid instance used to run the simulation; its state is mutated.
        max_time: Maximum number of iterations to simulate.
        target_loss: MSE threshold considered recovered.
        target: Target image tensor used for MSE calculations.
        target_count: Reference living-cell count used for count-based metric.

    Returns:
    -------
        A tuple ``(mse_time, count_time)`` of integers describing recovery
        times measured during the simulation.

    """
    mse_time = 0
    count_time = 0

    state = grid.batch_state.squeeze(0)
    reached_mse = False
    reached_count = False
    iteration = 0

    while iteration < max_time:
        iteration += 1

        # MSE based regeneration counting
        if not reached_mse:
            reached_mse = loss(state, target) < target_loss
            mse_time += 1
        diff = abs(int(torch.sum(state.detach()[0, :, :] > 0).detach()) - target_count)

        # Cell count based regeneration counting
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
    """Pick a random living cell from the grid and return its coordinates.

    The returned coordinates are given in ``(row, column)`` order and are
    selected uniformly from currently alive cells in the active batch.

    Args:
    ----
        grid: Grid whose current batch state is queried.
        gen: NumPy random generator used for deterministic selection.

    Returns:
    -------
        A ``(row, column)`` tuple of integers pointing to a living cell.

    """
    state = grid.batch_state.detach().squeeze(0).numpy()
    indices = np.where(state[0, :, :] > 0)
    removed_index = int(gen.choice(range(len(indices[0])), 1)[0])
    row = int(indices[0][removed_index])
    column = int(indices[1][removed_index])
    return (row, column)


def flood_fill_step(
    queue: list[tuple[int, int]],
    grid: Grid,
    gen: np.random.Generator,
) -> None:
    """Perform a single flood-fill removal step on the grid.

    The function mutates ``grid`` and the provided ``queue``. If ``queue`` is
    empty a random living cell is located and used as the starting point. The
    selected cell is set to the grid's empty state and all surrounding alive
    neighbours are added to the queue for future removals.

    Args:
    ----
        queue: A list acting as a stack/queue of ``(row, column)`` tuples.
        grid: Grid instance; its pool state is modified by clearing one cell.
        gen: NumPy random generator used when a start cell must be chosen.

    """
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
    grid: Grid,
    delay: int,
    max_time: int,
    target: torch.Tensor,
    *,
    seed: int = 43,
) -> tuple[list[list[int]], list[list[int]], list[list[int]], npt.NDArray]:
    """Progressively remove contiguous blobs and record recovery statistics.

    A number of iterations are performed where the grid is warmed-up, a
    contiguous region of living cells is removed using flood-fill steps and
    the simulation is advanced to measure recovery times. Results for each
    replication and removal fraction are collected and returned.

    Args:
    ----
        grid: Grid instance pre-initialised with weights and size.
        delay: Number of steps to run before the first removal (warm-up).
        max_time: Maximum simulation steps to run after each removal.
        target: Target image tensor used for MSE calculations.
        seed: RNG seed to make the blob removal deterministic.

    Returns:
    -------
        A tuple ``(mse_times, count_times, surviving_cells, removed_arr)``:
        - ``mse_times``: list of lists of mse recovery times (per replication).
        - ``count_times``: list of lists of count-based recovery times.
        - ``surviving_cells``: list of lists of surviving cell counts.
        - ``removed_arr``: 1-D numpy array with cumulative removed counts at
          each removal step.

    """
    iterations = 50

    mse_times = [[] for _ in range(iterations)]
    count_times = [[] for _ in range(iterations)]
    surviving_cells = [[] for _ in range(iterations)]

    gen = np.random.Generator(np.random.PCG64(seed))
    # Initialize removed_arr to satisfy static analyzers in case iterations == 0
    removed_arr: npt.NDArray = np.array([], dtype=int)
    for i in tqdm(range(iterations)):
        # Warm up
        grid.clear_and_seed(grid_idx=0)
        grid.set_batch([0])
        grid.load_batch_from_pool()

        local_state = grid.run_simulation(delay).squeeze(0)
        # Give 10% leeway in the MSE matching.
        target_loss = loss(local_state, target) * 1.1
        target_count = int(
            np.sum(grid.batch_state.detach().squeeze(0).numpy()[0, :, :] > 0),
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
                grid,
                max_time,
                target_loss,
                target,
                target_count,
            )
            mse_times[i].append(mse_time)
            count_times[i].append(count_time)

            surviving_cells[i].append(
                np.sum(grid.batch_state.detach().squeeze(0).numpy()[0, :, :] > 0),
            )
    return mse_times, count_times, surviving_cells, removed_arr


def random_destruction(
    grid: Grid,
    delay: int,
    max_time: int,
    target: torch.Tensor,
    *,
    seed: int = 43,
) -> tuple[list[list[int]], list[list[int]], list[list[int]], npt.NDArray]:
    """Remove randomly-selected living cells and record recovery statistics.

    The function repeatedly removes a fraction of currently alive cells at
    random (without replacement) and measures recovery statistics for each
    removal fraction across multiple replications.

    Args:
    ----
        grid: Grid instance to operate on; mutated during simulation steps.
        delay: Warm-up steps to run before starting removals.
        max_time: Maximum number of steps to simulate after each removal.
        target: Target image tensor used for MSE calculations.
        seed: RNG seed for deterministic random removals.

    Returns:
    -------
        A tuple ``(mse_times, count_times, surviving_cells, removed_arr)`` with
        the same layout as :func:`blob_destruction`.

    """
    # Warm up
    grid.clear_and_seed(grid_idx=0)
    grid.set_batch([0])
    grid.load_batch_from_pool()
    empty = grid.empty
    grid_state = grid.run_simulation(delay).squeeze(0)

    # Give 10% leeway in MSE calculation
    target_loss = loss(grid_state, target) * 1.1
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
        int,
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
    """Write survival or recovery statistics to a CSV file.

    The CSV will have a header row ``Removed,Iteration_1,...`` and each
    subsequent row corresponds to a removal count and the per-replication
    statistic for that removal level.

    Args:
        path: Filesystem path where CSV will be written (overwritten if exists).
        removed: 1-D numpy array of removal counts for each column.
        stats: List of per-replication lists; each inner list must have the
            same length as ``removed``.

    """
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["Removed"] + [f"Iteration_{i + 1}" for i in range(len(stats))])

        # Write survival times data
        for idx, remove_count in enumerate(removed):
            row = [remove_count] + [stats[i][idx] for i in range(len(stats))]
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return the populated namespace.

    The function constructs an ArgumentParser, registers the expected
    positional arguments and returns the result of ``parse_args()`` (an
    ``argparse.Namespace`` instance).

    Returns:
        An ``argparse.Namespace`` containing parsed command-line values.

    """
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

    return parser.parse_args()  # pyright: ignore[reportReturnType]


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
            grid,
            delay,
            max_time,
            target,
        )
    elif arg_type == "blob":
        mse_times, count_times, surviving_cells, removed = blob_destruction(
            grid,
            delay,
            max_time,
            target,
        )
    else:  # pragma: no cover
        arg_type_error = f"Unknown arg_type: {arg_type}"
        raise ValueError(arg_type_error)

    # Save results to CSV
    csv_folder = data_path / "MCAF"
    csv_folder.mkdir(parents=True, exist_ok=True)

    ## Surviving cell count.
    survive_path = csv_folder / f"{arg_type}_{weight_path.stem}_survived.csv"
    save_stats(survive_path, removed, surviving_cells)

    ## MSE based regeneration speed.
    survive_path = csv_folder / f"{arg_type}_{weight_path.stem}_mse.csv"
    save_stats(survive_path, removed, mse_times)

    # Count based regeneration speed.
    survive_path = csv_folder / f"{arg_type}_{weight_path.stem}_count.csv"
    save_stats(survive_path, removed, count_times)


if __name__ == "__main__":
    main()

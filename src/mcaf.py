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

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from grid import Grid
from utils import data_path, load_weights

removal_iterations = 100


def _simulate_to_end(grid: Grid, max_time: int) -> int:
    survival_time = 0
    while survival_time < max_time and (
        np.any(grid.batch_state.squeeze(0).detach().numpy()[0, :, :] > 0)
    ):
        grid.run_simulation_batch(1)
        survival_time += 1
    return survival_time


# def channel_destruction(
#     grid: Grid,
#     delay: int,
#     max_time: int,
#     channel: int,
#     *,
#     border_mod: float = 0.1,
#     seed: int = 43,
# ) -> tuple[list[list[int]], list[list[int]], npt.NDArray]:
#     grid = grid.deepcopy()
#     grid_state = grid.run_simulation(delay)

#     grid_state_numpy = grid_state.detach().numpy()
#     selected_channel = grid_state_numpy[:, :, channel] > border_mod

#     total = np.sum(selected_channel)
#     indices = np.where(selected_channel)

#     iterations = 10

#     survival_times = [[] for _ in range(iterations)]
#     surviving_cells = [[] for _ in range(iterations)]
#     gen = np.random.Generator(np.random.PCG64(seed))
#     for i in range(iterations):
#         for ratio in tqdm(np.linspace(0, 1, 50)):
#             local_grid = grid.deepcopy()

#             # Generate indices to be removed
#             removals = int(total * ratio)
#             removed_indices = gen.choice(
#                 range(len(indices[0])), removals, replace=False
#             )
#             removed_row = indices[0][removed_indices]
#             removed_column = indices[1][removed_indices]

#             # Remove the random cells
#             for r, c in zip(removed_row, removed_column, strict=True):
#                 local_grid.state()[r, c, channel] = 0

#             survival_time = _simulate_to_end(local_grid, max_time)
#             survival_times[i].append(survival_time)

#             surviving_cells[i].append(np.sum(local_grid.state(layer=0) > 0))
#     removed = (np.linspace(0, 1, 50) * total).astype(int)
#     return survival_times, surviving_cells, removed


# def channel_mask_destruction(
#     grid: Grid,
#     delay: int,
#     max_time: int,
#     channel: int,
#     *,
#     border_mod: float = 0.1,
#     seed: int = 43,
# ) -> tuple[list[list[int]], list[list[int]], npt.NDArray]:
#     grid = grid.deepcopy()
#     empty = grid.empty
#     grid_state = grid.run_simulation(delay)

#     grid_state_numpy = grid_state.detach().numpy()
#     selected_channel = grid_state_numpy[:, :, abs(channel)]
#     if channel >= 0:
#         alive = selected_channel > border_mod
#     else:
#         alive = selected_channel < border_mod

#     total = np.sum(alive)
#     indices = np.where(alive)

#     iterations = 10

#     survival_times = [[] for _ in range(iterations)]
#     surviving_cells = [[] for _ in range(iterations)]
#     gen = np.random.Generator(np.random.PCG64(seed))
#     for i in range(iterations):
#         for ratio in tqdm(np.linspace(0, 1, 50)):
#             local_grid = grid.deepcopy()

#             # Generate indices to be removed
#             removals = int(total * ratio)
#             removed_indices = gen.choice(
#                 range(len(indices[0])), removals, replace=False
#             )
#             removed_row = indices[0][removed_indices]
#             removed_column = indices[1][removed_indices]

#             # Remove the random cells
#             for r, c in zip(removed_row, removed_column, strict=True):
#                 local_grid.set_cell_state(c, r, empty)

#             survival_time = _simulate_to_end(local_grid, max_time)
#             survival_times[i].append(survival_time)

#             surviving_cells[i].append(np.sum(local_grid.state(layer=0) > 0))
#     removed = (np.linspace(0, 1, 50) * total).astype(int)
#     return survival_times, surviving_cells, removed


# def find_start_flood(grid: Grid, gen: np.random.Generator) -> tuple[int, int]:
#     state = grid.state()
#     indices = np.where(state[:, :, 0] > 0)
#     removed_index = int(gen.choice(range(len(indices[0])), 1)[0])
#     row = int(indices[0][removed_index])
#     column = int(indices[1][removed_index])
#     return (row, column)


# def flood_fill_step(queue: list[tuple[int, int]], grid: Grid, gen: np.random.Generator):
#     state = grid.state()

#     if queue == []:
#         row, column = find_start_flood(grid, gen)
#     else:
#         row, column = queue.pop()
#         while (state[row, column, 0] == 0) and (queue != []):
#             row, column = queue.pop()

#         if queue == []:
#             row, column = find_start_flood(grid, gen)

#     grid.set_cell_state(column, row, grid.empty)

#     # Check all surrounding cells
#     for row_offset in range(-1, 2):
#         for column_offset in range(-1, 2):
#             new_row = row + row_offset
#             new_column = column + column_offset

#             # If cell is alive
#             if state[new_row, new_column, 0] > 0:
#                 queue.insert(0, (new_row, new_column))


# def blob_destruction(
#     grid: Grid, delay: int, max_time: int, *, seed: int = 43
# ) -> tuple[list[list[int]], list[list[int]], npt.NDArray]:
#     grid = grid.deepcopy()
#     gen = np.random.Generator(np.random.PCG64(seed))

#     local_state = grid.run_simulation(delay)
#     alpha = local_state.detach().numpy()[:, :, 0]
#     alive = alpha > 0

#     total = np.sum(alive)

#     iterations = 10

#     survival_times = [[] for _ in range(iterations)]
#     surviving_cells = [[] for _ in range(iterations)]
#     for i in tqdm(range(iterations)):
#         local_grid = grid.deepcopy()

#         queue = [find_start_flood(local_grid, gen)]

#         last_removed = 0
#         for ratio in np.linspace(0, 1, 50):
#             removals = int(total * ratio) - last_removed
#             for _ in range(removals):
#                 flood_fill_step(queue, local_grid, gen)
#             last_removed += removals

#             local_grid_copy = local_grid.deepcopy()
#             survival_time = _simulate_to_end(local_grid_copy, max_time)
#             survival_times[i].append(survival_time)

#             surviving_cells[i].append(np.sum(local_grid_copy.state(layer=0) > 0))
#     removed = (np.linspace(0, 1, 50) * total).astype(int)
#     return survival_times, surviving_cells, removed


def random_destruction(
    grid: Grid, delay: int, max_time: int, *, seed: int = 43
) -> tuple[list[list[int]], list[list[int]], npt.NDArray]:
    # Warm up
    grid.clear_and_seed(grid_idx=0)
    grid.set_batch([0])
    grid.load_batch_from_pool()
    empty = grid.empty
    grid_state = grid.run_simulation(delay).squeeze(0)
    grid.write_batch_back_to_pool()
    print(np.sum(grid.batch_state.detach().squeeze(0).numpy()[0, :, :] > 0))

    alpha = grid_state.detach().numpy()[0, :, :]
    alive = alpha > 0

    total = np.sum(alive)
    indices = np.where(alive)
    iterations = 1

    survival_times = [[] for _ in range(iterations)]
    surviving_cells = [[] for _ in range(iterations)]
    gen = np.random.Generator(np.random.PCG64(seed))
    for i in tqdm(range(iterations)):
        for ratio in np.linspace(0, 1, removal_iterations):
            grid.set_batch([0])
            grid.load_batch_from_pool()
            # Generate indices to be removed
            removals = int(total * ratio)
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

            survival_time = _simulate_to_end(grid, max_time)
            survival_times[i].append(survival_time)

            surviving_cells[i].append(
                np.sum(grid.batch_state.detach().squeeze(0).numpy()[0, :, :] > 0)
            )
    removed = (np.linspace(0, 1, 50) * total).astype(int)
    return survival_times, surviving_cells, removed


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
        "testing_time",
        help="The amount of steps taken after removal occurs.",
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
        default=0,
    )

    return parser.parse_args()  # pyright: ignore[reportReturnType]def main():


def save_stats(path: Path, removed: npt.NDArray, stats: list[list[int]]):
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["Removed"] + [f"Iteration_{i + 1}" for i in range(len(stats))])

        # Write survival times data
        for idx, remove_count in enumerate(removed):
            row = [remove_count] + [stats[i][idx] for i in range(len(stats))]
            writer.writerow(row)


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
    channel = args.channel  # pyright: ignore[reportAttributeAccessIssue]
    max_time = args.testing_time  # pyright: ignore[reportAttributeAccessIssue]

    if arg_type == "random":
        survival_times, surviving_cells, removed = random_destruction(
            grid, delay, max_time
        )
    # elif arg_type == "channel":
    #     survival_times, surviving_cells, removed = channel_destruction(
    #         grid, delay, max_time, channel
    #     )
    # elif arg_type == "blob":
    #     survival_times, surviving_cells, removed = blob_destruction(
    #         grid, delay, max_time
    #     )
    # elif arg_type == "channel_mask":
    #     survival_times, surviving_cells, removed = channel_mask_destruction(
    #         grid, delay, max_time, channel, seed=seed
    #     )
    else:  # pragma: no cover
        raise ValueError(f"Unknown arg_type: {arg_type}")

    # Save results to CSV
    csv_folder = data_path / "MCAF"
    csv_folder.mkdir(parents=True, exist_ok=True)
    survive_path = csv_folder / f"{arg_type}_{weight_path.stem}_count.csv"
    save_stats(survive_path, removed, surviving_cells)
    survive_path = csv_folder / f"{arg_type}_{weight_path.stem}_time.csv"
    save_stats(survive_path, removed, survival_times)


if __name__ == "__main__":
    main()

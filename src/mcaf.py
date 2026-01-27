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
from pathlib import Path

import numpy as np
from tqdm import tqdm

from grid import Grid
from utils import load_weights


def _simulate_to_end(grid: Grid, max_time: int) -> int:
    survival_time = 0
    while survival_time < max_time and np.any(grid.state(layer=0) > 0):
        grid.step()
        survival_time += 1
    return survival_time


def channel_destruction(
    grid: Grid,
    delay: int,
    max_time: int,
    channel: int,
    *,
    border_mod: float = 0.1,
    seed: int = 43,
) -> None:
    grid = grid.deepcopy()
    grid_state = grid.run_simulation(delay)

    grid_state_numpy = grid_state.detach().numpy()
    selected_channel = grid_state_numpy[:, :, channel] > border_mod

    total = np.sum(selected_channel)
    indices = np.where(selected_channel)

    survival_times = []
    surviving_cells = []
    for ratio in tqdm(np.linspace(0, 1, 50)):
        local_grid = grid.deepcopy()
        gen = np.random.Generator(np.random.PCG64(seed))

        # Generate indices to be removed
        removals = int(total * ratio)
        removed_indices = gen.choice(range(len(indices[0])), removals, replace=False)
        removed_row = indices[0][removed_indices]
        removed_column = indices[1][removed_indices]

        # Remove the random cells
        for r, c in zip(removed_row, removed_column, strict=True):
            local_grid.state()[r, c, channel] = 0

        survival_time = _simulate_to_end(local_grid, max_time)
        survival_times.append(survival_time)

        surviving_cells.append(np.sum(local_grid.state(layer=0) > 0))
    print(survival_times)


def channel_mask_destruction(
    grid: Grid,
    delay: int,
    max_time: int,
    channel: int,
    *,
    border_mod: float = 0.1,
    seed: int = 43,
) -> None:
    grid = grid.deepcopy()
    empty = grid.empty
    grid_state = grid.run_simulation(delay)

    grid_state_numpy = grid_state.detach().numpy()
    selected_channel = grid_state_numpy[:, :, abs(channel)]
    if channel >= 0:
        alive = selected_channel > border_mod
    else:
        alive = selected_channel < border_mod

    total = np.sum(alive)
    indices = np.where(alive)

    survival_times = []
    surviving_cells = []
    for ratio in tqdm(np.linspace(0, 1, 50)):
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

        survival_time = _simulate_to_end(local_grid, max_time)
        survival_times.append(survival_time)

        surviving_cells.append(np.sum(local_grid.state(layer=0) > 0))
    print(survival_times)


def find_start_flood(grid: Grid, gen: np.random.Generator) -> tuple[int, int]:
    state = grid.state()
    indices = np.where(state[:, :, 0] > 0)
    removed_index = int(gen.choice(range(len(indices[0])), 1)[0])
    row = int(indices[0][removed_index])
    column = int(indices[1][removed_index])
    return (row, column)


def flood_fill_step(queue: list[tuple[int, int]], grid: Grid, gen: np.random.Generator):
    state = grid.state()

    if queue == []:
        row, column = find_start_flood(grid, gen)
    else:
        row, column = queue.pop()
        while (state[row, column, 0] == 0) and (queue != []):
            row, column = queue.pop()

        if queue == []:
            row, column = find_start_flood(grid, gen)

    grid.set_cell_state(column, row, grid.empty)

    # Check all surrounding cells
    for row_offset in range(-1, 2):
        for column_offset in range(-1, 2):
            new_row = row + row_offset
            new_column = column + column_offset

            # If cell is alive
            if state[new_row, new_column, 0] > 0:
                queue.insert(0, (new_row, new_column))


def blob_destruction(grid: Grid, delay: int, max_time: int, *, seed: int = 43) -> None:
    grid = grid.deepcopy()
    gen = np.random.Generator(np.random.PCG64(seed))

    local_state = grid.run_simulation(delay)
    alpha = local_state.detach().numpy()[:, :, 0]
    alive = alpha > 0

    total = np.sum(alive)

    iterations = 10

    survival_times = [[] for _ in range(iterations)]
    surviving_cells = [[] for _ in range(iterations)]
    for i in tqdm(range(iterations)):
        local_grid = grid.deepcopy()

        queue = [find_start_flood(local_grid, gen)]

        last_removed = 0
        for ratio in np.linspace(0, 1, 50):
            removals = int(total * ratio) - last_removed
            for _ in range(removals):
                flood_fill_step(queue, local_grid, gen)
            last_removed += removals

            local_grid_copy = local_grid.deepcopy()
            survival_time = _simulate_to_end(local_grid_copy, max_time)
            survival_times[i].append(survival_time)

            surviving_cells[i].append(np.sum(local_grid_copy.state(layer=0) > 0))
    mean_survival_times = np.mean(survival_times, axis=0)
    print(mean_survival_times)


def random_destruction(
    grid: Grid, delay: int, max_time: int, *, seed: int = 43
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
    surviving_cells = []
    for ratio in tqdm(np.linspace(0, 1, 50)):
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

        survival_time = _simulate_to_end(local_grid, max_time)
        survival_times.append(survival_time)

        surviving_cells.append(np.sum(local_grid.state(layer=0) > 0))
    print(surviving_cells)


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


def main() -> None:
    seed = 43

    args = parse_args()

    weight_path = Path(args.weights)  # pyright: ignore[reportAttributeAccessIssue]
    weights = load_weights(weight_path)

    num_channels = weights[1].shape[1]  # Number of outputs of the NN
    grid = Grid(50, 50, num_channels, seed=seed, weights=weights)

    delay = args.delay  # pyright: ignore[reportAttributeAccessIssue]
    arg_type = args.type  # pyright: ignore[reportAttributeAccessIssue]
    channel = args.channel  # pyright: ignore[reportAttributeAccessIssue]
    max_time = args.testing_time  # pyright: ignore[reportAttributeAccessIssue]

    if arg_type == "random":
        random_destruction(grid, delay, max_time)
    elif arg_type == "channel":
        channel_destruction(grid, delay, max_time, channel)
    elif arg_type == "blob":
        blob_destruction(grid, delay, max_time)
    elif arg_type == "channel_mask":
        channel_mask_destruction(grid, delay, max_time, channel, seed=seed)


if __name__ == "__main__":
    main()

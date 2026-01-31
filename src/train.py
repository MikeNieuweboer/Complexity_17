"""Allows for training a neural network for NCA using backpropagation.

Group:      17
Course:     Complex System Simulation

Description
-----------
Is used for training a NCA to grow to a certain target pattern. The main training loop
consists of storing a pool of grid states and itteratively picking random batches from
this pool. The grid states in the batch will be sorted on how well they fit the target
pattern, after which the best two grids will be damaged by applying a circular mask to
damage parts of the grid (entire state vector of the cells is set to 0). Additionally
the worst grid in the batch is cleared and re-seeded (as to make the NN not forget how
to grow from a single seed). Finally all grids in the batch will be ran for a certain
number of steps (determined by range x-y), and backpropagation is applied using MSE as
loss.

Notes
-----
- Argument setting for the training loop is still a bit crude, and can be done by alte-
ring the source code in the main() loop. Comments are in place to guide in this process.
- To store information (a loss csv file, and figures of the alpha channel of the batch
at certain steps) a "training" folder will be created in the parent directory of this
file.
- During the training loop, whenever new best weights are found, they are stored in a
"weights" folder, which if not present will be created in the parent directory of this
file.

AI usage
--------
Gemini 3.0 was used to generate docstrings for the functionality in this file.
This was done for each function individually using the inline chat and the
following prompt:
> Analyze the specific function and create a consise docstring
Afterwards manual changes were made to the liking of the author.

Acknowledgements
----------------
The ideas behind the persistent training are heavily inspired by the wonderfull
article:
> Mordvintsev, et al., "Growing Neural Cellular Automata", Distill, 2020.
Link: https://distill.pub/2020/growing-ca/

"""

import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torch.nn import functional
from tqdm import trange

from grid import Grid
from nn import NN
from plotting import plot_heatmaps
from utils import load_target_image

# create directories for loading and saving
ROOT_DIR = Path(__file__).parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)
TRAINING_DIR = ROOT_DIR / "training"
TRAINING_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)


# local timezone as variable to use in filenames
LOCAL_TIMEZONE = ZoneInfo("Europe/Amsterdam")


def damage_batch_bchw(
    states: torch.Tensor,
    radius_range: tuple[float, float] = (0.1, 0.4),
) -> torch.Tensor:
    """Apply a circular damage mask to a batch of states (zeros out a region).

    Args:
        states: (B, C, H, W) tensor
        radius_range: (min, max) radius as fraction of image width

    Returns:
        Damaged states tensor (B, C, H, W).

    """
    if states.dim() != 4:
        raise ValueError(f"states must be 4D (B,C,H,W), got dim={states.dim()}")

    B, C, H, W = states.shape
    r_min, r_max = radius_range
    device = states.device

    damaged = states.clone()

    # base grid (1, H, W)
    y = torch.arange(H, device=device)
    x = torch.arange(W, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)

    # random centers (B, 1, 1)
    center_x = (torch.rand(B, device=device) * W).view(B, 1, 1)
    center_y = (torch.rand(B, device=device) * H).view(B, 1, 1)

    # random radius (B, 1, 1) in pixels
    radius_frac = torch.rand(B, device=device) * (r_max - r_min) + r_min
    radius_px = (radius_frac * W).view(B, 1, 1)

    # -> (B, H, W)
    dist_sq = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2

    # remove inside circle: (B, 1, H, W)
    mask = (dist_sq > radius_px**2).unsqueeze(1)

    damaged *= mask.to(damaged.dtype)
    return damaged


def generate_circle_target_chw(grid_size: int, device: torch.device) -> torch.Tensor:
    """Generate a square grid containing a filled circle centered in the middle.

    The circle has a value of 1.0, while the background is 0.0. The radius is fixed
    at 30% of the grid size.

    Args:
        grid_size (int): The height and width of the square grid.
        device (torch.device): The PyTorch device (CPU or GPU) to create the tensor on.

    Returns:
        torch.Tensor: A 2D floating-point tensor of shape (grid_size, grid_size)
        representing the target mask.

    """
    target = torch.zeros((grid_size, grid_size), device=device, dtype=torch.float32)

    yy = torch.arange(grid_size, device=device).view(-1, 1)
    xx = torch.arange(grid_size, device=device).view(1, -1)

    cx = grid_size // 2
    cy = grid_size // 2
    radius = grid_size * 0.3

    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    target[mask] = 1.0
    return target


def train(
    grid_size: int = 50,
    n_channels: int = 8,
    hidden_size: int = 64,
    steps: int = 1000,
    min_steps: int = 50,
    max_steps: int = 90,
    lr: float = 2e-3,
    update_prob: float = 0.5,
    masking_th: float = 0.1,
    pool_size: int = 64,
    batch_size: int = 8,
    plot_steps: list[int] | None = None,
    n_to_damage: int = 2,
    damage_radius_range: tuple[float, float] = (0.1, 0.2),
    target_pattern: torch.Tensor | None = None,
) -> None:
    """Trains a Neural Cellular Automata (NCA) model to grow a specific target pattern.

    This function manages the training loop using a persistent sample pool strategy. At
    each step, it samples a batch, sorts by loss to prioritize damaging high-performing
    samples (robustness) and replacing low-performing ones, evolves the states, and
    updates the neural network weights based on Mean Squared Error (MSE) loss against
    the target.

    Args:
        grid_size (int): Height and width of the simulation grid. Defaults to 50.
        n_channels (int): Number of state channels (alpha + hidden). Defaults to 8.
        hidden_size (int): Size of the hidden layer in the 1x1 Conv neural network.
            Defaults to 64.
        steps (int): Total number of training iterations. Defaults to 1000.
        min_steps (int): Minimum number of simulation steps per forward pass.
            Defaults to 50.
        max_steps (int): Maximum number of simulation steps per forward pass.
            Defaults to 90.
        lr (float): Learning rate for the Adam optimizer. Defaults to 2e-3.
        update_prob (float): Probability of a cell updating during a step
            (stochasticity). Defaults to 0.5.
        masking_th (float): Threshold for the "alive" masking logic. Defaults to 0.1.
        pool_size (int): Total size of the persistent state pool. Defaults to 64.
        batch_size (int): Number of samples processed per training step. Defaults to 8.
        plot_steps (list[int] | None): Specific step indices to generate heatmap
            visualizations. Defaults to None.
        n_to_damage (int): Number of best-performing batch samples to apply damage to.
            Defaults to 2.
        damage_radius_range (tuple[float, float]): Range relative to grid size for
            circular damage masks. Defaults to (0.1, 0.2).
        target_pattern (torch.Tensor | None): The 2D target tensor (H, W). If None,
            generates a default circle (radius 0.3 of grid_size).

    Returns:
        None: Weights are saved to disk and loss metrics are logged to CSV.

    """
    # create data folder in training folder for this specific training
    curtime = datetime.now(tz=LOCAL_TIMEZONE).strftime("%Y%m%d-%H%M%S")
    data_dir = TRAINING_DIR / curtime
    Path.mkdir(data_dir, exist_ok=True, parents=True)

    # create filename for weights
    best_weights_path = WEIGHTS_DIR / f"Gr{grid_size}-Ch{n_channels}-Hi{hidden_size}_{curtime}.pt"

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # target logic, defaults to circle if no target_pattern is given.
    if target_pattern is not None:
        target = target_pattern.to(device)
        if target.shape != (grid_size, grid_size):
            raise ValueError(f"grid has size ({grid_size}, {grid_size}), target_pattern has shape {target.shape}.")
    else:
        # standard circle
        target = generate_circle_target_chw(grid_size, device)

    # unqueeze for broadcasting later
    target_alpha = target.unsqueeze(0) # (H, W) -> (1, H, W)

    # grid with internal pool + batch buffer (B, C, H, W)
    grid = Grid(
        poolsize=pool_size,
        batch_size=batch_size,
        num_channels=n_channels,
        width=grid_size,
        height=grid_size,
        device=device,
    )
    grid.NN = NN(n_channels, hidden_size).to(device)

    # seed entire pool once
    grid.clear_and_seed(grid_idx=torch.arange(pool_size, device=device), in_batch=False)

    # store best loss and initialize optimizer
    best_loss = float("inf")
    optimizer = torch.optim.Adam(grid.NN.parameters(), lr=lr)

    # open csv file to write loss
    csv_path = data_dir / "loss.csv"
    with Path.open(csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["step", "loss"])

        t = trange(1, steps+1, desc="Training", leave=True)
        for step_idx in t:
            # sample indices + load batch from pool into grid._batch (B, C, H, W)
            idxs = grid.sample_batch()  # idxs on CPU
            batch = grid.batch_state  # (B, C, H, W)

            # sort batch by ascending loss (best first)
            with torch.no_grad():
                target_batch = target_alpha.unsqueeze(0).expand(batch_size, 1, grid_size, grid_size)  # (B,1,H,W)
                losses = functional.mse_loss(batch[:, 0:1], target_batch, reduction="none")  # (B,1,H,W)
                loss_per_sample = losses.mean(dim=(1, 2, 3))  # (B,)
                sort_idx = torch.argsort(loss_per_sample)  # on device

                # reorder batch and indices to match (best -> worst)
                batch = batch[sort_idx]
                idxs = idxs[sort_idx.detach().cpu()]
                grid._batch = batch
                grid._batch_idxs = idxs  # keep mapping correct for write-back

            # damage top n batch states
            if n_to_damage > 0:
                grid._batch[:n_to_damage] = damage_batch_bchw(
                    grid._batch[:n_to_damage],
                    radius_range=damage_radius_range,
                )

            # reseed worst individual in the batch
            grid.clear_and_seed(grid_idx=torch.tensor([batch_size - 1], device=device), in_batch=True)

            # evolve the whole batch together
            n_steps = int(torch.randint(min_steps, max_steps + 1, (1,), device=device).item())
            grid.run_simulation_batch(
                steps=n_steps,
                update_prob=update_prob,
                masking_th=masking_th,
                activate_print=False,
            )

            final_batch = grid.batch_state  # (B, C, H, W)

            # calculate batch loss on alpha channel
            target_batch = target_alpha.unsqueeze(0).expand(batch_size, 1, grid_size, grid_size)
            loss = functional.mse_loss(final_batch[:, 0:1], target_batch)

            # save best weights (.pt file)
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(
                    (
                        grid.NN.hidden_layer.weight.detach().cpu(),
                        grid.NN.output_layer.weight.detach().cpu(),
                    ),
                    best_weights_path,
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # gradient L2 normalization
            for p in grid.NN.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)

            optimizer.step()

            # persist evolved batch back into pool of Grid
            grid.write_batch_back_to_pool()

            # append loss to trange
            t.set_postfix(loss=f"{loss.item():.6f}")

            # add loss to csv
            csv_writer.writerow([step_idx, loss.item()])
            csv_file.flush()

            if plot_steps and step_idx in plot_steps:
                plot_heatmaps(grid.batch_state.detach()[:, 0, :, :].permute(1, 2, 0),
                            data_dir / f"step_{str(step_idx).zfill(len(str(steps)))}",
                            "Batch Index",
                            f"Alpha Channel of Batch {step_idx}",
                            )


def main() -> None:
    # HERE YOU CAN CHANGE THE DIRECTORY TO LOAD A TARGET IMAGE
    target = load_target_image(DATA_DIR / "targets" / "amongus.png", grid_size=50)

    # DEFINE THE TOTAL NUMBER OF STEPS AND WHEN TO PLOT BATCH
    n_steps = 2000
    plot_steps = [1, *list(range(10, 101, 10)), *list(range(150, n_steps + 1, 50))]

    # CHANGE PARAMS TO LIKING
    params = {
        "grid_size": 50,
        "n_channels": 8,
        "hidden_size": 64,
        "steps": n_steps,
        "min_steps": 50,
        "max_steps": 90,
        "lr": 2e-3,
        "update_prob": 0.5,
        "masking_th": 0.1,
        "pool_size": 64,
        "batch_size": 8,
        "plot_steps": plot_steps,
        "n_to_damage": 2,
        "damage_radius_range": (0.1, 0.2),
        "target_pattern": target,
    }
    train(**params)


if __name__ == "__main__":
    main()

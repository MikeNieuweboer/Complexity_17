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


# set local timezone as variable to use in filenames
LOCAL_TIMEZONE = ZoneInfo("Europe/Amsterdam")


def damage_batch_bchw(
    states: torch.Tensor,
    *,
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

    # Base grid (1, H, W)
    y = torch.arange(H, device=device)
    x = torch.arange(W, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)

    # Random centers (B, 1, 1)
    center_x = (torch.rand(B, device=device) * W).view(B, 1, 1)
    center_y = (torch.rand(B, device=device) * H).view(B, 1, 1)

    # Random radius (B, 1, 1) in pixels
    radius_frac = torch.rand(B, device=device) * (r_max - r_min) + r_min
    radius_px = (radius_frac * W).view(B, 1, 1)

    # (B, H, W)
    dist_sq = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2

    # Keep outside circle: (B, 1, H, W)
    mask = (dist_sq > radius_px**2).unsqueeze(1)

    damaged *= mask.to(damaged.dtype)
    return damaged


def generate_circle_target_chw(grid_size: int, device: torch.device) -> torch.Tensor:
    """Create (H, W) target with a filled circle."""
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
    """Train the NN inside Grid to form a circular alpha pattern, using Grid's internal pool+batch."""

    # create data folder in training folder for specific training
    curtime = datetime.now(tz=LOCAL_TIMEZONE).strftime("%Y%m%d-%H%M%S")
    data_dir = TRAINING_DIR / curtime
    Path.mkdir(data_dir, exist_ok=True, parents=True)

    # create filename for weights
    best_weights_path = WEIGHTS_DIR / f"Gr{grid_size}-Ch{n_channels}-Hi{hidden_size}_{curtime}.pt"

    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # target logic
    if target_pattern is not None:
        target = target_pattern.to(device)
        if target.shape != (grid_size, grid_size):
            raise ValueError(f"grid has size ({grid_size}, {grid_size}), target_pattern has shape {target.shape}.")
    else:
        # standard circle
        target = generate_circle_target_chw(grid_size, device)

    # unqueeze for broadcasting later
    target_alpha = target.unsqueeze(0) # (H, W) -> (1, H, W)

    # Grid with internal pool + batch buffer (B, C, H, W)
    grid = Grid(
        poolsize=pool_size,
        batch_size=batch_size,
        num_channels=n_channels,
        width=grid_size,
        height=grid_size,
        device=device,
    )
    grid.NN = NN(n_channels, hidden_size).to(device)

    # Seed entire pool once
    grid.clear_and_seed(grid_idx=torch.arange(pool_size, device=device), in_batch=False)


    best_loss = float("inf")
    optimizer = torch.optim.Adam(grid.NN.parameters(), lr=lr)

    # open csv file to write loss
    csv_path = data_dir / "loss.csv"
    with Path.open(csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["step", "loss"])

        t = trange(1, steps+1, desc="Training", leave=True)
        for step_idx in t:
            # Sample indices + load batch from pool into grid._batch (B, C, H, W)
            idxs = grid.sample_batch()  # idxs on CPU
            batch = grid.batch_state  # (B, C, H, W)

            # Sort batch by ascending loss (best first)
            with torch.no_grad():
                target_batch = target_alpha.unsqueeze(0).expand(batch_size, 1, grid_size, grid_size)  # (B,1,H,W)
                losses = functional.mse_loss(batch[:, 0:1], target_batch, reduction="none")  # (B,1,H,W)
                loss_per_sample = losses.mean(dim=(1, 2, 3))  # (B,)
                sort_idx = torch.argsort(loss_per_sample)  # on device

                # Reorder batch and indices to match (best -> worst)
                batch = batch[sort_idx]
                idxs = idxs[sort_idx.detach().cpu()]
                grid._batch = batch
                grid._batch_idxs = idxs  # keep mapping correct for write-back

            # Damage top n (best-performing) batch states
            if n_to_damage > 0:
                grid._batch[:n_to_damage] = damage_batch_bchw(
                    grid._batch[:n_to_damage],
                    radius_range=damage_radius_range,
                )

            # Force reseed worst individual in the batch
            grid.clear_and_seed(grid_idx=torch.tensor([batch_size - 1], device=device), in_batch=True)

            # Evolve the whole batch together
            n_steps = int(torch.randint(min_steps, max_steps + 1, (1,), device=device).item())
            grid.run_simulation_batch(
                steps=n_steps,
                update_prob=update_prob,
                masking_th=masking_th,
                activate_print=False,
            )

            final_batch = grid.batch_state  # (B, C, H, W)

            # Batch loss on alpha channel
            target_batch = target_alpha.unsqueeze(0).expand(batch_size, 1, grid_size, grid_size)
            loss = functional.mse_loss(final_batch[:, 0:1], target_batch)

            # Save best weights (Conv2d weight format)
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

            # Gradient L2 normalization
            for p in grid.NN.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)

            optimizer.step()

            # Persist evolved batch back into Grid's pool
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
    target = load_target_image(DATA_DIR / "targets" / "amongus.png", grid_size=50)
    plot_steps = [1, *list(range(10, 101, 10)), *list(range(150, 2001, 50))]

    params = {
        "grid_size": 50,
        "n_channels": 8,
        "hidden_size": 64,
        "steps": 2000,
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

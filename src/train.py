from pathlib import Path

import torch
from torch.nn import functional
from tqdm import trange

from grid import Grid
from nn import NN

root_dir = Path(__file__).parent.parent
weights_dir = root_dir / "weights"
weights_dir.mkdir(exist_ok=True, parents=True)


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


def generate_circle_target_chw(grid_size: int, n_channels: int, device: torch.device) -> torch.Tensor:
    """Create (C, H, W) target with a filled circle in the alpha channel (channel 0)."""
    target = torch.zeros((n_channels, grid_size, grid_size), device=device, dtype=torch.float32)

    yy = torch.arange(grid_size, device=device).view(-1, 1)
    xx = torch.arange(grid_size, device=device).view(1, -1)

    cx = grid_size // 2
    cy = grid_size // 2
    radius = grid_size * 0.3

    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    target[0, mask] = 1.0
    return target


def train(
    grid_size: int = 30,
    n_channels: int = 5,
    hidden_size: int = 32,
    steps: int = 1000,
    min_steps: int = 64,
    max_steps: int = 96,
    lr: float = 1e-3,
    update_prob: float = 0.5,
    masking_th: float = 0.1,
    pool_size: int = 64,
    batch_size: int = 16,
    log_interval: int = 50,
    n_to_damage: int = 3,
    damage_radius_range: tuple[float, float] = (0.1, 0.4),
) -> None:
    """Train the NN inside Grid to form a circular alpha pattern, using Grid's internal pool+batch."""
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(device)

    # Target alpha: (1, H, W)
    target = generate_circle_target_chw(grid_size, n_channels, device)
    target_alpha = target[0:1]  # (1, H, W)

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

    # For saving best weights
    best_loss = float("inf")
    best_weights_path = weights_dir / "TRAIN_BEST_TENS.pt"

    optimizer = torch.optim.Adam(grid.NN.parameters(), lr=lr)

    for step_idx in range(1, steps + 1):
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

        if step_idx % log_interval == 0 or step_idx == 1:
            print(f"step={step_idx} n_steps={n_steps} loss={loss.item():.6f}")

    # Visualize best weights
    weights = torch.load(best_weights_path, map_location=device)
    grid.set_weights_on_nn_from_tens(weights)

    # Load any batch, reseed it, run and record history
    grid.set_batch(list(range(batch_size)))
    grid.load_batch_from_pool()
    grid.clear_and_seed(grid_idx=torch.arange(batch_size, device=device), in_batch=True)
    states = grid.run_simulation(150, record_history=True)
    # animate_heatmaps(states)


def main() -> None:
    params = {
        "grid_size": 50,
        "n_channels": 5,
        "hidden_size": 32,
        "steps": 200,
        "min_steps": 20,
        "max_steps": 20,
        "lr": 1e-3,
        "update_prob": 0.5,
        "masking_th": 0.1,
        "pool_size": 64,
        "batch_size": 16,
        "log_interval": 10,
        "n_to_damage": 3,
        "damage_radius_range": (0.1, 0.2),
    }
    train(**params)


if __name__ == "__main__":
    main()

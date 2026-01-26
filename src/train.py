from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

#from analyze import animate_heatmaps
from ea import EA
from grid import Grid
from nn import NN

root_dir = Path(__file__).parent.parent
weights_dir = root_dir / "weights"
weights_dir.mkdir(exist_ok=True, parents=True)

class Pool:
    """Store batches of states (grid tensors) for training."""

    def __init__(self, pool_size: int,
                 height: int,
                 width: int,
                 n_channels: int,
                 device: torch.device):
        self.pool_size = pool_size
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.device = device

        # Initialize pool with zeros
        self._states = torch.zeros(
            (pool_size, height, width, n_channels),
            dtype=torch.float32,
            device=device,
        )

        # Fill pool with initial seeds
        self._fill_with_seeds()

    def _fill_with_seeds(self):
        """Reset the entire pool to seed states."""
        self._states.zero_()

        # Center coordinate
        cx, cy = self.width // 2, self.height // 2

        # Seed: Alpha=0, Hidden=1+
        self._states[:, cy, cx, :] = 1.0

    def sample(self, batch_size: int):
        """Sample a batch of states from the pool.

        Returns:
            indices: Indices of the sampled states in the pool.
            batch_states: The batch of states (Batch, H, W, C).
        """
        # Random indices
        indices = np.random.choice(self.pool_size, batch_size, replace=False)
        batch_states = self._states[indices].clone()

        return indices, batch_states

    def update(self, indices, new_states):
        """Update the pool with new states."""
        self._states[indices] = new_states.detach()


def damage_batch(states: torch.Tensor, device: torch.device,
                 radius_range: tuple = (0.1, 0.4),
                 damage_prob: float = 0.2):
    """Apply a circular damage mask to a batch of states (zeros out a region).

    Args:
        states: (B, H, W, C) tensor
        device: torch device
        radius_range: tuple of (min, max) radius as fraction of image size

    """
    B, H, W, C = states.shape
    r_min, r_max = radius_range

    # copy as to not change original tensor
    damaged_states = states.clone()

    # create grid
    y_range = torch.arange(H, device=device)
    x_range = torch.arange(W, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

    for i in range(B):
        # get random damage patch center and radius
        center_x = torch.rand(1, device=device).item() * W
        center_y = torch.rand(1, device=device).item() * H

        radius_frac = (torch.rand(1, device=device).item() * (r_max - r_min)) + r_min
        radius_pixels = radius_frac * W

        # get the squared distance for every pixel at once
        dist_sq = (grid_x - center_x)**2 + (grid_y - center_y)**2

        # mask to see if the pixel lies outside the circle
        mask_2d = dist_sq > radius_pixels**2

        # convert (H, W) mask to (H, W, 1) mask to damage
        # also all channels of the grid
        mask_3d = mask_2d.unsqueeze(-1)

        damaged_states[i] *= mask_3d.float()

    return damaged_states


def generate_circle_target(grid_size: int, n_channels: int, device: torch.device) -> torch.Tensor:
    """Create (H, W, C) target with a circle in alpha channel.

    The circle has a radius 1/4 of the grid size.
    """
    target = torch.zeros((grid_size, grid_size, n_channels), device=device)

    yy = torch.arange(grid_size, device=device).view(-1, 1)
    xx = torch.arange(grid_size, device=device).view(1, -1)

    # get center
    center_x = grid_size // 2
    center_y = grid_size // 2
    radius = grid_size * 0.25

    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
    target[mask, 0] = 1.0
    return target


def train(grid_size: int = 30, n_channels: int = 5, hidden_size: int = 32,
          steps: int = 1000, min_steps: int = 64, max_steps: int = 96,
          lr: float = 1e-3, update_prob: float = 0.5, masking_th: float = 0.1,
          pool_size: int = 64, batch_size: int = 16, log_interval: int = 50,
          n_to_damage: int = 3, damage_radius_range: tuple = (0.1, 0.4)) -> None:
    """Train the neural net for the Grid, to form a cirular pattern on the grid."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create target circular pattern for given grid size.
    target = generate_circle_target(grid_size, n_channels, device)
    target_alpha = target[:, :, 0:1]

    #initialize pool of grid and neural nets
    grid = Grid(width=grid_size, height=grid_size, num_channels=n_channels, device=device)
    grid.NN = NN(n_channels, hidden_size).to(device)
    pool = Pool(pool_size, grid_size, grid_size, n_channels, device)

    # for saving best weights
    best_loss = float("inf")
    best_weights_path = weights_dir / "TRAIN_BEST_TENS.pt"

    # init optimizer
    optimizer = torch.optim.Adam(grid.NN.parameters(), lr=lr)

    # main training loop
    for step_idx in range(1, steps + 1):
        indices, batch_states = pool.sample(batch_size)

        # sort batch_states by ascending loss-value
        with torch.no_grad():
            target_batch = target_alpha.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            losses = functional.mse_loss(batch_states[:, :, :, 0:1], target_batch, reduction='none')
            loss_per_sample = losses.mean(dim=[1, 2, 3])

            sort_idx = torch.argsort(loss_per_sample)
            indices = indices[sort_idx.cpu().numpy()]
            batch_states = batch_states[sort_idx]

        # damage top batch states
        batch_states[:n_to_damage] = damage_batch(
            batch_states[:n_to_damage],
            device=device,
            radius_range=damage_radius_range,
        )


        #TODO: change grid.py to incorporate batching better, instead of
        #this primitive for loop.
        final_batch_states = []
        n_steps = int(torch.randint(min_steps, max_steps + 1, (1,), device=device).item())
        for i in range(batch_size):
            state = batch_states[i]
            grid.set_state(state)

            for _ in range(n_steps):
                grid.step(update_prob=update_prob, masking_th=masking_th)

            final_batch_states.append(grid._grid_state)

        #stack the batch again
        final_batch_tensor = torch.stack(final_batch_states)

        # Expand target to match batch size: (H, W, 1) -> (B, H, W, 1)
        # and calculate mse_loss
        target_batch = target_alpha.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        loss = functional.mse_loss(final_batch_tensor[:, :, :, 0:1], target_batch)

        # save the best weights
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
        optimizer.step()
        pool.update(indices, final_batch_tensor.detach())

        # logging the loss
        if step_idx % log_interval == 0 or step_idx == 1:
            print(f"step={step_idx} n_steps={n_steps} loss={loss.item():.6f}")


    # to visualize the best weights
    # weights = torch.load(best_weights_path, map_location=device)
    # grid.set_weights_on_nn_from_tens(weights)
    # grid.clear_and_seed()
    # states = grid.run_simulation(150, record_history=True)
    # animate_heatmaps(states)

def main() -> None:

    params = {
        "grid_size": 50,
        "n_channels": 5,
        "hidden_size": 32,
        "steps": 50,
        "min_steps": 64,
        "max_steps": 96,
        "lr": 1e-3,
        "update_prob": 0.5,
        "masking_th": 0.1,
        "pool_size": 64,
        "batch_size": 16,
        "log_interval": 2,
        "n_to_damage": 3,
        "damage_radius_range": (0.1, 0.2),
    }
    train(**params)

if __name__ == "__main__":
    main()
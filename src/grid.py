from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional

from nn import NN

if TYPE_CHECKING:
    import numpy.typing as npt
    from torch.types import Device, Tensor


class Grid:
    def __init__(
        self,
        width: int,
        height: int,
        num_channels: int,
        *,
        seed: int = 43,
        device: Device | None = None,
        weights: tuple[npt.NDArray, ...] | None = None,
    ) -> None:
        self._width = width
        self._height = height
        self._num_channels = num_channels

        self._device = device if device is not None else torch.device("cpu")

        self._weights = weights

        # Grid state: (H, W, C)
        self._grid = torch.zeros(
            (height, width, num_channels), dtype=torch.float32, device=self._device
        )

        self.NN = NN()

        self._rng = torch.Generator(device=self._device)
        self._rng.manual_seed(seed)

    def seed_center(self, seed_vector: Tensor) -> None:
        """Initialize seed state vector in center of grid."""
        self.set_cell_state(
            x=(self._width // 2), y=(self._height // 2), state_vector=seed_vector
        )

    def set_state(self, new_state: Tensor) -> None:
        pass

    def set_cell_state(self, x: int, y: int, state_vector: torch.Tensor) -> None:
        """Alter state vector at (y,x) in the grid."""
        state_vector = state_vector.to(device=self._device, dtype=self._grid.dtype)
        self._grid[y, x] = state_vector

    def step(self, update_prob: float = 0.5, masking_th: float = 0.1) -> None:
        """Perform a single step of the grid's CA."""
        if update_prob > 1.0 or update_prob < 0.0:
            msg = f"update_prob must be in [0, 1], got {update_prob}"
            raise ValueError(msg)
        if masking_th < 0.0:
            msg = f"masking_th must be >= 0, got {masking_th}"
            raise ValueError(msg)

        ### Neural network
        # Get derivative of current grid state
        state_change = self.NN.perceive(self._grid)

        ### Stochastic Update
        # Stochastic mask with a probability for each cell
        rand_mask = (
            torch.rand(
                (self._height, self._width), generator=self._rng, device=self._device
            )
            < update_prob
        ).to(self._grid.dtype)

        # Expand dimensionality to alter all channels (H,W) -> (H, W, 1)
        rand_mask = rand_mask.unsqueeze(-1)
        # Update grid stochastically
        self._grid = self._grid + state_change * rand_mask

        ### Alive Masking
        # Allows wrapped alive masking with by multithreading on GPU
        alpha = self._grid[:, :, 3:4]  # (H, W, 1)
        alpha = alpha.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
        alpha = functional.pad(alpha, (1, 1, 1, 1), mode="circular")

        alive = functional.max_pool2d(alpha, 3, stride=1, padding=0) > masking_th
        alive = alive.squeeze(0).permute(1, 2, 0)  # (H, W, 1)

        self._grid *= alive.float()

    def run_simulation(
        self,
        steps: int = 20,
        update_prob: float = 0.5,
        masking_th: float = 0.1,
    ):
        """Run the grid CA for a fixed number of steps."""
        for t in range(steps):
            state = self.step(update_prob=update_prob, masking_th=masking_th)
            # TODO: make print toggleable
            print(f"Step ({t}/{steps})\n")
        return state

    def set_weights(self, new_weights: tuple[npt.NDArray, ...]) -> None:
        self._weights = new_weights

    def deepcopy(self) -> Grid:
        """More performant alternative to the built in deepcopy."""
        copy = Grid(self._width, self._height)
        copy.set_state(self._grid.copy())
        return copy

    @property
    def state(self) -> npt.NDArray:
        if self._grid.device == "cpu":
            return self._grid.numpy()
        return self._grid.detach().numpy()

    @property
    def weights(self) -> tuple[npt.NDArray, ...] | None:
        return self._weights


def main():
    # Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W, C = 64, 64, 16
    seed = 123

    # Create grid
    grid = Grid(width=W, height=H, num_channels=C, seed=seed, device=device)

    # Seed vector for the center cell
    seed_vector = torch.zeros(C, dtype=torch.float32, device=device)
    seed_vector[0:3] = 0.0  # RGB
    seed_vector[3] = 1.0  # alpha channel
    seed_vector[4:16] = 1.0  # state vector

    # Initialize center
    grid.seed_center(seed_vector)

    # Run a few steps
    grid.run_simulation(steps=20, update_prob=0.5, masking_th=0.1)

    final_grid = grid.state


if __name__ == "__main__":
    main()

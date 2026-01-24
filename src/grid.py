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

        if weights:
            self.set_weights_on_nn(weights)

        # Grid state: (H, W, C)
        self._grid_state = torch.zeros(
            (height, width, num_channels), dtype=torch.float32, device=self._device
        )

        self._seed = seed
        self._rng = torch.Generator(device=self._device)
        self._rng.manual_seed(seed)
        seed_center = [1 for i in range(num_channels)]
        if num_channels > 1:
            seed_center[1] = 0
        self.seed_center(torch.Tensor(seed_center))

    def set_weights_on_nn(self, weights: tuple[npt.NDArray, ...]) -> None:
        """Load pre-trained weights into the neural network.

        Args:
            weights: Tuple of two numpy arrays (hidden_layer, output_layer) containing
                    the weights for the hidden and output layers respectively.

        Raises:
            ValueError: If weights tuple doesn't have exactly 2 elements.

        """
        if len(weights) != 2:  # noqa: PLR2004
            raise ValueError(  # noqa: TRY003
                "weights should have dimension 2 (hidden_layer, output_layer)",  # noqa: EM101
                f", got {len(weights)}",
            )

        # ( (3*channel x hidden_n), (hidden_n x channel) )
        hidden_layer, output_layer = weights

        # Convert to tensors and set device/dtype, and transform
        # #(In, Out) -> (Out, In, 1, 1)
        hidden_tens = torch.from_numpy(hidden_layer).to(
            self._device,
            dtype=torch.float32,
        )
        hidden_tens = hidden_tens.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
        output_tens = torch.from_numpy(output_layer).to(
            self._device,
            dtype=torch.float32,
        )
        output_tens = output_tens.permute(1, 0).unsqueeze(-1).unsqueeze(-1)

        # create NN instance and load weights
        self.NN = NN(self._num_channels, hidden_layer.shape[1]).to(self._device)
        self.NN.load_weights(hidden_tens, output_tens)

        # set weights as attribute
        self._weights = weights

    def seed_center(self, seed_vector: Tensor) -> None:
        """Initialize seed state vector in center of grid."""
        self.set_cell_state(
            x=(self._width // 2),
            y=(self._height // 2),
            state_vector=seed_vector,
        )

    def set_state(self, new_state: Tensor) -> None:
        self._grid_state = new_state

    def set_cell_state(self, x: int, y: int, state_vector: torch.Tensor) -> None:
        """Alter state vector at (y,x) in the grid."""
        state_vector = state_vector.to(
            device=self._device, dtype=self._grid_state.dtype
        )
        self._grid_state[y, x] = state_vector

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
        state_change = self.NN.forward(self._grid_state)

        ### Stochastic Update
        # Stochastic mask with a probability for each cell
        rand_mask = (
            torch.rand(
                (self._height, self._width), generator=self._rng, device=self._device
            )
            < update_prob
        ).to(self._grid_state.dtype)

        # Expand dimensionality to alter all channels (H,W) -> (H, W, 1)
        rand_mask = rand_mask.unsqueeze(-1)
        # Update grid stochastically
        self._grid_state = self._grid_state + state_change * rand_mask
        self._grid_state = self._grid_state.clamp(0, 1)

        ### Alive Masking
        # Allows wrapped alive masking with by multithreading on GPU
        alpha = self._grid_state[:, :, 0:1]  # (H, W, 1)
        alpha = alpha.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
        alpha = functional.pad(alpha, (1, 1, 1, 1), mode="circular")

        alive = functional.max_pool2d(alpha, 3, stride=1, padding=0) > masking_th
        alive = alive.squeeze(0).permute(1, 2, 0)  # (H, W, 1)

        self._grid_state *= alive.float()

    def run_simulation(
        self,
        steps: int = 20,
        update_prob: float = 0.5,
        masking_th: float = 0.1,
        record_history: bool = False,
    ):
        """Run the grid CA for a fixed number of steps."""
        if record_history:
            state = self._grid_state.detach().clone().unsqueeze(0)
            for _ in range(steps):
                self.step(update_prob=update_prob, masking_th=masking_th)
                state = torch.cat( (state,
                                    self._grid_state.detach().clone().unsqueeze(0) ), 0)
            return state

        # otherwise only return final state
        for _ in range(steps):
            self.step(update_prob=update_prob, masking_th=masking_th)
        return self._grid_state.clone()

    def deepcopy(self) -> Grid:
        """More performant alternative to the built in deepcopy."""
        copy = Grid(
            self._width,
            self._height,
            self._num_channels,
            seed=self._seed,
            device=self._device,
        )
        copy.set_state(self._grid_state.detach().clone())
        return copy

    @property
    def state(self) -> npt.NDArray:
        if self._grid_state.device == "cpu":
            return self._grid_state.numpy()
        return self._grid_state.detach().numpy()

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

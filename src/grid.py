from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional

from nn import NN

if TYPE_CHECKING:
    import numpy.typing as npt
    from torch.types import Device, Tensor

from tqdm import trange

class Grid:
    def __init__(
        self,
        poolsize: int,
        batch_size: int,
        num_channels: int,
        width: int,
        height: int,
        *,
        seed: int = 43,
        device: Device | None = None,
        weights: tuple[npt.NDArray, ...],
    ) -> None:
        self._width = width
        self._height = height
        self._num_channels = num_channels
        self._batch_size = batch_size

        self._device = device if device is not None else torch.device("cpu")

        # Initialize weights for NN
        if weights:
            self.set_weights_on_nn(weights)

        # Grid states: (B, C, H, W)
        self._grids = torch.zeros(
            (poolsize, num_channels, height, width), dtype=torch.float32, device=self._device
        )

        # Points for all grid states 
        self._points = torch.zeros(poolsize, dtype=torch.float32, device=self._device)

        # Grids in current batch
        self._batch_idxs: torch.Tensor | None = None
        self.batch = torch.zeros(
            (batch_size, num_channels, height, width), dtype=torch.float32, device=self._device
        )
        self._batch_points = torch.zeros(batch_size, dtype=torch.float32, device=self._device)


        self._seed = seed
        self._rng = torch.Generator(device=self._device)
        self._rng.manual_seed(seed)

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

    def seed_center(self, grid_idx: int, seed_vector: Tensor) -> None:
        """Initialize seed state vector in center of grid."""
        self.set_cell_state(
            grid_idx=grid_idx,
            x=(self._width // 2),
            y=(self._height // 2),
            state_vector=seed_vector,
        )

    def set_batch(self, grid_idxs) -> None:
        idxs = torch.as_tensor(grid_idxs, dtype=torch.long, device="cpu")

        if idxs.numel() != self._batch_size:
            raise ValueError(
                f"Length of indices {idxs.numel()} does not match batch size {self._batch_size}."
            )

        self._batch_idxs = idxs

    def load_batch_from_pool(self) -> None:
        if self._batch_idxs is None:
            raise ValueError("Batch indices not set. Call set_batch(...) first.")
        self.batch = self._grids[self._batch_idxs.to(self._device)].clone()
        self._batch_points = self._points[self._batch_idxs.to(self._device)].clone()

    def write_batch_back_to_pool(self) -> None:
        """Optionally persist the evolved batch back into the pool."""
        if self._batch_idxs is None:
            raise ValueError("Batch indices not set.")
        self._grids[self._batch_idxs.to(self._device)] = self.batch
        self._points[self._batch_idxs.to(self._device)] = self._batch_points


    def reset_state(self, grid_idx) -> None:
        idxs = torch.as_tensor(grid_idx, dtype=torch.long, device=self._device)
        self._grids[idxs] = 0
        self._points[idxs] = 0


    def set_cell_state(self, grid_idx:int, x: int, y: int, state_vector: torch.Tensor) -> None:
        """Alter state vector at (y,x) in the grid."""
        state_vector = state_vector.to(
            device=self._device, dtype=self._grids.dtype
        )
        self._grids[grid_idx, :,  y, x] = state_vector

    def step(self, batch: Tensor, update_prob: float = 0.5, masking_th: float = 0.1):
        """Perform a single step of the grid's CA."""
        if update_prob > 1.0 or update_prob < 0.0:
            msg = f"update_prob must be in [0, 1], got {update_prob}"
            raise ValueError(msg)
        if masking_th < 0.0:
            msg = f"masking_th must be >= 0, got {masking_th}"
            raise ValueError(msg)

        ### Neural network
        # Get derivative of current grid state
        state_change = self.NN.forward(batch)

        ### Stochastic Update
        # Stochastic mask with a probability for each cell
        rand_mask = (
            torch.rand(
                (self.batch_size, self._height, self._width), generator=self._rng, device=self._device
            )
            < update_prob
        ).to(self.batch.dtype)

        # Expand dimensionality to alter all channels (B,H,W) -> (B, C, H, W)
        rand_mask = rand_mask.unsqueeze(1)

        # Update grid stochastically
        batch = batch + state_change * rand_mask

        ### Alive Masking
        # Allows wrapped alive masking with by multithreading on GPU
        alpha = batch[:, 1:2, :, :]  # (B, 1, H, W)
        alpha = functional.pad(alpha, (1, 1, 1, 1), mode="circular")
        alive = functional.max_pool2d(alpha, 3, stride=1, padding=0) > masking_th
        batch *= alive.float()
        return batch

    def run_simulation_batch(
        self,
        steps: int = 20,
        update_prob: float = 0.5,
        masking_th: float = 0.1,
        activate_print: bool = False,
    ):
        """Run the grid CA for a fixed number of steps."""
        iterator = trange(steps) if activate_print else range(steps)
        self.load_batch_from_pool()
        batch = self.batch
        for _ in iterator:
            batch = self.step(batch=batch, update_prob=update_prob, masking_th=masking_th)
        self.batch = batch

    # GPU
    @property
    def batch_state(self) -> torch.Tensor:
        return self.batch
    
    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def batch_points(self) -> npt.NDArray:
        return self._batch_points.detach().cpu().numpy()

    @property
    def shape(self) -> tuple[int]:
        return (self._height, self._width)

    # CPU
    @property
    def weights(self) -> tuple[npt.NDArray, ...] | None:
        return self._weights

    @property
    def points(self) -> npt.NDArray:
        return self._points.detach().cpu().numpy()




def main():
    pass


if __name__ == "__main__":
    main()

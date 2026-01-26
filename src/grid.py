from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional
from tqdm import trange

from nn import NN

if TYPE_CHECKING:
    import numpy.typing as npt
    from torch.types import Device, Tensor


class Grid:
    """Grid pool + batched Neural Cellular Automata (NCA) simulator.

    This class stores a pool of grid states (population) and evolves a selected
    batch of them efficiently on GPU.

    Typical usage:
        1) grid = Grid(...)
        2) grid.set_batch([...pool indices...])
        3) grid.load_batch_from_pool()
        4) grid.run_simulation_batch(...)
        5) (optional) update grid._batch_points via your fitness function
        6) grid.write_batch_back_to_pool()
    """

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
        weights: tuple[npt.NDArray, ...] | None = None,
    ) -> None:
        """Create a pool of grids and a reusable batch buffer.

        Args:
            poolsize: Number of grids in the population/pool.
            batch_size: Number of grids evolved at once.
            num_channels: Number of channels per cell (C).
            width: Grid width (W).
            height: Grid height (H).
            seed: Random seed for stochastic updates.
            device: Torch device, e.g. cpu/cuda.
            weights: Optional NN weights tuple (hidden_layer, output_layer).
        """
        self._width = width
        self._height = height
        self._num_channels = num_channels
        self._batch_size = batch_size

        self._device = device if device is not None else torch.device("cpu")

        # Initialize weights for NN
        if weights is not None:
            self.set_weights_on_nn(weights)

        # Grid states: (B, C, H, W)
        self._grids = torch.zeros(
            (poolsize, num_channels, height, width), dtype=torch.float32, device=self._device
        )

        # Points for all grid states
        self._points = torch.zeros(poolsize, dtype=torch.float32, device=self._device)

        # Grids in current batch
        self._batch_idxs: torch.Tensor | None = None
        self._batch = torch.zeros(
            (batch_size, num_channels, height, width), dtype=torch.float32, device=self._device
        )
        self._batch_points = torch.zeros(batch_size, dtype=torch.float32, device=self._device)

        self._seed = seed
        self._rng = torch.Generator(device=self._device)
        self._rng.manual_seed(seed)

    def set_weights_on_nn(self, weights: tuple[npt.NDArray, ...]) -> None:
        """Load pre-trained weights into the internal neural network.

        Args:
            weights: Tuple of two numpy arrays (hidden_layer, output_layer).

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
        hidden_tens = torch.from_numpy(hidden_layer).to(self._device, dtype=torch.float32)
        hidden_tens = hidden_tens.permute(1, 0).unsqueeze(-1).unsqueeze(-1)

        output_tens = torch.from_numpy(output_layer).to(self._device, dtype=torch.float32)
        output_tens = output_tens.permute(1, 0).unsqueeze(-1).unsqueeze(-1)

        # create NN instance and load weights
        self.NN = NN(self._num_channels, hidden_layer.shape[1]).to(self._device)
        self.NN.load_weights(hidden_tens, output_tens)

        # set weights as attribute
        self._weights = weights

    def seed_center(self, grid_idx: int, seed_vector: Tensor) -> None:
        """Seed one pool grid at the center cell."""
        self.set_cell_state(
            grid_idx=grid_idx,
            x=(self._width // 2),
            y=(self._height // 2),
            state_vector=seed_vector,
        )

    def set_batch(self, grid_idxs) -> None:
        """Set which pool indices will be loaded into the batch.

        Args:
            grid_idxs: Iterable of length batch_size with pool indices.

        Raises:
            ValueError: If number of indices != batch_size.
        """
        idxs = torch.as_tensor(grid_idxs, dtype=torch.long, device="cpu")

        if idxs.numel() != self._batch_size:
            raise ValueError(
                f"Length of indices {idxs.numel()} does not match batch size {self._batch_size}."
            )

        self._batch_idxs = idxs

    def load_batch_from_pool(self) -> None:
        """Copy selected pool grids/points into the batch buffers."""
        if self._batch_idxs is None:
            raise ValueError("Batch indices not set. Call set_batch(...) first.")
        idxs = self._batch_idxs.to(self._device)
        self._batch = self._grids[idxs].clone()
        self._batch_points = self._points[idxs].clone()

    def write_batch_back_to_pool(self) -> None:
        """Write batch buffers back into the pool (persist changes)."""
        if self._batch_idxs is None:
            raise ValueError("Batch indices not set.")
        idxs = self._batch_idxs.to(self._device)
        self._grids[idxs] = self._batch
        self._points[idxs] = self._batch_points

    def reset_state(self, grid_idx) -> None:
        """Reset one or more pool grids (and their points) to zero."""
        idxs = torch.as_tensor(grid_idx, dtype=torch.long, device=self._device)
        self._grids[idxs] = 0
        self._points[idxs] = 0

    def set_cell_state(self, grid_idx: int, x: int, y: int, state_vector: torch.Tensor) -> None:
        """Set a single cell state in a pool grid."""
        state_vector = state_vector.to(device=self._device, dtype=self._grids.dtype)
        self._grids[grid_idx, :, y, x] = state_vector

    def step(self, batch: Tensor, update_prob: float = 0.5, masking_th: float = 0.1) -> Tensor:
        """Run one CA update step on a batched tensor (B, C, H, W)."""
        if not hasattr(self, "NN"):
            raise RuntimeError("NN not initialized. Call set_weights_on_nn(...) first.")
        if not (0.0 <= update_prob <= 1.0):
            raise ValueError(f"update_prob must be in [0, 1], got {update_prob}")
        if masking_th < 0.0:
            raise ValueError(f"masking_th must be >= 0, got {masking_th}")

        ### Neural network
        # Get derivative of current grid state
        state_change = self.NN.forward(batch)

        ### Stochastic Update
        # Stochastic mask with a probability for each cell
        rand_mask = (
            torch.rand(
                (self._batch_size, self._height, self._width),
                generator=self._rng,
                device=self._device,
            )
            < update_prob
        ).to(batch.dtype)

        # Expand dimensionality to alter all channels (B,H,W) -> (B, C, H, W)
        rand_mask = rand_mask.unsqueeze(1)

        # Update grid stochastically
        batch = batch + state_change * rand_mask

        ### Alive Masking
        # Allows wrapped alive masking with by multithreading on GPU
        alpha = batch[:, 0:1, :, :]  # (B, 1, H, W)
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
    ) -> None:
        """Evolve the currently loaded batch for a fixed number of steps.

        Note: This evolves `self._batch` only. It does NOT automatically
        load from the pool or write back to the pool.
        """
        iterator = trange(steps) if activate_print else range(steps)
        batch = self._batch
        for _ in iterator:
            batch = self.step(batch=batch, update_prob=update_prob, masking_th=masking_th)
        self._batch = batch

    # GPU
    @property
    def batch_state(self) -> torch.Tensor:
        """Current batch tensor (B, C, H, W) on device."""
        return self._batch

    @property
    def batch_size(self) -> int:
        """Batch size (B)."""
        return self._batch_size

    @property
    def batch_points(self) -> npt.NDArray:
        """Batch points as NumPy (copies to CPU)."""
        return self._batch_points.detach().cpu().numpy()

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial grid shape (H, W)."""
        return (self._height, self._width)

    # CPU
    @property
    def weights(self) -> tuple[npt.NDArray, ...] | None:
        """Current NN weights tuple, or None if not set."""
        return getattr(self, "_weights", None)

    @property
    def points(self) -> npt.NDArray:
        """Pool points as NumPy (copies to CPU)."""
        return self._points.detach().cpu().numpy()


def main() -> None:
    pass


if __name__ == "__main__":
    main()

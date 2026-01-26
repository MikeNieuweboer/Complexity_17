from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn import functional
from tqdm import trange

import numpy as np

from nn import NN

if TYPE_CHECKING:
    import numpy.typing as npt
    from torch._tensor import Tensor
    from torch.types import Device


class Grid:
    """Grid pool + batched Neural Cellular Automata (NCA) simulator.

    This class stores a pool of grid states (population) and evolves a selected
    batch of them efficiently on GPU.

    Typical usage:
        1) grid = Grid(...)
        2) grid.set_batch([...pool indices...])
        3) grid.load_batch_from_pool()
        4) grid.run_simulation_batch(...)
        5) grid.write_batch_back_to_pool()
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
        else:
            self.NN = None

        # Grid states: (pool, C, H, W)
        self._grids = torch.zeros(
            (poolsize, num_channels, height, width), dtype=torch.float32, device=self._device
        )

        # Grids in current batch
        self._batch_idxs: torch.Tensor | None = None
        self._batch = torch.zeros(
            (batch_size, num_channels, height, width), dtype=torch.float32, device=self._device
        )

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

    def set_weights_on_nn_from_tens(self, weights: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Load pre-trained weights in Conv2d tensor format.

        Expected shapes:
            hidden: (hidden, 3*C, 1, 1)
            output: (C, hidden, 1, 1)

        Args:
            weights: Tuple (hidden_tens, output_tens) in Conv2d weight format.

        Raises:
            ValueError: If weights tuple doesn't have exactly 2 elements.
        """
        if len(weights) != 2:  # noqa: PLR2004
            raise ValueError(  # noqa: TRY003
                "weights should have dimension 2 (hidden_layer, output_layer)",  # noqa: EM101
                f", got {len(weights)}",
            )

        hidden_tens, output_tens = weights
        hidden_tens = hidden_tens.to(self._device, dtype=torch.float32)
        output_tens = output_tens.to(self._device, dtype=torch.float32)

        self.NN = NN(self._num_channels, hidden_tens.shape[0]).to(self._device)
        self.NN.load_weights(hidden_tens, output_tens)

        self._weights = (
            hidden_tens.detach().cpu().numpy(),
            output_tens.detach().cpu().numpy(),
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
        """Copy selected pool grids into the batch buffer."""
        if self._batch_idxs is None:
            raise ValueError("Batch indices not set. Call set_batch(...) first.")
        idxs = self._batch_idxs.to(self._device)
        self._batch = self._grids[idxs].clone()

    def write_batch_back_to_pool(self) -> None:
        """Write batch buffer back into the pool (persist changes)."""
        if self._batch_idxs is None:
            raise ValueError("Batch indices not set.")
        idxs = self._batch_idxs.to(self._device)
        self._grids[idxs] = self._batch

    def sample_batch(self) -> torch.Tensor:
        """Randomly choose pool indices, load them into the batch, and return the indices."""
        poolsize = self._grids.shape[0]
        idxs = torch.randperm(poolsize, device="cpu")[: self._batch_size]
        self.set_batch(idxs)
        self.load_batch_from_pool()
        return idxs

    def reset_state(self, grid_idx) -> None:
        """Reset one or more pool grids to zero."""
        idxs = torch.as_tensor(grid_idx, dtype=torch.long, device=self._device)
        self._grids[idxs] = 0

    def set_cell_state(self, grid_idx: int, x: int, y: int, state_vector: torch.Tensor) -> None:
        """Set a single cell state in a pool grid."""
        state_vector = state_vector.to(device=self._device, dtype=self._grids.dtype)
        self._grids[grid_idx, :, y, x] = state_vector

    def seed_center(self, grid_idx: int, seed_vector: Tensor) -> None:
        """Seed one pool grid at the center cell."""
        self.set_cell_state(
            grid_idx=grid_idx,
            x=(self._width // 2),
            y=(self._height // 2),
            state_vector=seed_vector,
        )

    def clear_and_seed(
        self,
        *,
        grid_idx: int | torch.Tensor,
        seed_vector: Tensor | None = None,
        in_batch: bool = False,
    ) -> None:
        """Clear state(s) and seed a center cell.

        Assumes batched tensors are always 4D (B, C, H, W).

        Args:
            grid_idx: Pool indices (in_batch=False) or batch positions (in_batch=True).
            seed_vector: Optional seed vector of shape (C,). If None, uses all ones.
            in_batch: Whether to operate on the currently loaded batch buffer.
        """
        if seed_vector is None:
            seed_vector = torch.ones(self._num_channels, dtype=torch.float32, device=self._device)
        seed_vector = seed_vector.to(device=self._device, dtype=torch.float32)

        idxs = torch.as_tensor(grid_idx, dtype=torch.long, device=self._device)
        cx, cy = self._width // 2, self._height // 2

        if in_batch:
            self._batch[idxs] = 0
            self._batch[idxs, :, cy, cx] = seed_vector
            return

        self._grids[idxs] = 0
        self._grids[idxs, :, cy, cx] = seed_vector

    def set_state(self, new_state: Tensor) -> None:
        """Replace the entire current batch state.

        Args:
            new_state: 4D tensor (B, C, H, W) matching the configured batch shape.

        Raises:
            ValueError: If shape does not match the batch buffer.
        """
        new_state = new_state.to(device=self._device, dtype=self._batch.dtype)
        if new_state.dim() != 4:
            raise ValueError(f"new_state must be 4D (B,C,H,W), got dim={new_state.dim()}")
        if tuple(new_state.shape) != tuple(self._batch.shape):
            raise ValueError(f"new_state must have shape {tuple(self._batch.shape)}, got {tuple(new_state.shape)}")
        self._batch = new_state

    def step(self, batch: Tensor, update_prob: float = 0.5, masking_th: float = 0.1) -> Tensor:
        """Run one CA update step on a batched tensor (B, C, H, W)."""
        if not hasattr(self, "NN"):
            raise RuntimeError("NN not initialized. Call set_weights_on_nn(...) first.")
        if batch.dim() != 4:
            raise ValueError(f"batch must be 4D (B,C,H,W), got dim={batch.dim()}")
        if not (0.0 <= update_prob <= 1.0):
            raise ValueError(f"update_prob must be in [0, 1], got {update_prob}")
        if masking_th < 0.0:
            raise ValueError(f"masking_th must be >= 0, got {masking_th}")
        if batch.shape[1] != self._num_channels or batch.shape[2] != self._height or batch.shape[3] != self._width:
            raise ValueError(
                f"batch must have shape (B,{self._num_channels},{self._height},{self._width}), got {tuple(batch.shape)}"
            )

        ### Neural network
        state_change = self.NN.forward(batch)

        ### Stochastic Update
        rand_mask = (
            torch.rand(
                (batch.shape[0], self._height, self._width),
                generator=self._rng,
                device=self._device,
            )
            < update_prob
        ).to(batch.dtype)
        rand_mask = rand_mask.unsqueeze(1)  # (B,1,H,W)
        batch = batch + state_change * rand_mask

        ### Alive Masking
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
        """Evolve the currently loaded batch for a fixed number of steps."""
        iterator = trange(steps) if activate_print else range(steps)
        batch = self._batch
        for _ in iterator:
            batch = self.step(batch=batch, update_prob=update_prob, masking_th=masking_th)
        self._batch = batch

    def step_test(self):
        # testing a simple CA update rule
        new_grid = np.copy(self._grid_state[:, :, 0])
        for i in range(self._grid_state.shape[0]):
            for j in range(self._grid_state.shape[1]):
                if any(new_grid[i - 1 : i + 2, j]) or any(new_grid[i, j - 1 : j + 2]):
                    new_grid[i, j] = 1
        self._grid_state[:, :, 0] = torch.tensor(new_grid)

    def step_test_speed(self, grid) -> np.ndarray:
        new_grid = np.copy(self._grid_state[:, :, 0])
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if any(self._grid_state[i - 1 : i + 2, j, 0]) or any(
                    self._grid_state[i, j - 1 : j + 2, 0]
                ):
                    new_grid[i, j] = 1
        new_grid = torch.tensor(new_grid)
        new_grid = new_grid.numpy()
        return new_grid

    def step_test(self):
        # testing a simple CA update rule
        grid_copy = np.copy(self._grid_state[:, :, 0])
        for i in range(grid_copy.shape[0]):
            for j in range(grid_copy.shape[1]):
                if (
                    grid_copy[i - 1 : i + 2, j].any()
                    or grid_copy[i, j - 1 : j + 2].any()
                ):
                    self._grid_state[i, j] = 1
        # self._grid_state[:,:,0] = torch.tensor(new_grid)

    def step_test_speed(self, grid) -> np.ndarray:
        new_grid = np.copy(self._grid_state[:, :, 0])
        for i in range(new_grid.shape[0]):
            for j in range(new_grid.shape[1]):
                if any(new_grid[i - 1 : i + 2, j]) or any(new_grid[i, j - 1 : j + 2]):
                    new_grid[i, j] = 1
        new_grid = torch.tensor(new_grid)
        new_grid = new_grid.numpy()
        return new_grid

    def run_simulation(
        self,
        steps: int = 20,
        update_prob: float = 0.5,
        masking_th: float = 0.1,
        record_history: bool = False,
    ) -> torch.Tensor:
        """Run the batched CA and return results.

        Assumes batch is always 4D (B, C, H, W).

        Returns:
            If record_history=True: (T, B, C, H, W) where T = steps + 1
            Else: (B, C, H, W) final batch state
        """
        if record_history:
            hist = [self._batch.detach().clone()]
            batch = self._batch
            for _ in range(steps):
                batch = self.step(batch=batch, update_prob=update_prob, masking_th=masking_th)
                hist.append(batch.detach().clone())
            self._batch = batch
            return torch.stack(hist, dim=0)

        self.run_simulation_batch(
            steps=steps,
            update_prob=update_prob,
            masking_th=masking_th,
            activate_print=False,
        )
        return self._batch.detach().clone()

    def state(self, layer=None) -> npt.NDArray:
        if layer == None:
            if self._grid_state.device == "cpu":
                return self._grid_state.numpy()
            return self._grid_state.detach().numpy()
        else:
            if self._grid_state.device == "cpu":
                return self._grid_state[:, :, layer].numpy()
            return self._grid_state[:, :, layer].detach().numpy()

    @property
    def pool_state(self) -> npt.NDArray:
        """Numpy view of the entire pool state. Shape: (pool, C, H, W)."""
        return self._grids.detach().cpu().numpy()

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
    def shape(self) -> tuple[int, int]:
        """Spatial grid shape (H, W)."""
        return (self._height, self._width)

    # CPU
    @property
    def weights(self) -> tuple[npt.NDArray, ...] | None:
        """Current NN weights tuple, or None if not set."""
        return getattr(self, "_weights", None)

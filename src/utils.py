from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch


def load_weights(path: Path) -> tuple[npt.NDArray, npt.NDArray]:
    """Load weights from either an .npz or .pt file.

    Args:
    ----
        path: Path to the weights file (.npz or .pt)

    Returns:
    -------
        Tuple of (hidden_layer_weights, output_layer_weights) as numpy arrays

    Raises:
    ------
        ValueError: If file format is not supported or weights structure is invalid

    """
    weight_count = 2
    if path.suffix == ".npz":
        # Load from NPZ file
        data = np.load(path)
        # NPZ files typically store multiple arrays, get them in order
        arrays = [data[key] for key in sorted(data.files)]
        if len(arrays) != weight_count:
            msg = f"NPZ file must contain 2 weight arrays, got {len(arrays)}"
            raise ValueError(msg)
        return tuple(arrays[:weight_count])
    if path.suffix == ".pt":
        # Load from PyTorch file
        state_dict = torch.load(path, weights_only=True)
        # Convert torch tensors to numpy arrays
        weights = [
            np.reshape(tensor.numpy(), (tensor.shape[0], tensor.shape[1])).T
            for tensor in state_dict
        ]

        if len(weights) != weight_count:
            msg = f"PT file must contain 2 weight tensors, got {len(weights)}"
            raise ValueError(msg)
        return (weights[0], weights[1])
    msg = f"Unsupported file format: {path.suffix}. Expected .npz or .pt"
    raise ValueError(msg)

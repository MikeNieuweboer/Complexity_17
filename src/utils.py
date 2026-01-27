from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image


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

def load_target_image(image_path: Path, grid_size: int) -> torch.Tensor:
    """Load an image, to utilize as a target for training.

    Process:
    1. Opens image and converts to Grayscale ('L').
    2. Resizes to (grid_size, grid_size).
    3. Normalizes pixel values to [0.0, 1.0].

    Returns:
        torch.Tensor: A tensor of shape (H, W) on the specified device.

    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    # load and process
    img = Image.open(image_path).convert("L")
    img = img.resize((grid_size, grid_size), Image.Resampling.LANCZOS)

    # convert to numpy -> tensor -> normalize
    img_data = np.array(img, dtype=np.float32) / 255.0

    # we need black=1.0, and white=0
    target_data = 1.0 - img_data

    # force close to 0 and 1 to be exactly that
    target_data[target_data > 0.9] = 1
    target_data[target_data < 0.1] = 0

    # Optional: Hard threshold to make it strictly black/white (binary)
    return torch.from_numpy(target_data)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data"
    image_path = data_dir / "targets" / "star.png"

    tens = load_target_image(image_path, 50)

    plt.imshow(tens)
    plt.show()
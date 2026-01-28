"""Two functions to plot/animate heatmaps of gridstates."""

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from grid import Grid
from nn import NN


def plot_heatmaps(tens: torch.Tensor,
                  save_path: Path,
                  comptitle: str | None = None,
                  suptitle: str | None = None,
                  ) -> None:
    """Plot heatmaps of the layers in a 3D Tensor.

    Args:
        tens:       A 3D tensor (H (height), W (width), * (misc)).
                    - USAGE NOTE: misc can for example be state vector, but also
                                  layer of state vector across batch.
        save_path:  Path object where figure will be saved.
        comptitle:  A string describing the layers. Will be used as subplot title
                    following: "comptitle [i]" (i is index of axis)
        suptitle:   Title for the grid of heatmaps

    """
    if isinstance(tens, torch.Tensor):
        tens = tens.detach().cpu().numpy()

    num_maps = tens.shape[2]

    side_len = ceil(np.sqrt(num_maps))
    fig, axes = plt.subplots(ceil(num_maps/side_len), side_len,
                             sharex=True, sharey=True, constrained_layout=True)

    for i, ax in enumerate(np.asarray(axes).flatten()):
        if i >= num_maps:
            ax.set_visible(False)
            continue

        layer = tens[:, :, i]
        im = ax.imshow(layer, cmap="inferno", vmin=0, vmax=1)

        if comptitle:
            ax.set_title(f"{comptitle} [{i}]")

    fig.colorbar(im, ax=axes.ravel().tolist())
    if suptitle:
        fig.suptitle(suptitle)

    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def animate_heatmaps(tens: torch.Tensor,
                     save_path: Path,
                     comptitle: str | None = None,
                     suptitle: str | None = None,
                     interval: int = 50,
                     ) -> None:
    """Animate a sequence of grid states as heatmaps.

    Args:
        tens:       A 4D tensor (T (frames), H (height), W (width), * (misc)).
                    - USAGE NOTE: misc can for example be state vector, but also
                                  layer of state vector across batch.
        save_path:  Path object where the animation will be saved.
        comptitle:  A string describing the layers. Will be used as subplot title
                    following: "comptitle [i]" (i is index of axis).
        suptitle:   Title for the grid of heatmaps.
        interval:   Time between frames in milliseconds.

    """
    if tens.dim() != 4:
        raise ValueError("tens must be 4D (T, H, W, C)")  # noqa: EM101, TRY003

    n_frames, _, _, num_maps = tens.shape
    side_len = ceil(np.sqrt(num_maps))
    fig, axes = plt.subplots(ceil(num_maps/side_len), side_len,
                             sharex=True, sharey=True, constrained_layout=True)

    images = []
    for i, ax in enumerate(np.asarray(axes).flatten()):
        if i >= num_maps:
            ax.set_visible(False)
            continue

        layer = tens[0, :, :, i]
        im = ax.imshow(layer, cmap="inferno", vmin=0, vmax=1, animated=True)

        if comptitle:
            ax.set_title(f"{comptitle} [{i}]")

        images.append(im)

    def update(frame_idx: int) -> list:
        for i, im in enumerate(images):
            im.set_data(tens[frame_idx, :, :, i])
        return images

    fig.colorbar(images[0], ax=axes.ravel().tolist())
    if suptitle:
        fig.suptitle(suptitle)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)
    anim.save(save_path, fps=1000//interval)
    plt.close(fig)

if __name__ == "__main__":
    # TESTING PLOTS
    # setting up directories
    root_dir = Path(__file__).parent.parent
    weights_dir = root_dir / "weights"
    best_weights_path = weights_dir / "Gr50-Ch8-Hi64_20260127-230604.pt"

    temp_dir = root_dir / "temp_figs"
    temp_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

    # setup grid for testing
    grid = Grid(
        poolsize=1,
        batch_size=1,
        num_channels=8,
        width=50,
        height=50,
        device=device,
    )
    grid.NN = NN(8, 64).to(device)
    weights = torch.load(best_weights_path, map_location=device)
    grid.set_weights_on_nn_from_tens(weights)

    # load batch, reseed and run
    grid.set_batch(list(range(1)))
    grid.load_batch_from_pool()
    grid.clear_and_seed(grid_idx=torch.arange(1, device=device), in_batch=True)
    states = grid.run_simulation(150, record_history=True)

    # testing plots
    plot_heatmaps(states.detach().squeeze(1)[150, :, :, :].permute(1, 2, 0),
                  temp_dir / "temp_fig.png",
                  None,
                  "Final state Heatmap",
                  )
    animate_heatmaps(states.detach().squeeze(1).permute(0, 2, 3, 1),
                     temp_dir / "temp_anim.mp4",
                     "Channel",
                     "Training Animation")

"""Two functions to plot or animate heatmaps.

Group:      17
Course:     Complex System Simulation

Description
-----------
Contains two functions to plot or animate heatmaps.
    plot_heatmaps requires a 3D tensor (H (height), W (width), * (misc)), and creates 
        a figure containing heatmaps for all grids (H, W) in the misc dimension.
    animate_heatmaps requires a 4D tensor (T (frames), H (height), W (width), * (misc))
        that creates a similar figure as plot_heatmaps, but now animates it depending
        on the T dimension.

AI usage
--------
Gemini 3.0 was used to generate docstrings for the functionality in this file.
This was done for each function individually using the inline chat and the
following prompt:
> Analyze the specific function and create a consise docstring
Afterwards manual changes were made to the liking of the author.

"""

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation


def plot_heatmaps(tens: torch.Tensor,
                  save_path: Path,
                  comptitle: str | None = None,
                  suptitle: str | None = None,
                  ) -> None:
    """Plot heatmaps of the layers in a 3D Tensor.

    Args:
        tens: A 3D tensor (H (height), W (width), * (misc)).
            - USAGE NOTE: misc can for example be state vector, but also
            layer of state vector across batch.
        save_path: Path object where figure will be saved.
        comptitle: A string describing the layers. Will be used as subplot title
            following: "comptitle [i]" (i is index of axis)
        suptitle: Title for the grid of heatmaps

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
        tens: A 4D tensor (T (frames), H (height), W (width), * (misc)).
            - USAGE NOTE: misc can for example be state vector, but also
            layer of state vector across batch.
        save_path: Path object where the animation will be saved.
        comptitle: A string describing the layers. Will be used as subplot title
            following: "comptitle [i]" (i is index of axis).
        suptitle: Title for the grid of heatmaps.
        interval: Time between frames in milliseconds.

    """
    if tens.dim() != 4:
        raise ValueError("tens must be 4D (T, H, W, *)")  # noqa: EM101, TRY003

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
        """Update function that is required by FuncAnimation."""
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
    pass

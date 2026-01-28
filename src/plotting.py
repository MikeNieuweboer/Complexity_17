from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from grid import Grid
from nn import NN

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# ---- Live plotting globals (module-level) ----
_LIVE_ENABLED: bool = False
_LIVE_FIG = None
_LIVE_AXES_FLAT = None
_LIVE_IMAGES = None
_LIVE_NUM_MAPS: int | None = None
_LIVE_CBAR = None
_LIVE_PAUSE_S: float = 0.001
_LIVE_VMIN: float = 0.0
_LIVE_VMAX: float = 1.0
_LIVE_CMAP: str = "inferno"
_LIVE_COMPTITLE: str | None = None
_LIVE_SUPTITLE: str | None = None
_LIVE_THROTTLE_EVERY: int = 1
_LIVE_STEP_COUNTER: int = 0

# ---- Live loss globals ----
_LIVE_LOSS_ENABLED: bool = False
_LIVE_LOSS_FIG = None
_LIVE_LOSS_AX = None
_LIVE_LOSS_LINE = None
_LIVE_LOSS_X: list[int] = []
_LIVE_LOSS_Y: list[float] = []
_LIVE_LOSS_MAX_POINTS: int | None = None  # None keeps all
_LIVE_LOSS_TITLE: str = "Loss (live)"


def enable_live_heatmaps(
    *,
    pause_s: float = 0.001,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "inferno",
    comptitle: str | None = None,
    suptitle: str | None = None,
    throttle_every: int = 1,
) -> None:
    """Enable a persistent live heatmap window updated by plot_heatmaps(save_path=None)."""
    global _LIVE_ENABLED, _LIVE_PAUSE_S, _LIVE_VMIN, _LIVE_VMAX, _LIVE_CMAP
    global _LIVE_COMPTITLE, _LIVE_SUPTITLE, _LIVE_THROTTLE_EVERY, _LIVE_STEP_COUNTER

    _LIVE_ENABLED = True
    _LIVE_PAUSE_S = pause_s
    _LIVE_VMIN = vmin
    _LIVE_VMAX = vmax
    _LIVE_CMAP = cmap
    _LIVE_COMPTITLE = comptitle
    _LIVE_SUPTITLE = suptitle
    _LIVE_THROTTLE_EVERY = max(int(throttle_every), 1)
    _LIVE_STEP_COUNTER = 0

    plt.ion()


def close_live_heatmaps() -> None:
    """Close the persistent live heatmap window if it exists."""
    global _LIVE_ENABLED, _LIVE_FIG, _LIVE_AXES_FLAT, _LIVE_IMAGES, _LIVE_NUM_MAPS, _LIVE_CBAR

    _LIVE_ENABLED = False

    if _LIVE_FIG is not None:
        plt.close(_LIVE_FIG)
        close_live_loss()

    _LIVE_FIG = None
    _LIVE_AXES_FLAT = None
    _LIVE_IMAGES = None
    _LIVE_NUM_MAPS = None
    _LIVE_CBAR = None

def enable_live_loss(*, max_points: int | None = 2000, title: str = "Loss (live)") -> None:
    """Enable a persistent live loss figure updated by plot_heatmaps(..., loss=...)."""
    global _LIVE_LOSS_ENABLED, _LIVE_LOSS_MAX_POINTS, _LIVE_LOSS_TITLE
    global _LIVE_LOSS_FIG, _LIVE_LOSS_AX, _LIVE_LOSS_LINE, _LIVE_LOSS_X, _LIVE_LOSS_Y

    _LIVE_LOSS_ENABLED = True
    _LIVE_LOSS_MAX_POINTS = max_points
    _LIVE_LOSS_TITLE = title

    _LIVE_LOSS_X = []
    _LIVE_LOSS_Y = []

    plt.ion()

    # Create the loss window
    _LIVE_LOSS_FIG, _LIVE_LOSS_AX = plt.subplots(constrained_layout=True)
    (_LIVE_LOSS_LINE,) = _LIVE_LOSS_AX.plot([], [])  # default style
    _LIVE_LOSS_AX.set_title(_LIVE_LOSS_TITLE)
    _LIVE_LOSS_AX.set_xlabel("Step")
    _LIVE_LOSS_AX.set_ylabel("Loss")
    _LIVE_LOSS_FIG.show()


def close_live_loss() -> None:
    """Close the persistent live loss window if it exists."""
    global _LIVE_LOSS_ENABLED, _LIVE_LOSS_FIG, _LIVE_LOSS_AX, _LIVE_LOSS_LINE
    global _LIVE_LOSS_X, _LIVE_LOSS_Y

    _LIVE_LOSS_ENABLED = False

    if _LIVE_LOSS_FIG is not None:
        plt.close(_LIVE_LOSS_FIG)

    _LIVE_LOSS_FIG = None
    _LIVE_LOSS_AX = None
    _LIVE_LOSS_LINE = None
    _LIVE_LOSS_X = []
    _LIVE_LOSS_Y = []

def _update_live_loss(step_idx: int, loss: float) -> None:
    """Append a loss point and redraw the loss figure."""
    if not _LIVE_LOSS_ENABLED or _LIVE_LOSS_FIG is None:
        return

    _LIVE_LOSS_X.append(int(step_idx))
    _LIVE_LOSS_Y.append(float(loss))

    if _LIVE_LOSS_MAX_POINTS is not None and len(_LIVE_LOSS_X) > _LIVE_LOSS_MAX_POINTS:
        # Keep the most recent points only
        _LIVE_LOSS_X[:] = _LIVE_LOSS_X[-_LIVE_LOSS_MAX_POINTS :]
        _LIVE_LOSS_Y[:] = _LIVE_LOSS_Y[-_LIVE_LOSS_MAX_POINTS :]

    _LIVE_LOSS_LINE.set_data(_LIVE_LOSS_X, _LIVE_LOSS_Y)
    _LIVE_LOSS_AX.relim()
    _LIVE_LOSS_AX.autoscale_view()

    _LIVE_LOSS_FIG.canvas.draw_idle()
    _LIVE_LOSS_FIG.canvas.flush_events()
    plt.pause(_LIVE_PAUSE_S)



def _ensure_live_window(num_maps: int) -> None:
    """Create the live window once, or recreate if num_maps changes."""
    global _LIVE_FIG, _LIVE_AXES_FLAT, _LIVE_IMAGES, _LIVE_NUM_MAPS, _LIVE_CBAR

    if _LIVE_FIG is not None and _LIVE_NUM_MAPS == num_maps:
        return

    # Recreate if needed
    if _LIVE_FIG is not None:
        plt.close(_LIVE_FIG)

    _LIVE_NUM_MAPS = num_maps
    side_len = ceil(np.sqrt(num_maps))
    nrows = ceil(num_maps / side_len)

    fig, axes = plt.subplots(nrows, side_len, sharex=True, sharey=True, constrained_layout=True)
    axes_flat = np.asarray(axes).flatten()

    images = []
    dummy = np.zeros((10, 10), dtype=np.float32)
    last_im = None

    for i, ax in enumerate(axes_flat):
        if i >= num_maps:
            ax.set_visible(False)
            continue

        last_im = ax.imshow(dummy, cmap=_LIVE_CMAP, vmin=_LIVE_VMIN, vmax=_LIVE_VMAX, animated=False)
        if _LIVE_COMPTITLE:
            ax.set_title(f"{_LIVE_COMPTITLE} [{i}]")
        images.append(last_im)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_flat.tolist())
    else:
        cbar = None

    if _LIVE_SUPTITLE:
        fig.suptitle(_LIVE_SUPTITLE)

    fig.show()

    _LIVE_FIG = fig
    _LIVE_AXES_FLAT = axes_flat
    _LIVE_IMAGES = images
    _LIVE_CBAR = cbar


def _update_live(frame_hwc: np.ndarray, suptitle: str | None) -> None:
    """Update live images with frame (H,W,num_maps)."""

    # Throttle using the global step counter that plot_heatmaps controls
    if (_LIVE_STEP_COUNTER % _LIVE_THROTTLE_EVERY) != 0:
        return

    num_maps = frame_hwc.shape[2]
    _ensure_live_window(num_maps)

    for im_i, im in enumerate(_LIVE_IMAGES):
        im.set_data(frame_hwc[:, :, im_i])

    if suptitle is not None and _LIVE_FIG is not None:
        _LIVE_FIG.suptitle(suptitle)

    _LIVE_FIG.canvas.draw_idle()
    _LIVE_FIG.canvas.flush_events()
    plt.pause(_LIVE_PAUSE_S)


def plot_heatmaps(
    tens: torch.Tensor,
    save_path: Path | None = None,
    comptitle: str | None = None,
    suptitle: str | None = None,
    loss: float | None = None,
) -> None:
    """Plot heatmaps of the layers in a 3D Tensor.

    Args:
        tens: A 3D tensor (H, W, misc). misc can be channels or batch index.
        save_path: If provided, saves a PNG to this path.
        comptitle: Used in subplot titles.
        suptitle: Title for the grid of heatmaps.
        loss: If provided and live is enabled, updates a live loss plot too.
    """
    global _LIVE_STEP_COUNTER
    step_idx = _LIVE_STEP_COUNTER

        # Convert to numpy once
    if isinstance(tens, torch.Tensor):
        tens_np = tens.detach().cpu().numpy()
    else:
        tens_np = np.asarray(tens)

    if tens_np.ndim != 3:
        raise ValueError(f"tens must be 3D (H,W,*) but got shape {tens_np.shape}")

    # Step index for the loss plot (tied to how often plot_heatmaps is called)
    _LIVE_STEP_COUNTER += 1
    step_idx = _LIVE_STEP_COUNTER

    # If live is enabled, update the persistent window whenever this is called
    if _LIVE_ENABLED:
        _update_live(tens_np, suptitle)

        if loss is not None:
            _update_live_loss(step_idx, float(loss))

        # If this call is only for live display, stop here (fast path)
        if save_path is None:
            return

    # If saving requested, do a normal one-off figure save WITHOUT opening a window
    if save_path is not None:
        num_maps = tens_np.shape[2]
        side_len = ceil(np.sqrt(num_maps))
        nrows = ceil(num_maps / side_len)

        fig = Figure(constrained_layout=True)
        FigureCanvas(fig)  # attach Agg canvas (no GUI window)

        axes = fig.subplots(nrows, side_len, sharex=True, sharey=True)
        axes_flat = np.asarray(axes).flatten()

        last_im = None
        for i, ax in enumerate(axes_flat):
            if i >= num_maps:
                ax.set_visible(False)
                continue

            layer = tens_np[:, :, i]
            last_im = ax.imshow(layer, cmap="inferno", vmin=0, vmax=1)

            if comptitle:
                ax.set_title(f"{comptitle} [{i}]")

        if last_im is not None:
            fig.colorbar(last_im, ax=axes_flat.tolist())

        if suptitle:
            fig.suptitle(suptitle)

        fig.savefig(save_path, dpi=300)
        return


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

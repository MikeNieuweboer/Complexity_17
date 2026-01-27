from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

# tqdm
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtWidgets

from grid import Grid

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.backend_bases import MouseEvent

gridsize = 100


class ToolBar(QtWidgets.QWidget):
    """The toolbar widget containing controlls for viewing and updating the grid."""

    # signals for transmitting button and slider data
    step_requested = QtCore.pyqtSignal(bool)
    update_toggle = QtCore.pyqtSignal(bool)
    reset_requested = QtCore.pyqtSignal(bool)
    sim_speed_requested = QtCore.pyqtSignal(int)
    erase_size_requested = QtCore.pyqtSignal(int)
    on_new_weights = QtCore.pyqtSignal(tuple)

    def __init__(
        self,
        parent: MainWindow,
        analysis_tool: dict[str, Callable[[npt.NDArray], str]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFixedSize(QtCore.QSize(150, 300))
        self._analysis_tool = analysis_tool if analysis_tool is not None else {}
        self._parent = parent
        self.grid = None

        # setting up buttons

        combobox_items = [
            "",
            *(list(analysis_tool.keys()) if analysis_tool is not None else []),
        ]
        self.analysis_tool = QtWidgets.QComboBox()
        self.analysis_tool.addItems(combobox_items)
        self.analysis_tool_label = QtWidgets.QLabel("^ choose analysis tool")

        self.file_picker = QtWidgets.QPushButton("Choose Weights")

        self.playpause_button = QtWidgets.QPushButton("Play")
        self.playpause_button.setCheckable(True)

        self.step_button = QtWidgets.QPushButton("Step")
        self.reset_button = QtWidgets.QPushButton("Reset")

        self.speed_label = QtWidgets.QLabel("Simulation Speed: 1")
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(1)

        self.erase_label = QtWidgets.QLabel("Erase Size: 1")
        self.erase_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.erase_slider.setMinimum(1)
        self.erase_slider.setMaximum(6)
        self.erase_slider.setValue(1)

        # connecting buttons to functions
        self.playpause_button.toggled.connect(self._play_pause_toggled)
        self.step_button.clicked.connect(self._step_clicked)
        self.reset_button.clicked.connect(self._reset_clicked)
        self.speed_slider.valueChanged.connect(self._change_sim_speed)
        self.erase_slider.valueChanged.connect(self._change_erase_size)
        self.file_picker.clicked.connect(self._open_weight_picker)
        self.analysis_tool.currentTextChanged.connect(self.update_analysis_tool_label)

        # vertical layout of the toolbar buttons and sliders
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.analysis_tool)
        layout.addWidget(self.analysis_tool_label)
        layout.addWidget(self.file_picker)
        layout.addWidget(self.playpause_button)
        layout.addWidget(self.step_button)
        layout.addWidget(self.reset_button)
        layout.addWidget(self.speed_label)
        layout.addWidget(self.speed_slider)
        layout.addWidget(QtWidgets.QLabel(""))
        layout.addWidget(self.erase_label)
        layout.addWidget(self.erase_slider)
        self.setLayout(layout)

    def _change_sim_speed(self, value: float) -> None:
        self.speed_label.setText(f"Simulation Speed: {value}")
        self.sim_speed_requested.emit(value)

    def _change_erase_size(self, value: int) -> None:
        self.erase_label.setText(f"Erase Size: {value}")
        self.erase_size_requested.emit(value)

    def _open_weight_picker(self) -> None:
        dialog = QtWidgets.QFileDialog()
        dialog.setNameFilter("*.npz")
        if dialog.exec():
            filenames = dialog.selectedFiles()
            if filenames:
                file_path = filenames[0]
                try:
                    npz = np.load(file_path)
                except OSError as _:
                    print("Weights are not found.")
                    return

                arrays = [npz[file] for file in npz.files]
                self.on_new_weights.emit((arrays[0], arrays[1]))

    def _play_pause_toggled(self, checked: bool) -> None:  # noqa: FBT001
        if checked:
            self.playpause_button.setText("Pause")
        else:
            self.playpause_button.setText("Play")
        self.update_toggle.emit(checked)

    def _step_clicked(self) -> None:
        self.step_requested.emit(True)  # noqa: FBT003

    def _reset_clicked(self) -> None:
        self.reset_requested.emit(True)  # noqa: FBT003

    def update_analysis_tool_label(self, text: str) -> None:
        if text == "":
            self.analysis_tool_label.setText("^ choose analysis tool")
        elif self.grid is None:
            self.analysis_tool_label.setText("run simulation step to analyze")
        else:
            grid = self._parent.get_grid()
            self.analysis_tool_label.setText(
                str(self._analysis_tool[text](grid.state(layer=0))),
            )


class MainWindow(QtWidgets.QWidget):
    """Creates the main window.

    next_step_function:
        REQUIRES a next_step_function for which it inputs the current grid and
        expects the updated grid to be returned.

    analysis_tool dictionary:
        A dictionary of analysis tools to be used in the toolbar.
        Key: Name of the tool as shown in the combobox.
        Value: A function that takes the current grid as input and returns
               a float or string to be displayed in the analysis tool label.

    """

    def __init__(
        self,
        next_step_function: Callable | None = None,
        analysis_tool: dict[str, Callable[[npt.NDArray], Any]] | None = None,
        grid: Grid | None = None,
    ) -> None:
        if grid is None:
            grid = Grid(width=gridsize, height=gridsize, num_channels=5)
        self.grid = grid

        # initial settings
        if analysis_tool is None:
            analysis_tool = {}
        self.next_step_function = next_step_function
        self.speed = 1
        self.erase_size = 1
        super().__init__()
        self.grid_view = GridView(self.grid)
        self.toolbar = ToolBar(self, analysis_tool=analysis_tool)

        # timer for simulation speed
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000 // self.speed)  # ms
        self.timer.timeout.connect(self.step_simulation)

        # layout splitting grid-view and toolbar
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.grid_view)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

        # connecting buttons and sliders to functions
        self.toolbar.step_requested.connect(self.step_simulation)
        self.toolbar.sim_speed_requested.connect(self._set_sim_speed)
        self.toolbar.update_toggle.connect(self._on_play_toggled)
        self.grid_view.creation_signal.connect(self._create_function)
        self.grid_view.erase_signal.connect(self._erase_function)
        self.toolbar.erase_size_requested.connect(self._set_erase_size)
        self.toolbar.reset_requested.connect(self._reset_grid)
        self.toolbar.on_new_weights.connect(self._set_weights)

        # window settings
        self.setWindowTitle("Evolution simulator")
        self.resize(600, 600)

    def step_simulation(self) -> None:
        if self.next_step_function is None:
            raise RuntimeError("No next_step_function defined")

        self.grid.step()
        self.grid_view.update_grid(self.grid.state(layer=0))
        self.toolbar.update_analysis_tool_label(
            self.toolbar.analysis_tool.currentText(),
        )

    def _on_play_toggled(self, checked: bool):  # noqa: FBT001
        if checked:
            self.timer.start()
        else:
            self.timer.stop()

    def _set_weights(self, weights: tuple[npt.NDArray, npt.NDArray]) -> None:
        self.grid.set_weights_on_nn(weights)

    def _set_sim_speed(self, speed: int) -> None:
        self.speed = speed
        self.timer.setInterval(1000 // self.speed)

    def _set_erase_size(self, size: int) -> None:
        self.erase_size = size

    def _create_function(self, row: int, col: int) -> None:
        self.grid_view.grid[row, col] = 1
        self.grid_view.update_grid(self.grid_view.grid)

    def _erase_function(self, row: int, col: int) -> None:
        coordinates = get_filled_circle_coordinates(row, col, self.erase_size)
        empty = self.grid.empty
        for r, c in coordinates:
            if (
                0 <= r < self.grid_view.grid.shape[0]
                and 0 <= c < self.grid_view.grid.shape[1]
            ):
                self.grid.set_cell_state(c, r, empty)
        self.grid_view.update_grid(self.grid_view.grid)

    def _reset_grid(self) -> None:
        self.grid.clear_and_seed()
        self.grid_view.update_grid(self.grid_view.grid)
        self.toolbar.update_analysis_tool_label(
            self.toolbar.analysis_tool.currentText(),
        )

    def get_grid(self) -> Grid:
        return self.grid


class GridView(FigureCanvas):
    """The visualisation "widget" of the grid."""

    # signals for mouse interactions
    erase_signal = QtCore.pyqtSignal(int, int)
    creation_signal = QtCore.pyqtSignal(int, int)

    def __init__(self, grid: Grid):
        # setting up the matplotlib figure
        self.fig = Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111)

        super().__init__(self.fig)
        self.setMinimumSize(QtCore.QSize(500, 500))

        # grid setup
        self.seed_vector = torch.zeros(5, dtype=torch.float32, device=None)
        self.seed_vector[0] = 1.0  # aliveness
        self.seed_vector[1:5] = 0.0  # alpha channel

        self.grid_source = grid
        self.grid = self.grid_source.state(layer=0)
        # TODO(sijmen): add combobox for person to change view to be drawn
        # colormap setup
        self.cmap = ListedColormap(["white", "black"])
        self.norm = BoundaryNorm([0, 1, 2], self.cmap.N)
        self.im = self.ax.imshow(
            self.grid,
            cmap=self.cmap,
            norm=self.norm,
            origin="upper",
            interpolation="nearest",
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # connecting mouse events
        self.mpl_connect("motion_notify_event", self.draw_cells)

    def update_grid(self, new_grid: npt.NDArray) -> None:
        self.grid = new_grid

        self.im.set_data(self.grid)
        self.draw_idle()

    def draw_cells(self, event: MouseEvent) -> None:
        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        row = int(event.ydata + 0.5)
        col = int(event.xdata + 0.5)

        if 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]:
            if event.button == 3:  # right click
                self.creation_signal.emit(row, col)

            elif event.button == 1:  # left click
                self.erase_signal.emit(row, col)


def get_filled_circle_coordinates(
    center_row,
    center_col,
    radius,
):  # <-- this is an LLM function
    """Get all grid coordinates inside a circle (filled circle).

    Parameters
    ----------
    center_row : int
        Row coordinate of the circle center
    center_col : int
        Column coordinate of the circle center
    radius : int
        Radius of the circle

    Returns
    -------
    list of tuples
        List of (row, col) coordinates inside the circle

    """
    coordinates = []
    radius_squared = radius * radius

    # Iterate through a bounding box
    for row in range(center_row - radius, center_row + radius + 1):
        for col in range(center_col - radius, center_col + radius + 1):
            # Calculate distance squared (avoids sqrt for performance)
            dx = col - center_col
            dy = row - center_row
            if dx * dx + dy * dy <= radius_squared:
                coordinates.append((row, col))
    return coordinates


def test_CA(grid: np.ndarray) -> np.ndarray:
    # testing a simple CA update rule
    new_grid = np.copy(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if any(grid[i - 1 : i + 2, j]) or any(grid[i, j - 1 : j + 2]):
                new_grid[i, j] = 1
    return new_grid


if __name__ == "__main__":
    grid = Grid(gridsize, gridsize, 5)
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(test_CA)
    w.show()
    app.exec()

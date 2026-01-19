import numpy as np


class Grid:
    def __init__(self, width: int, height: int) -> None:
        self._grid = np.zeros((height, width))

    def set_state(self, x: int, y: int, val):
        pass

    def step() -> None:
        """Perform a single step of the grid's CA."""
        pass

    @property
    def grid(self):
        return self._grid


# example of structure required for GUI and inputs
# from ui import MainWindow
# import sys
# from PyQt6 import QtWidgets
# import numpy as np

# def filled_calculator(grid:np.ndarray) -> float:
#     return np.sum(grid) / (grid.shape[0] * grid.shape[1])

# def tester_obama(grid:np.ndarray) -> np.ndarray:
#     return "obama"

# def test_CA(grid:np.ndarray) -> np.ndarray:
#     # testing a simple CA update rule
#     new_grid = np.copy(grid)
#     for i in range(grid.shape[0]):
#         for j in range(grid.shape[1]):
#             if any(grid[i-1:i+2, j]) or any (grid[i, j-1:j+2]):
#                 new_grid[i,j] = 1
#     return new_grid


def main():
    # app = QtWidgets.QApplication(sys.argv)
    # analysis_tool = {"Filled Calculator": filled_calculator, "Tester Obama": tester_obama}
    # w = MainWindow(next_step_function=test_CA, analysis_tool=analysis_tool)
    # w.show()
    # app.exec()  

    print("Hello from complexity-17!")


if __name__ == "__main__":
    main()

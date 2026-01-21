"""Allows for use of evolutionary algorithms for neural network training.

Name:        Mike Nieuweboer
Course:      Complex System Simulation

Description:
------------
Allows for the creation of the EA object. This object has several preset ways to employ
evolutionary strategies in creating and improving the fitness of a population of
the Neural CA. This fitness can be dependent on several factors, such as how much
the Neural CA looks like a circle.
"""

import logging
from collections.abc import Callable
from enum import Enum
from itertools import product
from pathlib import Path

# import cProfile
# import pstats
import nevergrad as ng
import numpy as np
import numpy.typing as npt
from nevergrad.parametrization.parameter import Parameter
from tqdm import trange

from grid import Grid

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_dir = Path(__file__).parent.parent / "logs"
file_handler = logging.FileHandler(log_dir / "ea.log", mode="a", encoding="utf-8")
logger.addHandler(file_handler)


class FitnessType(Enum):
    """Defines the different types of fitness evaluation."""

    TESTING = 0
    FULL_CIRCLE = 1


class EAType(Enum):
    """Defines the different modes of learning for the EA."""

    BASIC = 0
    PERSISTENT = 1
    REGENERATING = 2


class FitnessFunctions:
    @staticmethod
    def full_circle(grid: npt.NDArray, *, radius: float = 10) -> float:
        shape = grid.shape
        center = (shape[1] / 2, shape[0] / 2)

        def circle(x: float, y: float):
            return x * x + y * y - radius * radius

        loss = 0
        for x, y in product(range(shape[1]), range(shape[0])):
            result = circle(x - center[0], y - center[1])
            active = True  # TODO: True if alive, False otherwise
            loss += 1 if result >= 0 == active else 0
        return loss


class EA:
    def __init__(
        self,
        inputs: int,
        outputs: int,
        fitness_type: FitnessType,
        ea_type: EAType = EAType.BASIC,
        *,
        pop_count: int = 50,
        gen_count: int = 500,
        grid_size: int = 50,
    ) -> None:
        """TODO: Docstring"""
        self._gen = np.random.Generator(np.random.PCG64())

        self._fitness_type = fitness_type
        self._ea_type = ea_type
        self._pop_count = pop_count
        self._gen_count = gen_count

        self._grid_size = grid_size

        self._fitnesses = []

        self._init_optimizer(inputs, outputs)

    def _init_optimizer(self, inputs: int, outputs: int) -> None:
        # Paper used inputs->128->ReLu->16=delta vector
        weight_shapes = [
            (inputs, 64),
            (64, outputs),
        ]

        # Create the optimizer
        param = ng.p.Instrumentation(
            *[ng.p.Array(shape=weight, mutable_sigma=True) for weight in weight_shapes],
        )

        param.value = (
            tuple(self._gen.uniform(-5, 5, shape) for shape in weight_shapes),
            {},
        )

        self._optimizer = ng.optimizers.registry["CMAstd"](
            param,
            num_workers=self._pop_count,
            budget=self._pop_count * self._gen_count,
        )

    def save_stats(self) -> None:
        pass

    def save_weights(self) -> None:
        pass

    def save_optimizer(self) -> None:
        # fl.save("tmp.h5", optimizer.__getstate__())
        pass

    def load_optimizer(self) -> None:
        # # If the flag -r is set, use previous state of optimizer
        # if cli_args.restart:
        #     logger.info("Loading")
        #     attrs = fl.load("tmp.h5")
        #     for name, val in attrs.items():
        #         if name in {"budget", "num_workers"}:
        #             continue
        #         optimizer.__setattr__(name, val)
        pass

    def _evolve_testing(self, individual: Parameter) -> float:
        return np.sum(np.abs(individual.args[0] - 1))

    def _basic_simulation(self, grid: Grid, fitness: Callable) -> float:
        # Low and high taken from paper
        low = 64
        high = 96

        steps = int(self._gen.integers(low, high))
        grid.simulate(steps)
        return fitness(self._grid.grid)

    def _evolve_full_circle(self, grid: Grid) -> float:
        return self._basic_simulation(grid, FitnessFunctions.full_circle)

    def _evolve_step_basic(self, generation: int, function: Callable) -> None:
        # Reproduction
        population = [self._optimizer.ask() for _ in range(self._pop_count)]
        logger.info("Starting generation %d", generation)
        results = list(
            map(
                function,
                (Grid(50, 50, individual) for individual in population),
            ),
        )

        best = max(results)
        mean = np.mean(results)
        std = np.std(results)
        self._fitnesses.append(results)

        logger.info(
            "Finished generation with best: %f, mean: %f, std: %f",
            best,
            mean,
            std,
        )

        # Survivor selection
        for individual, loss in zip(population, results, strict=True):
            self._optimizer.tell(individual, loss)

    def _evolve_step_persistent(self, generation: int, function: Callable):
        pass

    def evolve(self) -> None:
        function = None
        match self._fitness_type:
            case FitnessType.TESTING:
                function = self._evolve_testing
            case FitnessType.FULL_CIRCLE:
                function = self._evolve_full_circle

        if function is None:
            logger.error("Current evolution type is not supported.")
            return

        for gen in trange(self._gen_count):
            match self._ea_type:
                case EAType.BASIC:
                    self._evolve_step_basic(gen, function)
                case EAType.PERSISTENT:
                    self._evolve_step_persistent(gen, function)


def main():
    ea = EA(48, 16, FitnessType.TESTING)
    ea.evolve()


if __name__ == "__main__":
    main()

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
import multiprocessing as mp
from collections.abc import Callable
from enum import Enum
from itertools import product
from multiprocessing.pool import Pool
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


class EAType(Enum):
    """Defines the different types of fitness evaluation."""

    TESTING = 0
    FULL_CIRCLE = 1


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
        ea_type: EAType,
        *,
        pop_count: int = 50,
        gen_count: int = 500,
        mp: bool = False,
    ) -> None:
        """TODO: Docstring"""
        self._gen = np.random.Generator(np.random.PCG64())
        self._mp = mp

        self._ea_type = ea_type
        self._pop_count = pop_count
        self._gen_count = gen_count

        self._grid = None

        self.fitnesses = []

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

        # TODO: Proper initialization
        self._optimizer = ng.optimizers.CMA(
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
        return 1
        return np.sum(np.abs(individual.args[0] - 1))

    def _basic_simulation(self, fitness: Callable) -> float:
        if self._grid is None:
            logger.error("No grid configured for basic simulation")
        # Low and high taken from paper
        low = 64
        high = 96

        steps = int(self._gen.integers(low, high))
        self._grid.simulate(steps)
        return fitness(self._grid.grid)

    def _evolve_full_circle(self, individual: Parameter) -> float:
        self._grid = Grid(40, 40)
        return self._basic_simulation(FitnessFunctions.full_circle)

    def _evolve_step(self, gen: int, function: Callable, pool: Pool | None) -> None:
        logger.info("Starting generation %d", gen)
        # self._population = [self._optimizer.ask() for _ in range(self._pop_count)]
        self._population = [1 for _ in range(self._pop_count)]
        if pool:
            results = pool.map(function, self._population)
        else:
            results = list(map(function, self._population))
        best = max(results)
        mean = np.mean(results)
        std = np.std(results)
        logger.info(
            "Finished generation with best: %f, mean: %f, std: %f",
            best,
            mean,
            std,
        )
        # for individual, loss in zip(self._population, results, strict=True):
        #     self._optimizer.tell(individual, loss)

    def evolve(self) -> None:
        function = None
        match self._ea_type:
            case EAType.TESTING:
                function = self._evolve_testing
            case EAType.FULL_CIRCLE:
                function = self._evolve_full_circle

        if function is None:
            logger.error("Current evolution type is not supported.")
            return

        if self._mp:
            with mp.Pool(16) as pool:
                for gen in trange(self._gen_count):
                    self._evolve_step(gen, function, pool)
        else:
            for gen in range(self._gen_count):
                self._evolve_step(gen, function, None)


def main():
    ea = EA(48, 16, EAType.TESTING, mp=True)
    # profiler = cProfile.Profile()
    # profiler.enable()
    ea.evolve()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("cumtime")
    # stats.print_stats()


if __name__ == "__main__":
    main()

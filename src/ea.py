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

import nevergrad as ng
import numpy as np
import numpy.typing as npt

from grid import Grid

logger = logging.getLogger(__name__)


class EAType(Enum):
    """Defines the different types of fitness evaluation."""

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
        logging.basicConfig(level=logging.INFO)

        self._ea_type = ea_type
        self._pop_count = pop_count
        self._gen_count = gen_count

        self._gen = np.random.Generator(np.random.PCG64())

        self._mp = mp

        self._grid = None

        # Paper used inputs->128->ReLu->16=delta vector
        weight_shapes = [
            (inputs, 64),
            (64, outputs),
        ]

        # Create the optimizer
        param = ng.p.Instrumentation(
            *[ng.p.Array(shape=weight, mutable_sigma=True) for weight in weight_shapes],
        )

        self._optimizer = ng.optimizers.CMA(
            param,
            num_workers=pop_count,
            budget=pop_count * gen_count,
        )

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

    def _basic_simulation(self, fitness: Callable) -> float:
        if self._grid is None:
            logger.error("No grid configured for basic simulation")
        # Low and high taken from paper
        low = 64
        high = 96

        steps = int(self._gen.integers(low, high))
        self._grid.simulate(steps)
        return fitness(self._grid.grid)

    def _evolve_full_circle(self, individual: object) -> float:
        self._grid = Grid(40, 40)
        return self._basic_simulation(FitnessFunctions.full_circle)

    def evolve(self) -> None:
        process_count = self._pop_count if self._mp else 1

        function = None
        match self._ea_type:
            case EAType.FULL_CIRCLE:
                function = self._evolve_full_circle

        if function is None:
            logger.error("Current evolution type is not supported.")
            return

        with mp.Pool(process_count) as pool:
            for gen in range(self._gen_count):
                logger.info("Starting generation %d", gen)
                population = [self._optimizer.ask() for _ in range(self._pop_count)]
                results = pool.map(function, population)
                for individual, loss in zip(population, results, strict=True):
                    self._optimizer.tell(individual, loss)

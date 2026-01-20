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
import pickle
from enum import Enum

import nevergrad as ng
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class EAType(Enum):
    """Defines the different types of fitness evaluation."""

    FULL_CIRCLE = 1


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

        # Paper used inputs->128->ReLu->16=delta vector
        weight_shapes = [
            (inputs, 64),
            (64, 16),
        ]

        # Create the optimizer
        param = ng.p.Instrumentation(
            *[ng.p.Array(shape=weight, mutable_sigma=True) for weight in weight_shapes]
        )
        optimizer = ng.optimizers.CMA(
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
        #         optimizer.__setattr__(name, val)  # noqa: PLC2801
        pass

    def _step_ca(self) -> npt.NDArray:
        return np.zeros((1, 1, 1))

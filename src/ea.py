"""Allows for use of evolutionary algorithms for neural network training.

Name:        Mike Nieuweboer
Course:      Complex System Simulation

Description:
------------
Allows for the creation of the EA object. This object has several presets for learning
a neural network to minimize a grid-based fitness value. These presets include:
- Basic fitness evaluation after a certain number of steps
- Persistent evolution using the worst grid states from previous generations to improve
    robustness
The fitness values depend on the given fitness type, of which the following
are implemented:
- Testing: Lowest fitness is achieved with all weights in the first layer being 1.
- FullCircle: Lowest fitness is achieved when the neural ca produces a circle structure.

AI Usage:
---------
AI was used to generate most of the doc comments for classes and functions, using the
following prompt:
> Analyse the functionality of @src/ea.py, assuming that the rest of the codebase is
> implemented and write the required docstrings for the classes and functions within.

"""

import csv
import logging
from collections.abc import Callable
from enum import Enum
from itertools import product
from pathlib import Path

import nevergrad as ng
import numpy as np
import numpy.typing as npt
from nevergrad.optimization import Optimizer
from tqdm import trange

from grid import Grid

root_dir = Path(__file__).parent.parent

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_dir = root_dir / "logs"
log_dir.mkdir(exist_ok=True, parents=True)
file_handler = logging.FileHandler(log_dir / "ea.log", mode="a", encoding="utf-8")
logger.addHandler(file_handler)

# Extra file paths
weights_dir = root_dir / "weights"
weights_dir.mkdir(exist_ok=True, parents=True)
data_dir = root_dir / "data"
data_dir.mkdir(exist_ok=True, parents=True)
state_dir = data_dir / "states"
state_dir.mkdir(exist_ok=True, parents=True)


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
    """Provides static methods for evaluating Neural CA fitness.

    Fitness functions measure how well a Neural CA configuration achieves
    a desired pattern or behavior. Lower fitness values indicate better
    performance.
    """

    @staticmethod
    def full_circle(grid: npt.NDArray, *, radius: float = 10) -> float:
        """Calculate fitness for circle pattern formation.

        Evaluates how well the Neural CA produces a circular pattern at the grid
        center. The fitness is the count of grid cells that deviate from the
        expected circle pattern.

        Args:
        ----
            grid: A 3D numpy array representing the Neural CA grid state.
            radius: The target radius of the circle in grid units. Defaults to 10.

        Returns:
        -------
            The fitness loss value (non-negative float). Lower values indicate better
            circle pattern formation. Returns 0 for perfect circle formation.

        """
        shape = grid.shape
        center = (shape[1] / 2, shape[0] / 2)

        def circle(x: float, y: float) -> float:
            return x * x + y * y - radius * radius

        loss = 0
        for x, y in product(range(shape[1]), range(shape[0])):
            result = circle(x - center[0], y - center[1])
            active = True  # TODO: True if alive, False otherwise
            loss += 1 if result >= 0 == active else 0
        return loss


class EA:
    """Evolutionary Algorithm for optimizing Neural CA network parameters.

    Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to evolve the weights
    of a Neural CA network to achieve desired fitness objectives. Supports multiple
    evolution strategies (basic and persistent) and fitness evaluation modes.
    """

    def __init__(  # noqa: PLR0913
        self,
        inputs: int,
        outputs: int,
        fitness_type: FitnessType,
        ea_type: EAType = EAType.BASIC,
        *,
        pop_count: int = 50,
        gen_count: int = 500,
        grid_size: int = 50,
        seed: int = 43,
        performance: bool = False,
    ) -> None:
        """Initialize the Evolutionary Algorithm optimizer.

        Sets up the CMA-ES optimizer with specified network architecture and
        evolution parameters. Creates grids and initializes tracking structures
        for fitness history.

        Args:
        ----
        inputs: Number of input neurons in the first layer.
        outputs: Number of output neurons in the final layer.
        fitness_type: The fitness evaluation mode (FitnessType enum value).
        ea_type: The evolution strategy to use (EAType enum value).
        pop_count: Population size per generation.
        gen_count: Number of generations to evolve.
        grid_size: Dimensions of the Neural CA grid (grid_size x grid_size).
        seed: Random seed for reproducibility.
        performance: Enable performance optimizations (diagonal CMA, high speed mode).

        """
        self._gen = np.random.Generator(np.random.PCG64(seed))
        np.random.seed(seed)  # noqa: NPY002

        self._fitness_type = fitness_type
        self._ea_type = ea_type
        self._performance = performance
        self._pop_count = pop_count
        self._gen_count = gen_count

        self._grids = []
        self._grid_size = grid_size

        self._fitnesses = []

        self._init_optimizer(inputs, outputs)

    def _init_optimizer(self, inputs: int, outputs: int) -> None:
        """Initialize the CMA-ES optimizer with network weight parameters.

        Sets up a parameterized CMA-ES optimizer configured for the Neural CA network
        architecture. Creates weight matrices for the network layers and initializes
        them with random values in the range [-1, 1].

        Args:
        ----
        inputs: Number of input features for the first weight matrix.
        outputs: Number of outputs for the final weight matrix.

        """
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
            tuple(self._gen.uniform(-1, 1, shape) for shape in weight_shapes),
            {},
        )

        self._optimizer: Optimizer = ng.optimizers.ParametrizedCMA(
            scale=0.4699,
            popsize_factor=3,
            diagonal=self._performance,
            high_speed=self._performance,
        ).set_name("CMAcustom", register=False)(
            param,
            num_workers=self._pop_count,
            budget=self._pop_count * self._gen_count,
        )
        self._optimizer.enable_pickling()

    def save_stats(self, *, append: bool = False) -> None:
        """Save fitness statistics to disk.

        Persists fitness history and evolution statistics for analysis and
        visualization.
        """
        file_path = data_dir / f"{self._fitness_type.name}_{self._ea_type.name}.csv"
        with file_path.open("w" if not append else "a") as file:
            writer = csv.writer(file)
            writer.writerows(self._fitnesses)

    def save_weights(self) -> None:
        """Save best network weights to disk.

        Persists the weights of the best-performing individual from the final
        generation.
        """
        file_path = weights_dir / f"{self._fitness_type.name}_{self._ea_type.name}.npz"
        weights = self._optimizer.recommend().args
        np.savez(file_path, *weights)

    @staticmethod
    def load_weights(
        fitness_type: FitnessType,
        ea_type: EAType,
    ) -> list[npt.NDArray]:
        """Load the network weights from the disk.

        Args:
        ----
        fitness_type: The fitness type the weights were evolved for.
        ea_type: The type of evolution process used in evolving the weights.

        """
        file_path = weights_dir / f"{fitness_type.name}_{ea_type.name}.npz"
        try:
            npz = np.load(file_path)
        except OSError as _:
            logger.warning("Weights are not found.")
            return []

        return [npz[file] for file in npz.files]

    def save_optimizer(self) -> None:
        """Save the complete optimizer state for resuming evolution.

        Persists the optimizer's internal state, allowing evolution to resume from
        the current generation without losing progress.
        """
        file_path = state_dir / f"{self._fitness_type.name}_{self._ea_type.name}.pickle"
        self._optimizer.dump(file_path)

    def load_optimizer(self) -> None:
        """Load a previously saved optimizer state to resume evolution.

        Restores the optimizer's internal state from disk, enabling continuation of
        evolution from the generation where it was previously saved.
        This optimizer uses its own matrix shapes for the weights, but does use
        the new number of generations and population size.
        """
        file_path = state_dir / f"{self._fitness_type.name}_{self._ea_type.name}.pickle"
        self._optimizer = Optimizer.load(file_path)
        self._optimizer.num_workers = self._pop_count
        self._optimizer.budget = self._pop_count * self._gen_count

    def _evolve_testing(self, grid: Grid) -> float:
        """Evaluate fitness using a simple testing metric.

        Args:
        ----
        grid: The Neural CA grid with candidate weights.

        Returns:
        -------
        Fitness loss value. Lower values indicate weights closer to 1.

        """
        if grid.weights is None:
            logger.error("No weights assigned")
            return 0
        return np.sum(np.abs(grid.weights[0] - 1))

    def _basic_simulation(self, grid: Grid, fitness: Callable) -> float:
        """Run Neural CA simulation and evaluate fitness.

        Simulates the Neural CA for a random number of steps within the paper's
        recommended range, then evaluates the resulting pattern with the provided
        fitness function.

        Args:
        ----
        grid: The Neural CA grid to simulate.
        fitness: A callable that takes a grid array and returns a fitness score.

        Returns:
        -------
        The fitness loss value after simulation.

        """
        # Low and high taken from paper
        low = 64
        high = 96

        steps = int(self._gen.integers(low, high))
        grid.run_simulation(steps=steps)
        return fitness(grid.state)

    def _evolve_full_circle(self, grid: Grid) -> float:
        """Evaluate fitness for circle pattern formation.

        Returns
        -------
        Fitness loss value. Lower values indicate better circle formation.


        """
        return self._basic_simulation(grid, FitnessFunctions.full_circle)

    def _evolve_step_basic(
        self,
        generation: int,
        function: Callable[[Grid], float],
    ) -> None:
        """Execute one generation of basic evolutionary strategy.

        Creates a population of candidate solutions, evaluates their fitness,
        and updates the optimizer using the results. The basic strategy treats each
        generation independently without maintaining state across generations.

        Args:
        ----
        generation: Current generation number (for logging).
        function: Fitness evaluation function that takes a Grid and returns a float.

        """
        # Reproduction
        population = [self._optimizer.ask() for _ in range(self._pop_count)]
        logger.info("Starting generation %d", generation)
        results = list(
            map(
                function,
                (Grid(50, 50, weights=individual.args) for individual in population),
            ),
        )

        best = min(results)
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

    def _evolve_step_persistent(
        self,
        generation: int,
        fitness_eval: Callable[[Grid], float],
    ) -> None:
        """Execute one generation of persistent evolutionary strategy.

        Maintains a pool of Neural CA grids across generations. Each individual from
        the current population is applied to sampled grids from the pool. The sampled
        grids are replaced with evolved versions, creating a persistent
        environment where changes accumulate over time.

        Args:
        ----
        generation: Current generation number (for logging).
        fitness_eval: Fitness evaluation function that takes a Grid and returns a float.

        """
        pool_size = 1024
        if self._grids == []:
            self._grids = [(Grid(50, 50), 0) for _ in range(pool_size)]

        # Samples
        sample_size = 32
        indices = self._gen.choice(pool_size, (sample_size,), replace=False)  # pyright: ignore[reportCallIssue, reportArgumentType]
        samples: list[tuple[Grid, int]] = [self._grids[i] for i in indices]
        smallest_sample = min(range(sample_size), key=lambda x: samples[x][1])
        samples[smallest_sample] = (Grid(50, 50), 0)

        # Reproduction
        population = [self._optimizer.ask() for _ in range(self._pop_count)]

        # Fitness evaluation
        results = []
        new_samples = []
        logger.info("Starting generation %d", generation)
        for i, individual in enumerate(population):
            mean = 0
            for j, sample in enumerate(samples):
                copy = sample[0].deepcopy()
                copy.set_weights(individual.args)
                result = fitness_eval(copy)
                mean += result

                # Pick the worst evolutions of the grids as the
                # new samples for the next generation.
                if i == 0:
                    new_samples.append((copy, result))
                elif result > new_samples[j][1]:
                    new_samples[j] = (copy, result)

            # Mean of loss, mimicking the batched gradient descent.
            mean /= sample_size
            results.append(mean)

        # Place the newly generated samples back
        for i, j in enumerate(indices):
            self._grids[j] = new_samples[i]

        best = min(results)
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

    def evolve(self) -> None:
        """Execute the main evolutionary algorithm loop.

        Runs the evolution process for the specified number of generations, using the
        configured fitness type and evolutionary strategy. Logs progress and fitness
        statistics at each generation.
        """
        function = None
        match self._fitness_type:
            case FitnessType.TESTING:
                function = self._evolve_testing
            case FitnessType.FULL_CIRCLE:
                function = self._evolve_full_circle

        if function is None:
            logger.error("Current evolution type is not supported.")
            return
        self._grids = []

        for gen in trange(self._gen_count, smoothing=0.01):
            match self._ea_type:
                case EAType.BASIC:
                    self._evolve_step_basic(gen, function)
                case EAType.PERSISTENT:
                    self._evolve_step_persistent(gen, function)


def main() -> None:  # noqa: D103
    ea = EA(48, 16, FitnessType.TESTING, ea_type=EAType.BASIC, performance=True)
    ea.evolve()
    ea.save_optimizer()


if __name__ == "__main__":
    main()

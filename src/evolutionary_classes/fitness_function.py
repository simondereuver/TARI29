"""
Module containing fitness function(s)
"""

import warnings
from typing import List
from multiprocessing import Pool
import numpy as np

class FitnessFunction:
    """FitnessFunction class definiton"""
    def __init__(self, graph: np.ndarray, bounds: tuple = None):
        """Fitness function init"""
        if graph is not None:
            self.graph = graph
        else:
            raise RuntimeError("No graph. Exiting...")
        if bounds is not None:
            self.bounds = bounds
        else:
            warnings.warn("No boundaries set. FitnessFunction.bound is None", RuntimeWarning)

    def _calculate_distance(self, route: List[int]) -> int:
        """Helper function for calculating the distance"""
        if route is None or len(route) < 2:
            raise ValueError("The provided route is less than 2. Exiting...")

        x = np.sum(self.graph[route[:-1], route[1:]])
        x += self.graph[route[-1], route[0]]

        return x

    def fitness_function_distance_based(self, route: List[int]) -> int:
        """Distance based fitness function"""
        # evaluates the fitness score based on the total_distance. The lower the value, the better.
        return self._calculate_distance(route)

    def fitness_function_normalized(self, route: List[int]) -> float:
        """Normalized version"""
        # if there are no bounds, then we evaluate the fitness score as 1 / total_distance
        if self.bounds is None:
            return 1 / self._calculate_distance(route)

        # else we normalize the value with the bounds as
        # (higher - total_distance) / (higher - lower),
        # where the fitness score is represented as a float from 0 to 1 where 1 is a perfect fit
        x = self._calculate_distance(route)

        lower, higher = self.bounds

        if higher is None:
            warnings.warn("Only a lower bound was provided. Returning lower_bound / total_distance",
                           RuntimeWarning)
            return lower / x

        return (higher - x) / (higher - lower)

    def compute_fitness_scores(self, generation: list) -> np.ndarray:
        """Compute all fitness scores for an entire generation"""
        gen = np.array(generation)
        def eval_fitness(path: np.array) -> float:
            return self.bounds[0] / self._calculate_distance(path)

        with Pool() as pool:
            fitness_scores = np.array(pool.map(eval_fitness, gen))

        return fitness_scores

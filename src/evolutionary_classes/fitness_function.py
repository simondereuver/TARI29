"""
Module containing fitness function(s)
"""

from typing import List
import numpy as np

class FitnessFunction:
    """FitnessFunction class definiton"""
    def __init__(self, graph: np.ndarray, bounds: tuple = None):
        """Fitness function init"""
        self.graph = graph
        self.bounds = bounds

    def fitness_function_distance_based(self, route: List) -> int:
        """Distance based fitness function"""
        # evaluates the fitness score based on the total_distance. The lower the value, the better.
        x = 0

        for i in range(len(route) - 1):
            x += self.graph[route[i]][route[i + 1]]

        return x

    def fitness_function_normalized(self, route: List) -> float:
        """Normalized version"""
        # if there are no bounds, then we evaluate the fitness score as 1 / total_distance
        if self.bounds is None:
            x = 0
            for i in range(len(route) - 1):
                x += self.graph[route[i]][route[i + 1]]

            return 1/x

        # else we normalize the value with the bounds as
        # (higher - total_distance) / (higher - lower),
        # where the fitness score is represented as a float from 0 to 1 where 1 is a perfect fit
        x = 0
        for i in range(len(route) - 1):
            x += self.graph[route[i]][route[i + 1]]

        lower, higher = self.bounds
        return (higher - x) / (higher - lower)

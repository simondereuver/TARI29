"""Implementation of fitness calculations, uses the lowerbound"""
# pylint: disable=too-many-arguments,line-too-long
import numpy as np

def calculate_distance(graph: np.ndarray,
                       tour: np.ndarray) -> int:
    """Helper function for calculating the distance"""

    if tour is None or len(tour) < 2:
        raise ValueError("The provided path is less than 2. Exiting...")
    #sum the distance
    x = np.sum(graph[tour[:-1], tour[1:]])
    x += graph[tour[-1], tour[0]]

    return x

def eval_fitness(graph: np.ndarray,
                 tour: np.array,
                 bounds: tuple) -> float:
    """Evaluates the fitness score with the lowerbound"""
    return bounds[0] / calculate_distance(graph, tour)

def compute_fitness_scores(graph: np.ndarray,
                           generation: np.ndarray,
                           bounds: tuple) -> np.ndarray:
    """Compute all fitness scores for an entire generation"""

    fitness_scores = np.array([eval_fitness(graph, p, bounds) for p in generation])

    return fitness_scores

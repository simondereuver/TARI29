"""Test implementation of threaded fitness calculations"""
# pylint: disable=too-many-arguments,line-too-long
#from multiprocessing import Pool
#from concurrent.futures import ProcessPoolExecutor
import numpy as np


def calculate_distance(graph: np.ndarray, path: np.ndarray) -> int:
    """Helper function for calculating the distance"""

    if path is None or len(path) < 2:
        raise ValueError("The provided path is less than 2. Exiting...")

    x = np.sum(graph[path[:-1], path[1:]])
    x += graph[path[-1], path[0]]

    return x

def eval_fitness(graph: np.ndarray, path: np.array, bounds: tuple) -> float:
    """Evaluates the fitness score with the lowerbound"""
    return bounds[0] / calculate_distance(graph, path)

def compute_fitness_scores(graph: np.ndarray, generation: np.ndarray, bounds: tuple) -> np.ndarray:
    """Compute all fitness scores for an entire generation"""

    fitness_scores = np.array([eval_fitness(graph, p, bounds) for p in generation])

    #with Pool() as pool:
    #    fitness_scores = np.array(pool.starmap(eval_fitness, [(graph, p, bounds) for p in generation]))

    #with ProcessPoolExecutor(max_workers=8) as executor:
    #    futures = [executor.submit(eval_fitness, graph, p, bounds) for p in generation]
    #    fitness_scores = np.array([f.result() for f in futures])

    return fitness_scores

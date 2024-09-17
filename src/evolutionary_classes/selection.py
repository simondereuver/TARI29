# selection.py
"""
Module containing selection, solutions are selected based on their fitness to produce offspring
"""
import random
import numpy as np

class Selection:
    """A class to choose the survivors of a generation"""
    def __init__(self):
        """
        Initialize Selection class.
        """
        #add something to __init__, maybe which selection style to be used or something

    def distance(self, graph: np.ndarray, path: list) -> int:
        """Calculate the total distance of the given path in the graph."""
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += graph[path[i]][path[i + 1]]
        total_distance += graph[path[-1]][path[0]]

        return total_distance

    def survivors(self, graph: np.ndarray, old_generation: list) -> list:
        """Select half of the population based on minimum distances."""
        #implement the fitness function here instead
        survivors = []
        random.shuffle(old_generation)
        mid = len(old_generation) // 2

        for i in range(mid):
            distance_1 = self.distance(graph, old_generation[i])
            distance_2 = self.distance(graph, old_generation[i + mid])
            if distance_1 < distance_2:
                survivors.append(old_generation[i])
            else:
                survivors.append(old_generation[i + mid])

        return survivors

    # --- Optional Selection Methods ---

    #   Implement something like this instead of current survivors logic method:
    # def elitist_selection() -> list:
    # def tournament_selection() -> list:

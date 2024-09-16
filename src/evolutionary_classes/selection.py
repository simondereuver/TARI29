"""
Module containing selection, solutions are selected based on their fitness to produce offspring
"""
import random
import numpy as np

class Selection:

    def _distance(self, graph: np.ndarray, path : list) -> int:
        """Calculate the total distance of the given path in the graph."""
        total_distance = 0

        # Iterate through the path to sum the distances between consecutive nodes
        for i in range(len(path) - 1):
            total_distance += graph[path[i]][path[i + 1]]

        # Add the distance from the last node back to the start node to complete the tour
        total_distance += graph[path[-1]][path[0]]

        return total_distance

    def survivors(self, graph: np.ndarray, old_generation: list) -> list:
        """Select half of the population based on minimum distances."""
        survivors = []
        random.shuffle(old_generation)  #shuffle to randomize selection
        mid = len(old_generation) // 2


        for i in range(mid):
            distance_1 = self._distance(graph, old_generation[i])
            distance_2 = self._distance(graph, old_generation[i + mid])

            #append the path with the smallest distance
            if distance_1 < distance_2:
                survivors.append(old_generation[i])
            else:
                survivors.append(old_generation[i + mid])

        return survivors

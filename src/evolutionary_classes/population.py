"""
Module containing population-based search
"""
# pylint: disable=too-many-arguments,line-too-long
import random
import numpy as np
from evolutionary_classes.crossover import Crossover
from evolutionary_classes.selection import Selection

class Population:
    """Class for generating and modifying the population"""

    def __init__(
        self,
        mutation_rate=0.01,
        crossover_method="OX",
        graph=None
    ):
        """
        Initialize Population class.

        Args:
            mutation_rate (float): Probability of mutation (0 <= mutation_rate <= 1).
            population_size_range (tuple): Range for the initial population size.
            crossover_method (str): Method to use for crossover.
            graph (optional): Graph data required for SCX method.
        """
        if not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")

        self.graph = graph
        self.mutation_rate = mutation_rate
        self.crossover_method = crossover_method
    """
    def initial_population(self, graph: np.ndarray, population_size: int) -> np.ndarray:
        #Generate the initial population.
        total_destinations = graph.shape[0]

        random_paths = np.empty((population_size, total_destinations), dtype=int)

        for i in range(population_size):
            path = list(range(1, total_destinations))
            random.shuffle(path)

            path = [0] + path

            random_paths[i] = path

        return random_paths
    """
    def initial_population(self, graph: np.ndarray, population_size: int) -> np.ndarray:
        #Generate the initial population.
        total_destinations = graph.shape[0]

        random_paths = np.empty((population_size, total_destinations), dtype=int)

        for i in range(population_size):
            if i >= population_size // 2:
                random_paths[i] = self.nearest_neighbor(graph)
            else:
                path = list(range(1, total_destinations))
                random.shuffle(path)

                path = [0] + path

                random_paths[i] = path

        return random_paths

    def nearest_neighbor(self, graph: np.ndarray):
        #Nearest neighbour for initial pop
        nodes = len(graph)
        start = random.randint(0, nodes - 1)

        unvisited = set(range(nodes))
        unvisited.remove(start)
        tour = [start]
        current_node = start

        while unvisited:
            next_node = min(unvisited, key=lambda node: graph[current_node][node])
            tour.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node

        return tour

    def crossovers(self, survivors: np.ndarray, num_children_needed: int) -> np.ndarray:
        """Creates the required number of children from the survivors."""
        crossover = Crossover(self.crossover_method, self.graph)

        num_survivors = len(survivors)
        num_genes = survivors.shape[1]

        children = np.empty((num_children_needed, num_genes), dtype=survivors.dtype)

        for idx in range(num_children_needed):
            #randomly select two parents from the survivors
            parent_indices = np.random.choice(num_survivors, 2, replace=False)
            parent_a = survivors[parent_indices[0]]
            parent_b = survivors[parent_indices[1]]

            #create a child from the parents
            child = crossover.create_children(parent_a, parent_b)
            children[idx] = child

        return children


    def mutate_population(self, generation: np.ndarray) -> np.ndarray:
        """Mutates a small percentage of the population using numpy arrays."""

        population_size, path_length = generation.shape

        for i in range(population_size):
            if random.random() < self.mutation_rate:
                path = generation[i]
                index1, index2 = random.sample(range(1, path_length), 2)
                path[index1], path[index2] = path[index2], path[index1]
                generation[i] = path

        return generation

    def gen_new_population(self, curr_gen: np.ndarray, selection: Selection, fitness_scores: np.ndarray) -> np.ndarray:
        """
        Generate a new population using selection, crossover, and mutation.
        """
        population_size = len(curr_gen)

        survivors = selection.select_survivors(curr_gen, fitness_scores)
        num_survivors = len(survivors)
        num_children_needed = population_size - num_survivors
        #print(f"num children needed{num_children_needed}")

        children = self.crossovers(survivors, num_children_needed)
        new_population = np.concatenate((survivors, children))
        new_population = self.mutate_population(new_population)

        return new_population

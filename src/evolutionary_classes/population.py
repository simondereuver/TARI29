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
        mutation_rate: float = 0.01,
        crossover_method: str = "OX",
        graph: np.ndarray = None
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

    def initial_population(self,
                           graph: np.ndarray,
                           population_size: int) -> np.ndarray:
        """
        Generate the initial population.
        Half are randomly generated paths other half are generated using the nearest neighbor heuristic.
        """
        #get total number of nodes in graph
        total_destinations = graph.shape[0]

        #intialize the array for random tours
        random_tours = np.empty((population_size, total_destinations), dtype=int)

        for i in range(population_size):
            if i >= population_size // 2:
                random_tours[i] = self.nearest_neighbor(graph)
            else:
                tour = list(range(1, total_destinations))
                random.shuffle(tour)
                tour = [0] + tour
                random_tours[i] = tour

        return random_tours

    def nearest_neighbor(self, graph: np.ndarray):
        """
        Nearest neighbour for initial pop

        Args:
            graph (np.ndarray): A 2D array representing the graph (distance matrix) of destinations.

        Returns:
            np.ndarray: A list representing the path generated by the nearest neighbor heuristic.
        """
        #get total number of nodes in graph
        nodes = graph.shape[0]
        #choose a random starting node
        start = random.randint(0, nodes - 1)

        unvisited = set(range(nodes))
        unvisited.remove(start)
        tour = [start]
        current_node = start

        #create the tour by selecting the nearest unvisited node each time
        while unvisited:
            next_node = min(unvisited, key=lambda node: graph[current_node][node])
            tour.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node

        return tour

    def crossovers(self,
                   survivors: np.ndarray,
                   num_children_needed: int) -> np.ndarray:
        """
        Creates the required number of children from the survivors.

        Args:
            survivors (np.ndarray): Array of the survivors from the previous generation.
            num_children_needed (int): The number of children required to form a new population.

        Returns:
            np.ndarray: Children generated from the survivors.
        """
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


    def mutate_population(self,
                          generation: np.ndarray) -> np.ndarray:
        """
        Mutates a small percentage of the population using numpy arrays.
        
        Args:
            generation (np.ndarray): The current generation.

        Returns:
            np.ndarray: The mutated generation.
        """

        population_size, path_length = generation.shape
        #randomly select and mutate individuals in the population
        for i in range(population_size):
            if random.random() < self.mutation_rate:
                path = generation[i]
                index1, index2 = random.sample(range(1, path_length), 2)
                path[index1], path[index2] = path[index2], path[index1]
                generation[i] = path

        return generation

    def gen_new_population(self,
                           current_generation: np.ndarray,
                           selection: Selection,
                           fitness_scores: np.ndarray) -> np.ndarray:
        """
        Generate a new population using selection, crossover, and mutation.

        Args:
            curr_gen (np.ndarray): The current generation.
            selection (Selection): The selection class used to choose survivors.
            fitness_scores (np.ndarray): Fitness scores corresponding to the current generation.

        Returns:
            np.ndarray: The new population after selection, crossover, and mutation.
        """
        population_size = len(current_generation)

        survivors = selection.select_survivors(current_generation, fitness_scores)
        num_survivors = len(survivors)
        num_children_needed = population_size - num_survivors
        #print(f"num children needed{num_children_needed}")

        children = self.crossovers(survivors, num_children_needed)
        new_population = np.concatenate((survivors, children))
        new_population = self.mutate_population(new_population)

        return new_population

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
        # Maybe should use the mutation range based on the size of population
        if not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")

        self.graph = graph
        self.mutation_rate = mutation_rate
        self.crossover_method = crossover_method

    def initial_population(self, graph: np.ndarray, population_size: int) -> np.ndarray:
        """Generate the initial population."""
        total_destinations = graph.shape[0]

        random_paths = np.empty((population_size, total_destinations), dtype=int)

        for i in range(population_size):
            path = list(range(1, total_destinations))
            random.shuffle(path)

            path = [0] + path

            random_paths[i] = path

        return random_paths

    def crossovers(self, survivors: np.ndarray) -> np.ndarray:
        """Creates crossovers using NumPy arrays."""
        crossover = Crossover(self.crossover_method, self.graph)

        mid = survivors.shape[0] // 2
        parents_a = survivors[:mid]
        parents_b = survivors[mid:]

        num_children = 2 * mid
        num_genes = survivors.shape[1]

        children = np.empty((num_children, num_genes), dtype=survivors.dtype)

        for idx, (parent_a, parent_b) in enumerate(zip(parents_a, parents_b)):
            child1 = crossover.create_children(parent_a, parent_b)
            child2 = crossover.create_children(parent_b, parent_a)
            children[2 * idx] = child1
            children[2 * idx + 1] = child2

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
        """Generate a new population using selection, crossover, and mutation."""
        #elites = selection.elitism(curr_gen, fitness_scores, 0.1)
        #selection_methods = [
        #                        (selection.roulette_wheel_selection, 0.8),
                                #(selection.tournament, 0.4),
                                #(selection.rank_selection, 0.05),
                                #(selection.stochastic_universal_sampling, 0.8)
        #                    ]
        survivors = np.empty((0, len(curr_gen[0])), dtype=int)
        survivors = selection.select_survivors(curr_gen, fitness_scores)


        #for method, survive_rate in selection_methods:
        #    selected = method(curr_gen, fitness_scores, survive_rate)
        #    survivors = np.concatenate((survivors, selected))

        #children = self.crossovers(np.concatenate((survivors, elites)))
        children = self.crossovers(survivors)
        #combined_population = np.concatenate((children, elites))
        #new_population = self.mutate_population(combined_population)
        new_population = np.concatenate((survivors, children))
        new_population = self.mutate_population(new_population)
        return new_population

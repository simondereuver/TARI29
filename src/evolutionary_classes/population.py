"""
Module containing population-based search
"""

import random
import numpy as np
from evolutionary_classes.crossover import Crossover
from evolutionary_classes.selection import Selection
from evolutionary_classes.fitness_function import FitnessFunction


class Population:
    """Class for generating and modifying the population"""

    def __init__(
        self,
        mutation_rate=0.01,
        population_size_range=(10, 50),
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
        if not (isinstance(population_size_range, tuple) and len(population_size_range) == 2):
            raise ValueError("population_size_range must be a tuple with two values")
        if population_size_range[0] >= population_size_range[1]:
            raise ValueError(
                "population_size_range must be a tuple (min, max) where min < max"
            )

        self.graph = graph
        self.mutation_rate = mutation_rate
        self.population_size_range = population_size_range
        self.crossover_method = crossover_method

        if crossover_method == "SCX":
            self.crossover_manager = Crossover(crossover_method, graph)
        else:
            self.crossover_manager = Crossover(crossover_method)

    def initial_population(self, graph: np.ndarray) -> list:
        """Generate the initial population."""
        total_destinations = graph.shape[0]
        random_paths = []

        # Implement some logic based on number of nodes
        # Since it is n! we should probably have a max limit on this
        # so we don't get too big of a population
        min_size = self.population_size_range[0] * total_destinations
        max_size = self.population_size_range[1] * total_destinations
        population_size = random.randint(min_size, max_size)

        for _ in range(population_size):
            random_path = list(range(1, total_destinations))
            random.shuffle(random_path)
            random_path = [0] + random_path
            random_paths.append(random_path)

        return random_paths

    def crossovers(self, survivors: list) -> list:
        """Creates crossovers using the _create_children method."""
        # There are different crossover methods, we need to test to see which gives best result
        # crossover = Crossover(self.crossover_method)
        crossover = self.crossover_manager

        children = []
        mid = len(survivors) // 2
        for i in range(mid):
            parent_a, parent_b = survivors[i], survivors[i + mid]
            children.append(crossover.create_children(parent_a, parent_b))
            children.append(
                crossover.create_children(parent_b, parent_a)
            )  # pylint: disable=arguments-out-of-order
        return children

    def mutate_population(self, generation: list) -> list:
        """Mutates a small percentage of the population."""
        # Maybe should use the mutation percentage based on the size of population
        # Test different percentages
        mutated_population = []
        for path in generation:
            if random.random() < self.mutation_rate:
                index1, index2 = random.sample(range(1, len(path)), 2)
                path[index1], path[index2] = path[index2], path[index1]
            mutated_population.append(path)
        return mutated_population

    def gen_new_population(
        self, curr_gen: list, selection: Selection, ff: FitnessFunction
    ) -> list:
        """Generate a new population using selection, crossover, and mutation."""
        survivors = selection.survivors(curr_gen, ff)
        children = self.crossovers(survivors)
        combined_population = survivors + children
        new_population = self.mutate_population(combined_population)
        return new_population

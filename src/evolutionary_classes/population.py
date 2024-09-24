# population.py
"""
Module containing population-based search
"""

import random
import numpy as np
from evolutionary_classes.selection import Selection
from evolutionary_classes.fitness_function import FitnessFunction

class Population:
    """Class for generating and modifying the population"""
    def __init__(self, mutation_rate=0.01, population_size_range=(10, 50)):
        """
        Initialize Population class.
        
        Args:
            mutation_rate (float): Probability of mutation (0 <= mutation_rate <= 1).
            population_size_range (tuple): Range for the initial population size.
        """
        #maybe should use the mutation range based on the size of population
        if not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")
        if not (isinstance(population_size_range, tuple) and len(population_size_range) == 2):
            raise ValueError("population_size_range must be a tuple with two values")
        if population_size_range[0] >= population_size_range[1]:
            raise ValueError("population_size_range must be a tuple (min, max) where min < max")

        self.mutation_rate = mutation_rate
        self.population_size_range = population_size_range

    def initial_population(self, graph: np.ndarray) -> list:
        """Generate the initial population."""
        total_destinations = graph.shape[0]
        random_paths = []

        #implement some logic based on number of nodes
        #since it is n! we should probably have a max limit on this
        #so we dont get too big of a population
        min_size = self.population_size_range[0] * total_destinations
        max_size = self.population_size_range[1] * total_destinations
        population_size = random.randint(min_size, max_size)

        for _ in range(population_size):
            random_path = list(range(1, total_destinations))
            random.shuffle(random_path)
            random_path = [0] + random_path
            random_paths.append(random_path)

        return random_paths

    def _create_children(self, parent_a: list, parent_b: list) -> list:
        """Creates a child out of a pair of parents."""
        #there are different crossover methods, we need to test to see which gives best result
        #Order Crossover (OX)
        #Cycle Crossover (CX)
        #Edge Recombination Crossover (ERX)
        children = []
        start = random.randint(0, len(parent_a) - 1)
        end = random.randint(start, len(parent_a))
        sub_path_a = parent_a[start:end]
        sub_path_b = [item for item in parent_b if item not in sub_path_a]
        for i in range(len(parent_a)):
            if start <= i < end:
                children.append(sub_path_a.pop(0))
            else:
                children.append(sub_path_b.pop(0))
        return children

    def crossovers(self, survivors: list) -> list:
        """Creates crossovers using the _create_children method."""
        #there are different crossover methods, we need to test to see which gives best result
        children = []
        mid = len(survivors) // 2
        for i in range(mid):
            parent_a, parent_b = survivors[i], survivors[i + mid]
            children.append(self._create_children(parent_a, parent_b))
            children.append(self._create_children(parent_b, parent_a)) # pylint: disable=arguments-out-of-order
        return children

    def mutate_population(self, generation: list) -> list:
        """Mutates a small percentage of the population."""
        #maybe should use the mutation percentage based on the size of population
        #test different percentages
        mutated_population = []
        for path in generation:
            if random.random() < self.mutation_rate:
                index1, index2 = random.sample(range(1, len(path)), 2)
                path[index1], path[index2] = path[index2], path[index1]
            mutated_population.append(path)
        return mutated_population

    def gen_new_population(self, curr_gen: list, selection: Selection, fitness_scores: np.ndarray) -> list:
        """Generate a new population using selection, crossover, and mutation."""
        selection_methods = [
                                (selection.stochastic_universal_sampling, 0.3),
                                (selection.roulette_wheel_selection, 0.2)
                            ]
        survivors = np.empty((0, len(curr_gen[0])), dtype=int)
        #survivors = selection.stochastic_universal_sampling(curr_gen, fitness_scores)
        for method, percentage in selection_methods:
            selected = method(curr_gen, fitness_scores, percentage)
            survivors = np.concatenate((survivors, selected))

        survivors = survivors.tolist()
        children = self.crossovers(survivors)
        combined_population = survivors + children
        new_population = self.mutate_population(combined_population)
        return new_population

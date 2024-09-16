"""
Module containing population-based search
"""
import random
import numpy as np

class Population:
    """Class for generating and modifying the population"""
    def inital_population(self, graph: np.ndarray):
        """Maybe swap graph to just be total destinations"""
        total_destinations = graph.shape[0]
        random_paths = []
        min_size = 10 * total_destinations
        max_size = 50 * total_destinations
        #maybe just have max_size, but I dont think it should be hardcoded
        population_size = random.randint(min_size, max_size)

        for _ in range (population_size):
            random_path = list(range(1, total_destinations))
            random.shuffle(random_path)
            random_path = [0] + random_path
            random_paths.append(random_path)
        return random_paths


    def _create_children(self, parent_a: list, parent_b: list) -> list:
        children = []
        start = random.randint(0, len(parent_a) - 1)
        end = random.randint(start, len(parent_a))
        sub_path_a = parent_a[start:end]
        sub_path_b = list([item for item in parent_b if item not in sub_path_a])
        for i in range(len(parent_a)):
            if start <= i < end:
                children.append(sub_path_a.pop(0))
            else:
                children.append(sub_path_b.pop(0))
        return children

    def crossovers(self, survivors: list):
        children = []
        mid = len(survivors) // 2
        for i in range(mid):
            parent_a, parent_b = survivors[i], survivors[i + mid]
            children.append(self._create_children(parent_a, parent_b))
            children.append(self._create_children(parent_b, parent_a))
        return children

    def mutate_population(self, generation: list) : 
        mutated_population = []
        for path in generation:
            if random.randint(0, 1000) < 9: #apply on less than 1%
                index1, index2 = random.randint(1, len(path) - 1), random.randint(1, len(path) -1)
                path[index1], path[index2], = path[index2], path[index1]
            mutated_population.append(path)

    def generate_new_population():

"""
    populate(): Initializes a population of potential solutions.
    
    selection(): Selects the best solutions to pass on their genes to the next generation.
    crossover(): Combines solutions from the population to generate new offspring.
    mutation(): Introduces random changes to a solution to maintain diversity in the population.
    fitness(): Evaluates the quality of a solution based on the route distance.
    
"""
"""
Module containing selection, solutions are selected based on their fitness to produce offspring
"""

import random
import numpy as np
from evolutionary_classes.fitness_function import FitnessFunction


class Selection:
    """A class to choose the survivors of a generation"""

    def __init__(self):
        """
        Initialize Selection class.
        """
        # Add attributes to __init__, such as selection style if needed

    def distance(self, graph: np.ndarray, path: list) -> int:
        """Calculate the total distance of the given path in the graph."""
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += graph[path[i]][path[i + 1]]
        total_distance += graph[path[-1]][path[0]]

        return total_distance

    def survivors(self, curr_generation: list, ff: FitnessFunction) -> list:
        """
        Select half of the population based on the fitness function.

        This method shuffles the current generation, splits it into two halves,
        and selects the fitter individual from each pair to survive to the next generation.

        Args:
            curr_generation (list): The current population.
            ff (FitnessFunction): The fitness function to evaluate individuals.

        Returns:
            list: The list of selected survivors.
        """
        # Implement the fitness function here instead
        survivors = []
        random.shuffle(curr_generation)
        mid = len(curr_generation) // 2

        for i in range(mid):
            fit_1 = ff.fitness_function_normalized(curr_generation[i])
            fit_2 = ff.fitness_function_normalized(curr_generation[i + mid])
            if fit_1 > fit_2:
                survivors.append(curr_generation[i])
                # print(f"Choosing the better fit: {fit_1}")
            else:
                survivors.append(curr_generation[i + mid])
                # print(f"Choosing the better fit: {fit_2}")

        return survivors

    def elitism(
        self,
        curr_generation: list,
        ff: FitnessFunction,
        survive_rate: float = 0.5
    ) -> list:
        """
        Selects survivors based on elitism, i.e., the top individuals.

        This method sorts the current generation in descending order of fitness and selects
        the top portion as survivors based on the specified survival rate.

        Args:
            curr_generation (list): The current population.
            ff (FitnessFunction): The fitness function to evaluate individuals.
            survive_rate (float, optional): 
            The proportion of the population to survive. Defaults to 0.5.

        Returns:
            list: The list of selected survivors.
        """
        # Sort in descending order
        sorted_population = sorted(
            curr_generation,
            key=ff.fitness_function_normalized,
            reverse=True
        )
        num_survivors = int(len(curr_generation) * survive_rate)  # Calculate number of survivors
        survivors = sorted_population[:num_survivors]  # Select top survivors
        return survivors

    def tournament(
        self,
        generation: np.ndarray,
        fitness: np.ndarray,
        survive_rate: float,
        tournament_size: int
    ) -> np.ndarray:
        """
        Selects survivors using tournament selection.

        For each survivor to be selected, a subset of the population is chosen randomly,
        and the individual with the best fitness in the subset is selected as a survivor.

        Args:
            generation (np.ndarray): The current population.
            fitness (np.ndarray): The fitness scores corresponding to the population.
            survive_rate (float): The proportion of the population to survive.
            tournament_size (int): The number of individuals competing in each tournament.

        Returns:
            np.ndarray: The array of selected survivors.
        """
        population_size = len(generation)
        n_survive = max(1, int(population_size * survive_rate))

        survivors = []
        for _ in range(n_survive):
            # Randomly select individuals for the tournament
            tournament_indices = random.sample(range(population_size), tournament_size)

            # Retrieve their fitness scores
            tournament_fitness = fitness[tournament_indices]

            # Find the index of the individual with the best fitness (lowest score)
            winner_idx_in_tournament = np.argmin(tournament_fitness)
            winner_index = tournament_indices[winner_idx_in_tournament]

            # Append the winner to the survivors list
            survivors.append(generation[winner_index])

        return np.array(survivors)

    # --- Optional Selection Methods ---
    # Currently, we compare two solutions and pick the better one to be a survivor.
    # Implement additional selection strategies as needed:
    # - Elitist selection: choose the top 15% best solutions in the current generation
    # - Tournament selection: choose 25% of the remaining population through tournaments
    # - Roulette selection: choose 10% randomly to maintain diversity
    # For example, with a population of 100:
    # - Elitist: 15 best solutions
    # - Tournament: 25 selected via tournaments
    # - Roulette: 10 chosen randomly from the remaining 60 solutions

    # Consider experimenting with these methods to determine which yields the best solutions.
    # Also, decide if you want to mutate a small portion of the population to maintain diversity.

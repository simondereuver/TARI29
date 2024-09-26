"""Module containing selection"""
# pylint: disable=too-many-arguments,line-too-long
import math
import random
import numpy as np


class Selection:
    """A class to choose the survivors of a generation"""

    def __init__(self):
        """Initialize Selection class"""

    def survivors(self, old_generation: list, fitness_scores: np.ndarray) -> list:
        """Select half of the gen based on fitness function."""
        #implement the fitness function here instead
        survivors = []
        mid = len(old_generation) // 2

        for i in range(mid):
            if fitness_scores[i] > fitness_scores[i + mid]:
                survivors.append(old_generation[i])
            else:
                survivors.append(old_generation[i + mid])

        return survivors

    def elitism(self, generation: np.ndarray, fitness: np.ndarray, survive_rate: float = 0.5) -> np.ndarray:
        """Selects survivors based on elitism, ie the top"""
        #sort in descending order
        sorted_indices = np.argsort(fitness)[::-1]
        # cap the num_survivors to min==1 and max=int(math.floor(survive_rate * len(generation)))
        num_survivors = max(1, int(math.floor(survive_rate * len(generation))))
        num_survivors = min(num_survivors, len(generation))

        elite_indices = sorted_indices[:num_survivors]
        elites = generation[elite_indices]

        return elites

    def tournament(
        self,
        generation: np.ndarray,
        fitness: np.ndarray,
        survive_rate: float,
        tournament_size: int = 2
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
    #currently we compare two solutions, and pick the better one, to be a survivor
    #   Implement something like this instead of current survivors logic method:
    # elitist_selection, choose the 15% best solutions in the current generation,
    # tournament_selection, choose 25% of the remaining gen through tournament
    # choose 10% in roulette
    # lets say we have a gen of 100
    # elitist = 15 best solutions, tournament = 25, out of the remaining 60 solutions,
    # choose 10 at roulette(random) for maintaining diversity

    #we should try to look into this and see what gives us the best solution rate
    #do we also want to mutate a small portion of the gen?

    def roulette_wheel_selection(self, generation: np.ndarray, fitness: np.ndarray, survive_rate: float) -> np.ndarray:
        """Perform Roulette Wheel Selection"""

        total_fitness = fitness.sum()
        size_gen = len(generation)

        if total_fitness == 0:
            # If total fitness is zero, select uniformly at random
            probabilities = np.full(size_gen, 1 / size_gen)
        else:
            probabilities = fitness / total_fitness
            probabilities /= probabilities.sum()

        num_selected = max(1, int(math.floor(survive_rate * size_gen)))
        num_selected = min(num_selected, size_gen)

        selected_indices = np.random.choice(size_gen, size=num_selected, replace=True, p=probabilities)
        selected = generation[selected_indices]

        return selected

    def stochastic_universal_sampling(self, generation: np.ndarray, fitness: np.ndarray, survive_rate: float):
        """Stochastic universal sampling implementation"""
        total_fitness = fitness.sum()
        size_gen = len(generation)
        if total_fitness == 0:
            probabilities = np.full(size_gen, 1 / size_gen)
        else:
            probabilities = fitness / total_fitness

        cumulative_prob = np.cumsum(probabilities)
        cumulative_prob[-1] = 1.0

        num_selected = max(1, int(math.floor(survive_rate * size_gen)))
        num_selected = min(num_selected, size_gen)

        pointer_distance = 1.0 / num_selected
        start_point = np.random.uniform(0, pointer_distance)
        pointers = start_point + pointer_distance * np.arange(num_selected)

        indices = np.clip(np.searchsorted(cumulative_prob, pointers), 0, size_gen - 1)
        selected = generation[indices]

        return selected

    def rank_selection(self, generation: np.ndarray, fitness: np.ndarray, survive_rate: float) -> np.ndarray:
        """Perform Rank Selection"""
        size_gen = len(generation)

        num_selected = max(1, int(math.floor(survive_rate * size_gen)))
        num_selected = min(num_selected, size_gen)

        sorted_indices = np.argsort(-fitness)

        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(fitness) + 1)
        rank_sum = np.sum(ranks)

        selection_probs = ranks / rank_sum
        cumulative_probs = np.cumsum(selection_probs)

        random_values = np.random.rand(num_selected)

        selected_indices = np.searchsorted(cumulative_probs, random_values)
        selected = generation[selected_indices]

        return selected

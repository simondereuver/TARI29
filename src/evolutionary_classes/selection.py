"""Module containing selection"""
# pylint: disable=too-many-arguments,line-too-long
import math
import numpy as np


class Selection:
    """A class to choose the survivors of a generation"""

    def __init__(self,
                 selection_methods=None,
                 survive_rate=0.5,
                 tournament_size=2):
        """
        Initialize Selection class with a list of selection methods and their percentages.

        Args:
            selection_methods (list of tuples): (selection_method, percentage).
            survive_rate (float): Percentage of population that survives.
            tournament_size (int): Number of individuals in each tournament for tournament selection.
        """
        if selection_methods is None:
            selection_methods = [("elitism", 1.0)]  #default

        self.selection_methods = selection_methods
        self.survive_rate = survive_rate
        self.tournament_size = tournament_size
        self.selection_functions = {
            "elitism": self.elitism,
            "tournament": self.tournament,
            "roulette_wheel": self.roulette_wheel_selection,
            "rank_selection": self.rank_selection,
        }

    def select_survivors(self,
                         generation: np.ndarray,
                         fitness: np.ndarray
                         ) -> np.ndarray:
        """
        Select survivors using multiple selection methods and their percentages.

        Args:
            generation (np.ndarray): The current population.
            fitness (np.ndarray): The fitness scores corresponding to the population.

        Returns:
            np.ndarray: The selected survivors.
        """
        total_population_size = len(generation)
        total_survivors = max(1, int(total_population_size * self.survive_rate))
        survivors = np.empty((0, generation.shape[1]), dtype=generation.dtype)
        total_selected = 0

        for method, percentage in self.selection_methods: #remember we are unpacking a tuple with (method, percentage)
            num_to_select = int(total_survivors * percentage) #amout to select for a given method
            #print(f"Using {method} to select {num_to_select} survivors")
            selected = self.selection_functions[method](generation, fitness, num_to_select)
            survivors = np.vstack((survivors, selected))
            total_selected += num_to_select

        #print(f"Survivors selected: {len(survivors)} from multiple methods")
        return survivors

    def elitism(self,
                generation: np.ndarray,
                fitness: np.ndarray,
                num_to_select: int
                ) -> np.ndarray:
        """Selects the top individuals based on fitness (elitism)."""
        sorted_indices = np.argsort(fitness)[::-1]
        elites = generation[sorted_indices[:num_to_select]]
        return elites

    def tournament(self,
                   generation: np.ndarray,
                   fitness: np.ndarray,
                   num_to_select: int,
                   tournament_size: int = None
                   ) -> np.ndarray:
        """
        Selects survivors using tournament selection.
        """
        if tournament_size is None:
            tournament_size = self.tournament_size

        population_size = len(generation)
        survivors = []
        for _ in range(num_to_select):
            # Randomly select individuals for the tournament
            tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
            tournament_fitness = fitness[tournament_indices]

            # Find the index of the individual with the best fitness (highest score)
            winner_idx_in_tournament = np.argmax(tournament_fitness)
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

    def roulette_wheel_selection(self,
                                 generation: np.ndarray,
                                 fitness: np.ndarray,
                                 num_to_select: int
                                 ) -> np.ndarray:
        """Perform Roulette Wheel Selection."""
        total_fitness = fitness.sum()
        size_gen = len(generation)

        if total_fitness == 0:
            probabilities = np.full(size_gen, 1 / size_gen)
        else:
            probabilities = fitness / total_fitness
            probabilities /= probabilities.sum()  # Normalize

        selected_indices = np.random.choice(size_gen, size=num_to_select, replace=True, p=probabilities)
        selected = generation[selected_indices]
        return selected


    def stochastic_universal_sampling(self,
                                      generation: np.ndarray,
                                      fitness: np.ndarray,
                                      survive_rate: float = 0.5):
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
    """
    def rank_selection(self, generation: np.ndarray, fitness: np.ndarray, survive_rate: float = 0.5) -> np.ndarray:
        #Perform Rank Selection
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
    """
    def rank_selection(self, generation: np.ndarray, fitness: np.ndarray, num_to_select: int) -> np.ndarray:
        """Perform Rank Selection."""
        size_gen = len(generation)
        sorted_indices = np.argsort(fitness)

        # Assign ranks
        ranks = np.arange(1, size_gen + 1)
        probabilities = ranks / ranks.sum()

        # Since ranks are sorted in ascending fitness, reverse the indices
        selection_probs = probabilities[::-1]

        selected_indices = np.random.choice(size_gen, size=num_to_select, replace=True, p=selection_probs)
        selected = generation[selected_indices]
        return selected

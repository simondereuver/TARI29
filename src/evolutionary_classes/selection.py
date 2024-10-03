"""Module containing selection"""
# pylint: disable=too-many-arguments,line-too-long
import math
import numpy as np


class Selection:
    """A class to choose the survivors of a generation"""

    def __init__(self,
                 selection_methods: list[tuple[str, float]] = None,
                 survive_rate: float = 0.5,
                 tournament_size: int = 2):
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
                         fitness: np.ndarray) -> np.ndarray:
        """
        Select survivors using the selected selection method(s) and their percentages.

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

        for method, percentage in self.selection_methods: #remember we are unpacking a list of tuples (method, percentage)
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
                num_to_select: int) -> np.ndarray:
        """
        Selects the top individuals based on fitness (elitism).
        
        Args:
            generation (np.ndarray): The current population.
            fitness (np.ndarray): The fitness scores corresponding to the population.
            num_to_select: (int): The amount of survivors to select.
        Returns:
            np.ndarray: The selected survivors (elites).
        """
        #sort fitness in descending order and get sorted indices, then select survivors
        sorted_indices = np.argsort(fitness)[::-1]
        elites = generation[sorted_indices[:num_to_select]]
        return elites

    def tournament(self,
                   generation: np.ndarray,
                   fitness: np.ndarray,
                   num_to_select: int,
                   tournament_size: int = None) -> np.ndarray:
        """
        Selects survivors using tournament selection.

        Args:
            generation (np.ndarray): The current population.
            fitness (np.ndarray): The fitness scores corresponding to the population.
            num_to_select: (int): The amount of survivors to select.
            tournament_size: (int): The amount of individuals to be used in the tournament.
        Returns:
            np.ndarray: The selected survivors.
        """
        if tournament_size is None:
            tournament_size = self.tournament_size

        population_size = len(generation)
        survivors = []
        for _ in range(num_to_select):
            #randomly choose individuals for a tournament
            tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
            tournament_fitness = fitness[tournament_indices]

            #find the index of the individual with the best fitness (highest score)
            winner_idx_in_tournament = np.argmax(tournament_fitness)
            winner_index = tournament_indices[winner_idx_in_tournament]

            survivors.append(generation[winner_index])

        return np.array(survivors)


    def roulette_wheel_selection(self,
                                 generation: np.ndarray,
                                 fitness: np.ndarray,
                                 num_to_select: int) -> np.ndarray:
        """
        Perform Roulette Wheel Selection.
        
        Args:
            generation (np.ndarray): The current population.
            fitness (np.ndarray): The fitness scores corresponding to the population.
            num_to_select: (int): The amount of survivors to select.
            tournament_size: (int): The amount of individuals to be used in the tournament.
        Returns:
            np.ndarray: The selected survivors.
        """
        total_fitness = fitness.sum()
        size_gen = len(generation)
        #calculate selection probabilities based on fitness
        #if total fitness is 0, assign equal probability to all individuals
        if total_fitness == 0:
            probabilities = np.full(size_gen, 1 / size_gen)
        else:
            #normalize fitness values to probabilities
            probabilities = fitness / total_fitness
            probabilities /= probabilities.sum()
        #choose survivors randomly, based on the probabilities
        selected_indices = np.random.choice(size_gen, size=num_to_select, replace=True, p=probabilities)
        selected = generation[selected_indices]
        return selected

    def rank_selection(self,
                       generation: np.ndarray,
                       fitness: np.ndarray,
                       num_to_select: int) -> np.ndarray:
        """
        Perform Rank Selection.
        
        Args:
            generation (np.ndarray): The current population.
            fitness (np.ndarray): The fitness scores corresponding to the population.
            num_to_select: (int): The amount of survivors to select.

        Returns:
            np.ndarray: The selected survivors.
        """
        size_gen = len(generation)
        #sort fitness and get sorted indices
        sorted_indices = np.argsort(fitness)

        #assign ranks to individuals based on sorted fitness 
        #with the best (lowest distances) having the highest rank (size_gen)
        ranks = np.arange(1, size_gen + 1)

        #calculate selection probabilities based on rank
        selection_probs = ranks / ranks.sum()
        selection_probs = selection_probs[::-1]

        #choose survivors randomly, based on the probabilities
        selected_indices = np.random.choice(size_gen, size=num_to_select, replace=True, p=selection_probs)
        selected = generation[sorted_indices[selected_indices]]
        return selected

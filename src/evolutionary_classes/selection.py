# selection.py
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
        #add something to __init__, maybe which selection style to be used or something

    def distance(self, graph: np.ndarray, path: list) -> int:
        """Calculate the total distance of the given path in the graph."""
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += graph[path[i]][path[i + 1]]
        total_distance += graph[path[-1]][path[0]]

        return total_distance

    def survivors(self, old_generation: list, fitness_scores: np.ndarray) -> list:
        """Select half of the gen based on fitness function."""
        #implement the fitness function here instead
        survivors = []
        mid = len(old_generation) // 2

        for i in range(mid):
            if fitness_scores[i] > fitness_scores[i + mid]:
                survivors.append(old_generation[i])
                #print(f"Choosing the better fit: {fit_1}")
            else:
                survivors.append(curr_generation[i + mid])
                #print(f"Choosing the better fit: {fit_2}")

        return survivors

    def elitism(self,
                curr_generation: list,
                ff: FitnessFunction,
                survive_rate: float = 0.5) -> list:
        "Selects survivors based on elitism, ie the top"
        #sort in descending order
        sorted_population = sorted(curr_generation,
                                   key=ff.fitness_function_normalized,
                                   reverse=True)
        num_survivors = int(len(curr_generation) * survive_rate) #calc amount of survivors
        survivors = sorted_population[:num_survivors] #cut off the rest from the sorted_popilation
        return survivors

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

    def roulette_wheel_selection(self, generation: list, fitness, percentage_of_gen: float):
        """Perform Roulette Wheel Selection"""
        gen = np.array(generation)
        total_fitness = fitness.sum()

        if total_fitness == 0:
            # If total fitness is zero, select uniformly at random
            probabilities = np.full(len(fitness), 1 / len(fitness))
        else:
            probabilities = fitness / total_fitness

        num_selected = int(percentage_of_gen * len(gen))

        selected_indices = np.random.choice(len(gen), size=num_selected, replace=True, p=probabilities)

        selected = gen[selected_indices]

        return selected

    def stochastic_universal_sampling(self, generation, fitness: np.ndarray, percentage_of_gen: float):
        """Stochastic universal sampling implementation"""
        gen = np.array(generation)
        total_fitness = fitness.sum()

        if total_fitness == 0:
            # If total fitness is zero, select uniformly at random
            probabilities = np.full(len(fitness), 1 / len(fitness))
        else:
            probabilities = fitness / total_fitness

        # Compute cumulative probabilities
        cumulative_prob = np.cumsum(probabilities)
        
        cumulative_prob[-1] = 1.0

        # Set the number of pointers equal to the size of the generation
        num_selected = int(len(gen) * percentage_of_gen)

        # Distance between pointers
        pointer_distance = 1.0 / num_selected

        # Start point: a random number between 0 and pointer_distance
        start_point = np.random.uniform(0, pointer_distance)

        # Pointers
        pointers = start_point + pointer_distance * np.arange(num_selected)

        # Find the indices for each pointer
        indices = np.clip(np.searchsorted(cumulative_prob, pointers), 0, len(gen) - 1)

        # Select the individuals
        selected = gen[indices]

        return selected

    def rank_selection(self, population, fitness):
        """
        Perform Rank Selection.

        Parameters:
        - population (np.ndarray or list of lists): Population of paths.
        - fitness (np.ndarray): Array of fitness scores.

        Returns:
        - selected (np.ndarray): Array of selected individuals.
        """
        gen = np.array(population)
        gen = np.asarray(gen)
        fitness = np.asarray(fitness)

        # Ensure population and fitness have compatible shapes
        if gen.shape[0] != fitness.shape[0]:
            raise ValueError("Population and fitness must have the same number of individuals.")

        # Number of individuals to select (assuming selection size equals population size)
        num_selected = gen.shape[0] / 2

        # Get the sorted indices (ascending order since lower fitness might be better)
        # If higher fitness is better, use descending order by negating fitness
        sorted_indices = np.argsort(-fitness)
        
        # Assign ranks: highest fitness gets rank 1
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(fitness) + 1)
        
        # Compute selection probabilities based on rank
        # Here, using linear rank selection where probability is proportional to rank
        rank_sum = np.sum(ranks)
        selection_probs = ranks / rank_sum

        # Compute cumulative probabilities for efficient sampling
        cumulative_probs = np.cumsum(selection_probs)

        # Generate random numbers for selection
        random_values = np.random.rand(num_selected)

        # Find indices where random values would fit in the cumulative distribution
        selected_indices = np.searchsorted(cumulative_probs, random_values)

        # Select individuals based on selected indices
        selected = gen[selected_indices]

        return selected

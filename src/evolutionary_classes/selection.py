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

    def survivors(self, curr_generation: list, ff: FitnessFunction) -> list:
        """Select half of the population based on fitness function."""
        #implement the fitness function here instead
        survivors = []
        random.shuffle(curr_generation)
        mid = len(curr_generation) // 2

        for i in range(mid):
            fit_1 = ff.fitness_function_normalized(curr_generation[i])
            fit_2 = ff.fitness_function_normalized(curr_generation[i + mid])
            if fit_1 > fit_2:
                survivors.append(curr_generation[i])
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
    
    def tournament(self, generation: np.ndarray, fitness: np.ndarray, survive_rate: float, tournament_size: int) -> np.ndarray:

        population_size = len(generation)
        n_survive = max(1, int(population_size * survive_rate))  # Ensure at least one individual survives

        survivors = []
        available_indices = set(range(population_size))  # Indices of individuals not yet selected

        for _ in range(n_survive):
            if len(available_indices) < tournament_size:
                # If not enough individuals remain for a full tournament, adjust the tournament size
                current_tournament_size = len(available_indices)
            else:
                current_tournament_size = tournament_size

            # Randomly select individuals for the tournament from available_indices
            tournament_indices = random.sample(available_indices, current_tournament_size)

            # Retrieve their fitness scores
            tournament_fitness = fitness[tournament_indices]

            # Find the index of the individual with the best fitness (lowest score)
            winner_idx_in_tournament = np.argmin(tournament_fitness)
            winner_index = tournament_indices[winner_idx_in_tournament]

            # Append the winner to the survivors list
            survivors.append(generation[winner_index])

            # Remove the winner from available_indices to prevent multiple selections
            available_indices.remove(winner_index)

        return np.array(survivors)
    


    # --- Optional Selection Methods ---
    #currently we compare two solutions, and pick the better one, to be a survivor
    #   Implement something like this instead of current survivors logic method:
    # elitist_selection, choose the 15% best solutions in the current generation,
    # tournament_selection, choose 25% of the remaining population through tournament
    # choose 10% in roulette
    # lets say we have a population of 100
    # elitist = 15 best solutions, tournament = 25, out of the remaining 60 solutions,
    # choose 10 at roulette(random) for maintaining diversity

    #we should try to look into this and see what gives us the best solution rate
    #do we also want to mutate a small portion of the population?

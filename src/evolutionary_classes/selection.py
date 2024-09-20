# selection.py
"""
Module containing selection, solutions are selected based on their fitness to produce offspring
"""
import random
import numpy as np
from evolutionary_classes.fitness_function import FitnessFunction
import matplotlib.pyplot as plt
from scipy.stats import norm


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

    #def survivors(self, graph: np.ndarray, old_generation: list) -> list:
    #    """Select half of the population based on minimum distances."""
    #    #implement the fitness function here instead
    #    survivors = []
    #    random.shuffle(old_generation)
    #    mid = len(old_generation) // 2
#
    #    for i in range(mid):
    #        distance_1 = self.distance(graph, old_generation[i])
    #        distance_2 = self.distance(graph, old_generation[i + mid])
    #        if distance_1 < distance_2:
    #            survivors.append(old_generation[i])
    #        else:
    #            survivors.append(old_generation[i + mid])
#
    #    return survivors

    def survivors(self, old_generation: list, ff: FitnessFunction, generation: int) -> list:
        """Select half of the population based on fitness function."""
        survivors = []
        random.shuffle(old_generation)
        mid = len(old_generation) // 2

        for i in range(mid):
            fit_1 = ff.fitness_function_normalized(old_generation[i])
            fit_2 = ff.fitness_function_normalized(old_generation[i + mid])
            if fit_1 > fit_2:
                survivors.append(old_generation[i])
                #print(f"Generation {generation}: Choosing the better fit: {fit_1}")
            else:
                survivors.append(old_generation[i + mid])
                #print(f"Generation {generation}: Choosing the better fit: {fit_2}")

        return survivors
    
    def tournament_selection(self, population: list, ff: FitnessFunction, tournament_size: int) -> list:

        survivors = []
        population_size = len(population)
        for _ in range(population_size):
            # Randomly select individuals for the tournament
            tournament = random.sample(population, tournament_size)
            # Evaluate fitness for each individual in the tournament
            fitnesses = [(individual, ff.evaluate(individual)) for individual in tournament]
            # Select the individual with the best fitness (lowest distance)
            winner = min(fitnesses, key=lambda x: x[1])[0]
            survivors.append(winner)
        return survivors

    # --- Optional Selection Methods ---

    #   Implement something like this instead of current survivors logic method:
    # def elitist_selection() -> list:
    # def tournament_selection() -> list:


    def calculate_clustering(self, graph: np.ndarray) -> float:
        
        num_nodes = graph.shape[0]
        # For each node, find the distance to its nearest neighbor
        nearest_neighbor_distances = []
        for i in range(num_nodes):
            # Exclude (distance to self)
            distances = np.delete(graph[i], i)
            nearest_neighbor_distance = np.min(distances)
            nearest_neighbor_distances.append(nearest_neighbor_distance)

        mean_nn_distance = np.mean(nearest_neighbor_distances)
        # Expected mean nearest neighbor distance for a random distribution
        # Since we don't have coordinates, we can use the mean of all edge weights as an approximation
        all_distances = graph[np.triu_indices(num_nodes, k=1)]
        expected_distance = np.mean(all_distances)
        # Clustering coefficient is the ratio of observed to expected mean nearest neighbor distance
        clustering_coefficient = mean_nn_distance / expected_distance
        return clustering_coefficient
    
    
    
    def plot_tour_length_distribution(self, graph: np.ndarray, num_samples=10000):
        
        num_nodes = graph.shape[0]
        tour_lengths = []

        for _ in range(num_samples):
            tour = list(range(num_nodes))
            random.shuffle(tour)
            total_length = self.distance(graph, tour)
            tour_lengths.append(total_length)

        tour_lengths = np.array(tour_lengths)
        mean_length = np.mean(tour_lengths)
        std_length = np.std(tour_lengths)

        # Plot the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(tour_lengths, bins=50, density=True, alpha=0.6, color='g')

        # Plot the normal distribution curve
        x = np.linspace(tour_lengths.min(), tour_lengths.max(), 1000)
        p = norm.pdf(x, mean_length, std_length)
        plt.plot(x, p, 'k', linewidth=2)

        plt.title('Distribution of Random Tour Lengths')
        plt.xlabel('Total Tour Length')
        plt.ylabel('Density')
        plt.show()

        print(f"Mean Tour Length: {mean_length:.2f}")
        print(f"Standard Deviation: {std_length:.2f}")

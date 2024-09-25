# genetic_algorithm_solver.py
"""
Module using the needed classes to compute the TSP using the genetic algorithm
"""
import numpy as np
from tqdm import tqdm
from evolutionary_classes.fitness import compute_fitness_scores, calculate_distance
from evolutionary_classes.selection import Selection
from evolutionary_classes.fitness_function import FitnessFunction
from evolutionary_classes.population import Population


class TSPGeneticSolver:
    """Combines the other classes to genetically solve TSP"""

    # pylint: disable=too-many-arguments,line-too-long,too-many-positional-arguments
    def __init__(self, graph: np.ndarray, population_size_range=(10, 50), mutation_rate=0.01, bounds=None, crossover_method: str = "Simple"):
        """
        Initialize the GeneticAlgorithmSolver.
        """
        # pylint: disable=too-few-public-methods
        self.graph = graph

        #self.population_manager = Population(mutation_rate, population_size_range, crossover_method)
        if crossover_method == "SCX":
            self.population_manager = Population(mutation_rate, population_size_range, crossover_method, graph)
        else:
            self.population_manager = Population(mutation_rate, population_size_range, crossover_method)

        self.selection_manager = Selection()
        self.ff = FitnessFunction(graph, bounds)

    def get_fitness(self, path):
        """Uses the fitness function to return the fitness value"""
        return self.ff.fitness_function_distance_based(path)

    def run(self, generations=100, population_size=100):
        """
        Run the genetic algorithm for a specified number of generations.
        """
        progress_bar = tqdm(total=generations)

        current_generation = self.population_manager.initial_population(self.graph, population_size)

        fitness_scores = compute_fitness_scores(self.graph, current_generation, self.ff.bounds)
        index = np.argmax(fitness_scores)
        best_path = current_generation[index]

        for _ in range(generations):
            fitness_scores = compute_fitness_scores(self.graph, current_generation, self.ff.bounds)

            current_generation = self.population_manager.gen_new_population(
                current_generation,
                self.selection_manager,
                fitness_scores)

            index = np.argmax(fitness_scores)
            if calculate_distance(self.graph, best_path) > calculate_distance(self.graph, current_generation[index]):
                best_path = current_generation[index]

            progress_bar.update(1)
            progress_bar.set_postfix_str(f'fitness={np.max(fitness_scores):.4f}')

        if not np.any(current_generation):
            print("\nNo valid paths found in the final generation.")
            return None, None

        best_path = np.append(best_path, best_path[0])
        best_distance = calculate_distance(self.graph, best_path)

        return best_path, best_distance

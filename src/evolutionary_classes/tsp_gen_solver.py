# genetic_algorithm_solver.py
"""
Module using the needed classes to compute the TSP using the genetic algorithm
"""
import numpy as np
from tqdm import tqdm
from evolutionary_classes.fitness import compute_fitness_scores
from evolutionary_classes.selection import Selection
from evolutionary_classes.fitness_function import FitnessFunction
from evolutionary_classes.population import Population


class TSPGeneticSolver:
    """Combines the other classes to genetically solve TSP"""

    # pylint: disable=too-many-arguments,line-too-long,too-many-positional-arguments
    def __init__(self, graph: np.ndarray, population_size_range=(10, 50),mutation_rate=0.01,bounds=None,crossover_method: str = "Simple"):
        """
        Initialize the GeneticAlgorithmSolver.
        """
        # pylint: disable=too-few-public-methods
        self.graph = graph
        self.population_manager = Population(mutation_rate, population_size_range, crossover_method)
        self.selection_manager = Selection()
        self.ff = FitnessFunction(graph, bounds)

    def get_fitness(self, path):
        """Uses the fitness function to return the fitness value"""
        return self.ff.fitness_function_distance_based(path)

    def run(self, generations=100):
        """
        Run the genetic algorithm for a specified number of generations.
        """
        current_generation = self.population_manager.initial_population(self.graph)

        for generation in tqdm(range(generations), desc="Generations"):
            fitness_scores = compute_fitness_scores(self.graph, current_generation, self.ff.bounds)
            current_generation = self.population_manager.gen_new_population(
                current_generation,
                self.selection_manager,
                fitness_scores)

        if not current_generation:
            print("\nNo valid paths found in the final generation.")
            return None, None

        best_path = min(current_generation, key=self.get_fitness)
        best_distance = self.ff.fitness_function_distance_based(best_path)
        #quick fix to add 0 at end to make a "round" path
        best_path.append(best_path[0])

        return best_path, best_distance

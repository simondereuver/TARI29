"""Module using the needed classes to compute the TSP using the genetic algorithm"""
# pylint: disable=too-many-arguments,line-too-long
import numpy as np
from tqdm import tqdm
from evolutionary_classes.fitness import compute_fitness_scores, calculate_distance
from evolutionary_classes.selection import Selection
from evolutionary_classes.fitness_function import FitnessFunction
from evolutionary_classes.population import Population


class TSPGeneticSolver:
    """Combines the other classes to genetically solve TSP"""

    def __init__(self,
                 graph: np.ndarray,
                 mutation_rate=0.01,
                 bounds=None,
                 crossover_method: str = "Simple",
                 selection_methods = None,
                 survive_rate: float = 0.5,
                 tournament_size: int = None):
        """
        Initialize the GeneticAlgorithmSolver.
        """
        self.graph = graph

        if crossover_method == "SCX":
            self.population_manager = Population(mutation_rate, crossover_method, graph)
        else:
            self.population_manager = Population(mutation_rate, crossover_method)

        self.selection_manager = Selection(selection_methods=selection_methods,
                                           survive_rate=survive_rate,
                                           tournament_size=tournament_size)
        self.bounds = bounds
        self.ff = FitnessFunction(graph, bounds)

    def run(self, generations: int =100, population_size: int =100):
        """
        Run the genetic algorithm for a specified number of generations.
        """
        progress_bar = tqdm(total=generations, disable=False)

        current_generation = self.population_manager.initial_population(self.graph, population_size)

        fitness_scores = compute_fitness_scores(self.graph, current_generation, self.bounds)
        index = np.argmax(fitness_scores)
        best_path = current_generation[index]

        best_distance = calculate_distance(self.graph, best_path)

        #track convergence
        no_improvement_count = 0
        max_no_improvement_count = 0
        convergence_generation_start = None
        improved_at_least_once = False

        for generation in range(generations):
            current_generation = self.population_manager.gen_new_population(
                current_generation,
                self.selection_manager,
                fitness_scores)

            fitness_scores = compute_fitness_scores(self.graph, current_generation, self.bounds)

            index = np.argmax(fitness_scores)
            if calculate_distance(self.graph, best_path) > calculate_distance(self.graph, current_generation[index]):
                best_path = current_generation[index]

            current_best_distance = calculate_distance(self.graph, current_generation[index])

            #look for improvements
            if current_best_distance < best_distance:
                best_distance = current_best_distance
                best_path = current_generation[index]
                no_improvement_count = 0
                convergence_generation_start = None
                improved_at_least_once = True  # mark that an improvement has occurred
            else:
                no_improvement_count += 1

            #update the maximum convergence period if necessary
            if no_improvement_count > max_no_improvement_count:
                max_no_improvement_count = no_improvement_count
                convergence_generation_start = generation - no_improvement_count + 1

            progress_bar.set_postfix_str(f'fitness={np.max(fitness_scores):.3f}, Best={calculate_distance(self.graph, best_path)}')
            progress_bar.update(1)

        if not np.any(current_generation):
            print("\nNo valid paths found in the final generation.")
            return None, None

        if not improved_at_least_once:
            convergence_generation_start = 0
            max_no_improvement_count = generations

        best_path = np.append(best_path, best_path[0])

        return best_path, best_distance, convergence_generation_start, max_no_improvement_count

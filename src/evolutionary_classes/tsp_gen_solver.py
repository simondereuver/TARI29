# genetic_algorithm_solver.py
"""
Module using the needed classes to compute the TSP using the genetic algorithm
"""
import time
import numpy as np
from evolutionary_classes.selection import Selection
from evolutionary_classes.fitness_function import FitnessFunction
from evolutionary_classes.population import Population

class TSPGeneticSolver:
    """Combines the other classes to genetically solve TSP"""

    def __init__(self, graph: np.ndarray, population_size_range=(10, 50),
                 mutation_rate=0.01,
                 bounds=None):
        """
        Initialize the GeneticAlgorithmSolver.
        """
        # pylint: disable=too-few-public-methods
        self.graph = graph
        self.population_manager = Population(mutation_rate, population_size_range)
        self.selection_manager = Selection()
        self.fitness_function = FitnessFunction(graph, bounds)

    def get_fitness(self, path):
        """Uses the fitness function to return the fitness value"""
        return self.fitness_function.fitness_function_distance_based(path)
    

    def run(self, generations=100, poptype="random", patience=5):
        """
        Run the genetic algorithm for a specified number of generations.
        Use different population initialization based on poptype.
        Stop if there's no improvement after a certain number of iterations (patience).
        
        Args:
            generations (int): Number of generations to run.
            poptype (str): Type of population initialization ('random' or 'nearest_neighbor').
            patience (int): Early stopping parameter for no improvement.
            
        Returns:
            tuple: best path, best distance, execution time, stopped generation.
        """
        # Record the start time
        start_time = time.time()

        # Select initial population method based on poptype argument
        if poptype == "random":
            current_generation = self.population_manager.initial_population_random(self.graph)
        elif poptype == "nearest_neighbor":
            current_generation = self.population_manager.initial_population_nearest_neighbor(self.graph)
        elif poptype == "greedy_tour":
            current_generation = self.population_manager.initial_population_greedy_tour(self.graph)
        elif poptype == "christofides":
            current_generation = self.population_manager.initial_population_christofides(self.graph)
        elif poptype == "random_and_neighbor":
            self.population_manager.population_size_range = (
                self.population_manager.population_size_range[0] // 2, 
                self.population_manager.population_size_range[1] // 2   
            )
            half_generation = self.population_manager.initial_population_nearest_neighbor(self.graph)
            other_half_generation = self.population_manager.initial_population_random(self.graph)
            current_generation = half_generation + other_half_generation
        else:
            raise ValueError(f"Unknown population type: {poptype}")

        # Initialize variables to keep track of the best solution
        best_path = min(current_generation, key=self.get_fitness)
        best_distance = self.fitness_function.fitness_function_distance_based(best_path)

        # Track the number of generations without improvement
        no_improvement_counter = 0
        best_overall_distance = best_distance
        best_overall_path = best_path[:]
        stopped_generation = 0

        for gen in range(generations):
            # Generate a new population
            current_generation = self.population_manager.gen_new_population(
                current_generation, self.selection_manager, self.fitness_function
            )

            # Find the best path in the current generation
            current_best_path = min(current_generation, key=self.get_fitness)
            current_best_distance = self.fitness_function.fitness_function_distance_based(current_best_path)

            # Check if there's an improvement
            if current_best_distance < best_overall_distance:
                best_overall_distance = current_best_distance
                best_overall_path = current_best_path[:]
                no_improvement_counter = 0  # Reset counter if improvement occurs
            else:
                no_improvement_counter += 1

            # Stop if no improvement for 'patience' generations
            if no_improvement_counter >= patience:
                print(f"Early stopping: No improvement for {patience} generations.")
                stopped_generation = gen + 1  # Set the generation where the stopping occurred
                break

            stopped_generation = gen + 1  # Update the generation at each step

        # Record the end time
        end_time = time.time()
        execution_time = end_time - start_time

        # Add the starting city to complete the path
        best_overall_path.append(best_overall_path[0])

        # Return the best path, best distance, execution time, and the generation where it stopped
        return best_overall_path, best_overall_distance, execution_time, stopped_generation

        """
        Run the genetic algorithm for a specified number of generations.
        Stop if there's no improvement after a certain number of iterations (patience).
        """
        # Record the start time
        start_time = time.time()

        match poptype:
            case "random":
                current_generation = self.population_manager.initial_population_random(self.graph)

            case "nearest_neighbor":
                current_generation = self.population_manager.initial_population_nearest_neighbor(self.graph)

        # Initialize variables to keep track of the best solution
        best_path = min(current_generation, key=self.get_fitness)
        best_distance = self.fitness_function.fitness_function_distance_based(best_path)

        # Track the number of generations without improvement
        no_improvement_counter = 0
        best_overall_distance = best_distance
        best_overall_path = best_path[:]

        for gen in range(generations):
            # Generate a new population
            current_generation = self.population_manager.gen_new_population(
                current_generation, self.selection_manager, self.fitness_function
            )

            # Find the best path in the current generation
            current_best_path = min(current_generation, key=self.get_fitness)
            current_best_distance = self.fitness_function.fitness_function_distance_based(current_best_path)

            # Check if there's an improvement
            if current_best_distance < best_overall_distance:
                best_overall_distance = current_best_distance
                best_overall_path = current_best_path[:]
                no_improvement_counter = 0  # Reset counter if improvement occurs
            else:
                no_improvement_counter += 1

            # Stop if no improvement for 5 generations
            if no_improvement_counter >= patience:
                print(f"Early stopping: No improvement for {patience} generations.")
                break

        # Record the end time
        end_time = time.time()
        execution_time = end_time - start_time

        # Add the starting city to complete the path
        best_overall_path.append(best_overall_path[0])

        # Return the best path, best distance, and execution time
        return best_overall_path, best_overall_distance, execution_time


    #def run(self, generations=100, poptype = "random"):
        """
        Run the genetic algorithm for a specified number of generations.
        """
        start_time = time.time()

        match poptype:
            case "random":
                current_generation = self.population_manager.initial_population_random(self.graph)

            case "nearest_neighbor":
                current_generation = self.population_manager.initial_population_nearest_neighbor(self.graph)
        

        for _ in range(generations):
            current_generation = self.population_manager.gen_new_population(
                current_generation, self.selection_manager, self.fitness_function
            )


        if not current_generation:
            print("\nNo valid paths found in the final generation.")
            return None, None

        best_path = min(current_generation, key=self.get_fitness)
        best_distance = self.fitness_function.fitness_function_distance_based(best_path)
        #quick fix to add 0 at end to make a "round" path
        best_path.append(best_path[0])

        end_time = time.time()

        execution_time = end_time - start_time

        return best_path, best_distance, execution_time

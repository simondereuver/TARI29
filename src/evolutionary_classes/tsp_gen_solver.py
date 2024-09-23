# genetic_algorithm_solver.py
"""
Module using the needed classes to compute the TSP using the genetic algorithm
"""
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

    def preprocess_graph(self) -> tuple[dict, dict]:
        """
        Analyzes the graph gathering and returning statistics and recommended selection and crossover methods and mutation rate.
        
        Parameters:
        - non (uses the graph assigned to the class)

        Returns:
        - methods: a tuple of dicts containing statistics about the graph and recommended choices.
        """
        """
        mean = statistics['mean']
        median = statistics['median']
        std_dev = statistics['std_dev']
        min_val = statistics['min']
        max_val = statistics['max']
        #variance = statistics['variance']
        """
        upper_triangle = self.graph[np.triu_indices_from(self.graph, k=1)]
        min_val = np.min(upper_triangle)
        max_val = np.max(upper_triangle)
        mean = np.mean(upper_triangle)
        median = np.median(upper_triangle)
        std_dev = np.std(upper_triangle)
        num_edges = np.size(upper_triangle)

        #calculate skewness, if its positive normal dist is shifted to right
        #https://study.com/academy/lesson/skewness-in-statistics-definition-formula-example.html
        skewness = (3 * (mean - median)) / std_dev if std_dev != 0 else 0
        skewness_percentage = skewness / std_dev * 100
        #https://en.wikipedia.org/wiki/Coefficient_of_variation

        #standard deviation is x% of the mean => cv = x
        cv = (std_dev / mean) * 100 if mean != 0 else 0


        range_ratio = ((max_val - min_val) / mean) * 100 if mean != 0 else 0

        recommendations = {"selection": "", "crossover": "", "mutation_rate": 0}

        #percentiles and interquartile range (iqr), gotta love numpy
        #help us understand spread in middle 50% of our data
        q1 = np.percentile(upper_triangle, 25)
        q3 = np.percentile(upper_triangle, 75)
        iqr = q3 - q1

        relative_iqr = (iqr / mean) * 100 if mean != 0 else 0

        # Define thresholds
        skewness_threshold = 10  # You might adjust this based on your data
        cv_threshold = relative_iqr #the relative iqr tells us if we have a high variability or not
        low_range_threshold = ((q1 - min_val) / mean) * 100 if mean != 0 else 0
        high_range_threshold = ((max_val - q3) / mean) * 100 if mean != 0 else 0

        #in relation to the mean
        if skewness_percentage > skewness_threshold:
            recommendations["selection"] = "Tournament Selection" #more higher edge weights
        elif skewness_percentage < -skewness_threshold:
            recommendations["selection"] = "Elitism" #more lower edge weights
        else:
            recommendations["selection"] = "Roulette Wheel Selection" #many values close to median

        # Adjust crossover method based on CV
        if cv > cv_threshold:
            recommendations["crossover"] = "Uniform Crossover"
        else:
            recommendations["crossover"] = "One-point Crossover"

        # Adjust mutation rate based on range ratio
        if range_ratio > high_range_threshold:
            recommendations["mutation_rate"] = 0.05  # High mutation rate
        elif range_ratio < low_range_threshold:
            recommendations["mutation_rate"] = 0.01  # Low mutation rate
        else:
            recommendations["mutation_rate"] = 0.02  # Medium mutation rate

        statistics = {
        "min": min_val,
        "max": max_val,
        "mean": mean,
        "median": median,
        "std_dev": std_dev,
        "num_edges": num_edges,
        "skewness": skewness,
        "cv": cv,
        "range_ratio": range_ratio,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "relative_IQR": relative_iqr,
        }

        return statistics, recommendations



    def run(self, generations=100):
        """
        Run the genetic algorithm for a specified number of generations.
        """
        current_generation = self.population_manager.initial_population(self.graph)

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

        return best_path, best_distance

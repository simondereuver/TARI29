"""
Module containing selection, solutions are selected based on their fitness to produce offspring
"""
import random
from typing import List
import numpy as np

class Selection:
    """
    A class to handle various selection methods for genetic algorithms.
    """
    def __init__(self, population: List[List[int]], fitness_scores: List[float]) -> None:
        """
        Initialize the selection process with a population of routes and their corresponding fitness scores.
        
        Parameters:
        - population: A list of routes (each route is a list of nodes).
        - fitness_scores: A list of fitness scores corresponding to each route (e.g., route distance).
        """
        self.population = population
        self.fitness_scores = fitness_scores
        if len(population) != len(fitness_scores):
            raise ValueError("Population and fitness_scores must have the same length.")

    def tournament_selection(self, tournament_size: int, num_selections: int) -> List[List[int]]:
        """
        Perform tournament selection to choose routes for the next generation.
        
        Parameters:
        - tournament_size: The number of candidates to consider in each tournament.
        - num_selections: The number of routes to select for the next generation.
        
        Returns:
        - A list of selected routes.
        """
        selected_routes = []
        
        for _ in range(num_selections):
                # Randomly select a subset of routes for the tournament
            tournament_contenders = random.sample(range(len(self.population)), tournament_size)
                
                # Find the route with the best fitness (lowest distance)
            best_route = min(tournament_contenders, key=lambda idx: self.fitness_scores[idx])
                
                # Add the best route from the tournament to the selected routes
            selected_routes.append(self.population[best_route])
            
        return selected_routes
    
    def rank_selection(self, num_selections: int) -> list:
        """
        Perform rank-based selection.
        
        Parameters:
        - num_selections: Number of routes to select for the next generation.
        
        Returns:
        - List of selected routes.
        """

        if num_selections > len(self.population):
            raise ValueError("Number of selections cannot be greater than population size.")
        
        # Get indices sorted by fitness
        sorted_indices = np.argsort(self.fitness_scores)
        ranks = np.arange(1, len(self.fitness_scores) + 1)  # Rank from 1 to N
         # Convert ranks to probabilities (higher rank, higher chance of being selected)
        total_rank_sum = np.sum(ranks)
        selection_probabilities = ranks / total_rank_sum
        selected_indices = np.random.choice(sorted_indices, size=num_selections, p=selection_probabilities)
        return [self.population[i] for i in selected_indices]
    
    
    def roulette_wheel_selection(self, num_selections: int) -> List[List[int]]:
        """
        Perform roulette wheel selection (fitness-proportionate selection).
        
        Parameters:
        - num_selections: Number of routes to select for the next generation.
        
        Returns:
        - List of selected routes.
        """
        if num_selections > len(self.population):
            raise ValueError("Number of selections cannot be greater than population size.")
        
        if any(f == 0 for f in self.fitness_scores):
            raise ValueError("Fitness scores must not contain zero values.")
        
        # Invert fitness scores for selection (assuming lower fitness scores are better)
        inverted_fitness = 1 / np.array(self.fitness_scores)
        
        # Calculate the probabilities proportional to the inverted fitness
        total_fitness = np.sum(inverted_fitness)
        selection_probabilities = inverted_fitness / total_fitness
        
        # Select indices based on fitness probabilities
        selected_indices = np.random.choice(len(self.population), size=num_selections, p=selection_probabilities)
        
        return [self.population[i] for i in selected_indices]
    

  
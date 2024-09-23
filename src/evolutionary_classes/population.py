# population.py
"""
Module containing population-based search
"""

import random
import numpy as np
import networkx as nx
from evolutionary_classes.selection import Selection
from evolutionary_classes.fitness_function import FitnessFunction


class Population:
    """Class for generating and modifying the population"""
    def __init__(self, mutation_rate=0.01, population_size_range=(10, 50)):
        """
        Initialize Population class.
        
        Args:
            mutation_rate (float): Probability of mutation (0 <= mutation_rate <= 1).
            population_size_range (tuple): Range for the initial population size.
        """
        #maybe should use the mutation range based on the size of population
        if not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")
        if not (isinstance(population_size_range, tuple) and len(population_size_range) == 2):
            raise ValueError("population_size_range must be a tuple with two values")
        if population_size_range[0] >= population_size_range[1]:
            raise ValueError("population_size_range must be a tuple (min, max) where min < max")

        self.mutation_rate = mutation_rate
        self.population_size_range = population_size_range

    def initial_population_random(self, graph: np.ndarray) -> list:
        """Generate the initial population."""
        total_destinations = graph.shape[0]
        random_paths = []

        #implement some logic based on number of nodes
        #since it is n! we should probably have a max limit on this
        #so we dont get too big of a population
        min_size = self.population_size_range[0] * total_destinations
        max_size = self.population_size_range[1] * total_destinations
        population_size = random.randint(min_size, max_size)

        for _ in range(population_size):
            random_path = list(range(1, total_destinations))
            random.shuffle(random_path)
            random_path = [0] + random_path
            random_paths.append(random_path)

        return random_paths
    
    def initial_population_nearest_neighbor(self, graph: np.ndarray) -> list:
        """Generate the initial population using the nearest neighbor heuristic."""
        total_destinations = graph.shape[0]
        population = []

        # Determine the population size based on the graph's size
        min_size = self.population_size_range[0] * total_destinations
        max_size = self.population_size_range[1] * total_destinations
        population_size = np.random.randint(min_size, max_size)

        for _ in range(population_size):
            # Randomly choose a starting city
            start_city = np.random.randint(0, total_destinations)
            path = self.nearest_neighbor_tour(graph, start_city)
            population.append(path)

        return population

    def nearest_neighbor_tour(self, graph: np.ndarray, start_city: int) -> list:
        """Generate a single tour using the nearest neighbor heuristic."""
        total_destinations = graph.shape[0]
        visited = [False] * total_destinations
        visited[start_city] = True  # Mark the start city as visited
        tour = [start_city]

        current_city = start_city

        for _ in range(1, total_destinations):
            # Find the nearest unvisited city
            nearest_city = None
            nearest_distance = float('inf')

            for city in range(total_destinations):
                if not visited[city] and graph[current_city][city] < nearest_distance:
                    nearest_distance = graph[current_city][city]
                    nearest_city = city

            # Visit the nearest city
            visited[nearest_city] = True
            tour.append(nearest_city)
            current_city = nearest_city

        return tour
    
    def initial_population_greedy_tour(self, graph: np.ndarray) -> list:
        """Generate the initial population using a greedy approach."""
        total_destinations = graph.shape[0]
        greedy_paths = []

        # Implement some logic based on number of nodes with a max limit for the population
        min_size = self.population_size_range[0] * total_destinations
        max_size = self.population_size_range[1] * total_destinations
        population_size = random.randint(min_size, max_size)

        # Generate `population_size` greedy paths
        for _ in range(population_size):
            # Start at a random node
            start_node = random.randint(0, total_destinations - 1)
            unvisited = set(range(total_destinations))
            unvisited.remove(start_node)
            greedy_path = [start_node]

            current_node = start_node
            while unvisited:
                # Find the nearest neighbor to the current node
                nearest_neighbor = min(unvisited, key=lambda node: graph[current_node][node])
                greedy_path.append(nearest_neighbor)
                unvisited.remove(nearest_neighbor)
                current_node = nearest_neighbor

            # Optionally, return to the start node to make a complete cycle (optional based on problem formulation)
            greedy_paths.append(greedy_path)

        return greedy_paths
    
    def initial_population_christofides(self, graph: np.ndarray) -> list:
        """Generate the initial population using Christofides' Algorithm."""
        G = nx.Graph()
        total_destinations = graph.shape[0]
        
        # Add nodes and edges to the graph
        for i in range(total_destinations):
            for j in range(i + 1, total_destinations):
                G.add_edge(i, j, weight=graph[i][j])

        # Step 1: Create a Minimum Spanning Tree (MST)
        mst = nx.minimum_spanning_tree(G)
        
        # Step 2: Find all vertices with odd degree in the MST
        odd_vertices = [v for v in mst.nodes() if mst.degree(v) % 2 == 1]
        
        # Step 3: Create a minimum weight perfect matching among the odd degree vertices
        matching = nx.algorithms.matching.min_weight_matching(G.subgraph(odd_vertices))

        # Step 4: Combine the MST and the matching to create an Eulerian circuit
        eulerian_graph = nx.MultiGraph(mst)
        for u, v in matching:
            eulerian_graph.add_edge(u, v)

        # Step 5: Create an Eulerian circuit and convert it to a Hamiltonian circuit
        eulerian_circuit = list(nx.eulerian_circuit(eulerian_graph))
        eulerian_path = []
        visited = set()

        for u, v in eulerian_circuit:
            if u not in visited:
                visited.add(u)
                eulerian_path.append(u)
            eulerian_path.append(v)

        # Ensure the tour starts and ends at the depot (0)
        if eulerian_path[0] != 0:
            eulerian_path = [0] + eulerian_path
        eulerian_path = list(dict.fromkeys(eulerian_path))  # Remove duplicates while maintaining order
        eulerian_path.append(0)  # Return to depot

        # Create the initial population
        population = [eulerian_path]

        # Handle population size as per your requirements
        min_size = self.population_size_range[0] * total_destinations
        max_size = self.population_size_range[1] * total_destinations
        population_size = np.random.randint(min_size, max_size)

        # If you want more diverse paths, you can create variations of the initial path
        while len(population) < population_size:
            # Randomly shuffle the eulerian_path (not optimal, just for diversity) 
            new_path = eulerian_path[1:-1]  # Exclude the depot
            np.random.shuffle(new_path)
            new_path = [0] + new_path + [0]
            population.append(new_path)

        return population

    
    

    def _create_children(self, parent_a: list, parent_b: list) -> list:
        """Creates a child out of a pair of parents using Order Crossover (OX)."""
        
        children = []
        
        # Step 1: Randomly select a crossover segment from parent_a
        start = random.randint(0, len(parent_a) - 2)  # Ensure start < end
        end = random.randint(start + 1, len(parent_a) - 1)
        
        # Step 2: Copy the segment from parent_a
        sub_path_a = parent_a[start:end]
        
        # Step 3: Fill the remaining positions with parent_b, preserving the order
        sub_path_b = [item for item in parent_b if item not in sub_path_a]

        # Step 4: Construct the child by combining the segments
        for i in range(len(parent_a)):
            if start <= i < end:
                children.append(sub_path_a[i - start])
            else:
                # Check if sub_path_b still has elements before popping
                if sub_path_b:
                    children.append(sub_path_b.pop(0))
                else:
                    # In case sub_path_b runs out (shouldn't happen), just repeat parent_b or parent_a
                    children.append(parent_b[i])
        
        return children


    def crossovers(self, survivors: list) -> list:
        """Creates crossovers using the _create_children method."""
        #there are different crossover methods, we need to test to see which gives best result
        children = []
        mid = len(survivors) // 2
        for i in range(mid):
            parent_a, parent_b = survivors[i], survivors[i + mid]
            children.append(self._create_children(parent_a, parent_b))
            children.append(self._create_children(parent_b, parent_a)) # pylint: disable=arguments-out-of-order
        return children

    def mutate_population(self, generation: list) -> list:
        """Mutates a small percentage of the population."""
        #maybe should use the mutation percentage based on the size of population
        #test different percentages
        mutated_population = []
        for path in generation:
            if random.random() < self.mutation_rate:
                index1, index2 = random.sample(range(1, len(path)), 2)
                path[index1], path[index2] = path[index2], path[index1]
            mutated_population.append(path)
        return mutated_population

    def gen_new_population(self, curr_gen: list, selection: Selection, ff: FitnessFunction) -> list:
        """Generate a new population using selection, crossover, and mutation."""
        survivors = selection.survivors(curr_gen, ff)
        children = self.crossovers(survivors)
        combined_population = survivors + children
        new_population = self.mutate_population(combined_population)
        return new_population

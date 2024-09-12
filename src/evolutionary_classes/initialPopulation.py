"""
Generates the initial population of our Evolutionary algorithm
"""
import random


class InitialPopulation:

    @staticmethod
    def generate_initial_population(number_of_nodes, size):
        random_paths = []

        for _ in range(size):
            random_path = list(range(1,number_of_nodes))
            random.shuffle(random_path)
            random_path = [0] + random_path
            random_paths.append(random_path)
        
        return random_paths
    
    @staticmethod
    def testing(number_of_nodes):
        ip = InitialPopulation
        paths = ip.generate_initial_population(number_of_nodes)
        for path in paths:
            print(path)
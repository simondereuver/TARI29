"""
Module containing population-based search
"""
import random

class RandomPopulation:
    def __init__(self, nodes, size) -> None:
        self.values =[random.sample(nodes, len(nodes)) for _ in range(size)]

    def get_values(self):
        return self.values


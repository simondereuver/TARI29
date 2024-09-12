"""
Module containing selection, solutions are selected based on their fitness to produce offspring
"""
import random

class Selection:

    def calculate_distance(graph, path):
        total_dist = 0
        for i in range(len(path)):
            total_dist += graph[path[i-1],path[i]]

        return total_dist
    
    def pick_survivor(graph, old_gen):

        survivor = []

        random.shuffle(old_gen)
        half = len(old_gen)//2

        for i in range(half):
            if Selection.calculate_distance(graph, old_gen[i]) < Selection.calculate_distance(graph, old_gen[i + half]):
                survivor.append(old_gen[i])
            else:
                survivor.append(old_gen[i + half])

        return survivor
"""
Module containing main graph (warehouse representation)
"""

from itertools import permutations
from typing import List
import random
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    """Graph class functionality"""
    def __init__(self, nodes: int, weight_span: tuple) -> None:
        if nodes <= 1:
            raise ValueError("Graph must have more than 1 node")

        self.nodes = nodes

        if not isinstance(weight_span, tuple) or len(weight_span) != 2:
            raise TypeError("weight_span must be a tuple of two numbers.")

        if weight_span[0] > weight_span[1]:
            self.weight_span = weight_span[::-1]
        else:
            self.weight_span = weight_span

        self.solved = False
        self.shortest_path = []

    def generate_random_graph(self, seed: int = None) -> np.ndarray:
        """Function for generating a random graph"""

        # useful for generating pseudo-random graphs
        if seed is not None:
            random.seed(seed)

        # initialize a zeroed matrix
        graph = np.zeros((self.nodes, self.nodes), dtype=int)

        start_time = time.time()
        # only iterate over upper triangle and set random values within the weight span
        for i in range(self.nodes):
            for j in range(i + 1, self.nodes):
                graph[i][j] = random.randint(*self.weight_span)
                graph[j][i] = graph[i][j]

        end_time = time.time()
        print(f"Created graph in: {(end_time - start_time):.10f} seconds")
        return graph

    def show_graph(self, graph: np.ndarray) -> None:
        """Plots the graph"""

        g = nx.Graph()

        # add to nx graph g
        for i, row in enumerate(graph):
            for j, value in enumerate(row[i + 1:], start=i + 1):
                if value != 0:
                    g.add_edge(i, j, weight=value)

        # configs for plotting
        pos = nx.spring_layout(g,
                               k=0.5,
                               scale=10,
                               iterations=100)

        nx.draw(g,
                pos,
                with_labels=True,
                node_color='lightblue',
                node_size=max(50, 700 // len(graph)),
                font_size=12)

        edge_labels = nx.get_edge_attributes(g, 'weight')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)

        if self.solved:
            path_edges = list(zip(self.shortest_path[:-1], self.shortest_path[1:]))

            nx.draw_networkx_edges(g, pos, edgelist=path_edges, edge_color='red', width=1.5)

        plt.show()

    def solve_bf(self, graph: np.ndarray, start: int):
        """Brute-force solver"""
        def all_paths_lazy(total_nodes: int, start_node: int) -> List[List]:

            start_time = time.time()

            nodes = list(range(total_nodes))
            nodes.remove(start_node)

            all_paths_list = []

            for perm in permutations(nodes):
                path = [start_node] + list(perm)
                all_paths_list.append(path)

            end_time = time.time()
            print(f"Found all paths in: {(end_time - start_time):.10f} seconds")
            return all_paths_list

        paths = all_paths_lazy(len(graph), start)

        start_time = time.time()

        shortest_path = None
        min_weight = float('inf')

        for path in paths:
            weight = 0

            for i in range(len(path) - 1):
                weight += graph[path[i]][path[i+1]]

                if weight >= min_weight:
                    break

            weight += graph[path[-1]][start]

            if weight < min_weight:
                min_weight = weight
                shortest_path = path + [start]


        end_time = time.time()
        print(f"Found shortest path in: {(end_time - start_time):.10f} seconds")
        self.shortest_path = shortest_path
        self.solved = True
        return shortest_path, min_weight

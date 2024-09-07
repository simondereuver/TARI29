"""
Graph class
"""
import random
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

    def generate_random_graph(self, seed: int = None) -> np.ndarray:
        """Function for generating a random graph"""

        # useful for generating pseudo-random graphs
        if seed is not None:
            random.seed(seed)

        # initialize a zeroed matrix
        graph = np.zeros((self.nodes, self.nodes), dtype=int)
        
        # only iterate over upper triangle and set random values within the weight span
        for i in range(self.nodes):
            for j in range(i + 1, self.nodes):
                graph[i][j] = random.randint(*self.weight_span)
                graph[j][i] = graph[i][j]
        return graph

    def show_graph(self, graph: np.ndarray) -> None:
        """Plots the graph"""

        g = nx.Graph()

        # add to nx graph g
        for i in range(len(graph)):
            for j in range(i + 1, len(graph)):
                if graph[i][j] != 0:
                    g.add_edge(i, j, weight=graph[i][j])

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

        plt.show()

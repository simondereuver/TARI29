"""
Module containing main graph (warehouse representation)
"""
import heapq
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

    def one_tree(self, graph: np.ndarray, root: int) -> int:
        """Calculates the 1-tree for a given root node."""
        def prim_mst(graph: np.ndarray, exclude_node: int) -> int:
            """Prim's MST algorithm, with the modification of excluding the root node."""
            start_node = next(i for i in range(len(graph)) if i != exclude_node)
            mst_edges = []
            visited = set()
            priority_queue = []

            visited.add(exclude_node) #exclude the root node
            visited.add(start_node) #start building the MST from a random node

            for adj_node, weight in enumerate(graph[start_node]):
                if weight > 0 and adj_node != exclude_node:
                    heapq.heappush(priority_queue, (weight, start_node, adj_node))

            mst_weight = 0

            while priority_queue and len(mst_edges) < len(graph) - 2:  # n-2 edges in MST
                weight, u, v = heapq.heappop(priority_queue)
                if v not in visited:
                    visited.add(v)
                    mst_edges.append((u, v))
                    mst_weight += weight

                    for adj_node, edge_weight in enumerate(graph[v]):
                        if edge_weight > 0 and adj_node not in visited and adj_node != exclude_node:
                            heapq.heappush(priority_queue, (edge_weight, v, adj_node))

            return mst_weight

        #calculate the edges from the root and choose the smallest edges
        root_edges = []
        for adj_node, weight in enumerate(graph[root]):
            if weight > 0:
                root_edges.append((weight, root, adj_node))
        root_edges.sort()
        min_root_edges = root_edges[:2]

        #should not occour as we have a TSP graph
        if len(min_root_edges) < 2:
            raise ValueError("Not enough edges from root node to form a valid 1-tree.")

        #calculate the MST excluding the root (the rest of the graph)
        mst_weight = prim_mst(graph, root)
        #add the two smallest root edges to the MST weight for the 1-tree weight
        total_weight = mst_weight + sum(edge[0] for edge in min_root_edges)
        return total_weight

    def max_one_tree_lower_bound(self, graph: np.ndarray) -> int:
        """Find the maximum 1-tree lower bound by trying all nodes as roots"""
        max_1tree_weight = float('-inf')
        for root in range(self.nodes):
            one_tree_weight = self.one_tree(graph, root)
            max_1tree_weight = max(max_1tree_weight, one_tree_weight)

        return max_1tree_weight

    def nearest_neighbor(self, graph):
        """Nearest neighbour for upperbound"""
        nodes = len(graph)
        start = random.randint(0, nodes - 1)

        unvisited = set(range(nodes))
        unvisited.remove(start)
        tour = [start]
        current_node = start

        while unvisited:
            next_node = min(unvisited, key=lambda node: graph[current_node][node])
            tour.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node

        tour.append(start)

        x = np.sum(graph[tour[:-1], tour[1:]])
        x += graph[tour[-1], tour[0]]

        return x

    def preprocess_graph(self, graph: np.ndarray) -> tuple:
        """
        Analyzes the graph and returns statistics and recommended choices.

        Parameters:
        - graph (np.ndarray): The graph to analyze.

        Returns:
        - statistics (dict): Statistics about the graph.
        - recommendations (dict): Recommended parameters for the genetic algorithm.
        """

        #extract the upper triangle of the adjacency matrix (since the graph is undirected)
        upper_triangle = graph[np.triu_indices_from(graph, k=1)]

        min_val = np.min(upper_triangle)
        max_val = np.max(upper_triangle)
        mean = np.mean(upper_triangle)
        median = np.median(upper_triangle)
        std_dev = np.std(upper_triangle)
        num_edges = np.size(upper_triangle)

        #calculate skewness using pearsons second skewness coefficient
        skewness = (3 * (mean - median)) / std_dev if std_dev != 0 else 0
        skewness_percentage = skewness * 100

        #coefficient of bariation
        cv = (std_dev / mean) * 100 if mean != 0 else 0

        range_ratio = ((max_val - min_val) / mean) * 100 if mean != 0 else 0

        recommendations = {
            "selection_methods": [],
            "crossover_method": "",
            "mutation_rate": 0,
            "tournament_size": None
        }

        #calc quartiles and iqr
        q1 = np.percentile(upper_triangle, 25)
        q3 = np.percentile(upper_triangle, 75)
        iqr = q3 - q1

        relative_iqr = (iqr / mean) * 100 if mean != 0 else 0

        skewness_threshold = 1
        cv_threshold = relative_iqr
        low_range_threshold = ((q1 - min_val) / mean) * 100 if mean != 0 else 0
        high_range_threshold = ((max_val - q3) / mean) * 100 if mean != 0 else 0

        if skewness_percentage > skewness_threshold:
            selection_method_name = "tournament"
            selection_parameters = {"tournament_size": 5}
            elitism_rate = 0.1
        elif skewness_percentage < -skewness_threshold:
            selection_method_name = "rank_selection"
            selection_parameters = {}
            elitism_rate = 0.2
        else:
            selection_method_name = "roulette_wheel"
            selection_parameters = {}
            elitism_rate = 0.15

        selection_methods = [("elitism", elitism_rate), (selection_method_name, 1 - elitism_rate)]

        recommendations["selection_methods"] = selection_methods
        if selection_method_name == "tournament":
            recommendations["tournament_size"] = selection_parameters.get("tournament_size", 2)

        #adjust crossover method based on CV and skewness
        if cv > cv_threshold:
            if skewness_percentage > skewness_threshold:
                recommendations["crossover_method"] = "SCX"
            else:
                recommendations["crossover_method"] = "CX"
        else:
            if skewness_percentage < -skewness_threshold:
                recommendations["crossover_method"] = "CX"
            else:
                recommendations["crossover_method"] = "OX"

        if range_ratio > high_range_threshold:
            recommendations["mutation_rate"] = 0.05  
        elif range_ratio < low_range_threshold:
            recommendations["mutation_rate"] = 0.01
        else:
            recommendations["mutation_rate"] = 0.02

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

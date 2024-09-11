"""main program code here"""


# example code for generating random graphs
from main_classes.graph import Graph

NUMBER_OF_NODES = 5
EDGE_WEIGHT_SPAN = (1, 25)

g = Graph(NUMBER_OF_NODES, EDGE_WEIGHT_SPAN)

graph = g.generate_random_graph(seed=1)

print(graph)

shortest_path, weight = g.solve_bf(graph, 0)

print(f"Shortest path: {shortest_path}, {weight} meters")

g.show_graph(graph)

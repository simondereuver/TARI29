"""
Random graph generator
"""

from graph import Graph

NUMBER_OF_NODES = 7
EDGE_WEIGHT_SPAN = (1, 25)

g = Graph(NUMBER_OF_NODES, EDGE_WEIGHT_SPAN)

graph = g.generate_random_graph()

g.show_graph(graph)

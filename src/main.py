"""main program code here"""


# example code for generating random graphs
from main_classes.graph import Graph

NUMBER_OF_NODES = 5
EDGE_WEIGHT_SPAN = (1, 25)

g = Graph(NUMBER_OF_NODES, EDGE_WEIGHT_SPAN)

graph = g.generate_random_graph(seed=0)

print(graph)

shortest_path, weight = g.solve_bf(graph, 0)

lowerbound1 = g.max_one_tree_lower_bound(graph)

print(f"Shortest path: {shortest_path}, {weight} meters, Lowerbound_1_tree: {lowerbound1}")

g.show_graph(graph)

# Example usage to run the 1-tree on 10 different graphs
#for i in range(10):
#    graph = g.generate_random_graph(i)
#
#    print(graph)

#    shortest_path, weight = g.solve_bf(graph, 0)

    #lowerbound = g.lower_bound(graph)
#    lowerbound1 = g.max_one_tree_lower_bound(graph)

#    print(f"Shortest path: {shortest_path}, {weight} meters, Lowerbound_1_tree: {lowerbound1}")

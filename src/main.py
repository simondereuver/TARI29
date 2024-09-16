"""main program code here"""


# example code for generating random graphs
from main_classes.graph import Graph
from evolutionary_classes.population_search import RandomPopulation

NUMBER_OF_NODES = 5
EDGE_WEIGHT_SPAN = (1, 25)

g = Graph(NUMBER_OF_NODES, EDGE_WEIGHT_SPAN)

graph = g.generate_random_graph(seed=0)

print(graph)

shortest_path, weight = g.solve_bf(graph, 0)

lowerbound1 = g.max_one_tree_lower_bound(graph)

print(f"Shortest path: {shortest_path}, {weight} meters, Lowerbound_1_tree: {lowerbound1}")

g.show_graph(graph)


# Generate a population of random node permutations
population_size = 5  
random_population = RandomPopulation(nodes=list(range(NUMBER_OF_NODES)), size=population_size, seed=42)

node_permutations = random_population.get_values()

print(f"Generated Node Permutations: {node_permutations}")

# Example usage to run the 1-tree on 10 different graphs
#for i in range(10):
#    graph = g.generate_random_graph(i)
#
#    print(graph)

#    shortest_path, weight = g.solve_bf(graph, 0)

    #lowerbound = g.lower_bound(graph)
#    lowerbound1 = g.max_one_tree_lower_bound(graph)

#    print(f"Shortest path: {shortest_path}, {weight} meters, Lowerbound_1_tree: {lowerbound1}")

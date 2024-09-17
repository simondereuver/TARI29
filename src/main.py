"""main program code here"""


# example code for generating random graphs
from main_classes.graph import Graph
from evolutionary_classes.selection import Selection
#from evolutionary_classes.fitness_function import FitnessFunction
from evolutionary_classes.population import Population

NUMBER_OF_NODES = 10
EDGE_WEIGHT_SPAN = (1, 25)

g = Graph(NUMBER_OF_NODES, EDGE_WEIGHT_SPAN)

graph = g.generate_random_graph(seed=0)

print(graph)

shortest_path, weight = g.solve_bf(graph, 0)

lowerbound1 = g.max_one_tree_lower_bound(graph)

print(f"Shortest path: {shortest_path}, {weight} meters, Lowerbound_1_tree: {lowerbound1}")

#g.show_graph(graph)

population_mngr = Population()
selection_mngr = Selection()
initial_population = population_mngr.initial_population(graph)

NUM_GENERATIONS = 100
curr_gen = initial_population

#run for a set of generations
for generation in range(NUM_GENERATIONS):
    curr_gen = population_mngr.gen_new_population(graph, curr_gen, selection_mngr)

if not curr_gen:
    print("\nNo valid paths found in the final generation.")
else:
    best_path = min(curr_gen, key=lambda path: selection_mngr.distance(graph, path))
    best_distance = selection_mngr.distance(graph, best_path)
    # + [0] is a quick fix for now to make it look similar to the solution in the bruteforce
    print(f"\nBest path found: {best_path + [0]} with distance: {best_distance}")


#Example usage to run the 1-tree on 10 different graphs
#for i in range(10):
#    graph = g.generate_random_graph(i)
#
#    print(graph)

#    shortest_path, weight = g.solve_bf(graph, 0)

    #lowerbound = g.lower_bound(graph)
#    lowerbound1 = g.max_one_tree_lower_bound(graph)

#    print(f"Shortest path: {shortest_path}, {weight} meters, Lowerbound_1_tree: {lowerbound1}")

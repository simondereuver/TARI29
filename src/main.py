"""main program code here"""

# example code for generating random graphs
from main_classes.graph import Graph
#from evolutionary_classes.selection import Selection
#from evolutionary_classes.fitness_function import FitnessFunction
#from evolutionary_classes.population import Population

from evolutionary_classes.tsp_gen_solver import TSPGeneticSolver

def print_results(poptype, best_path, best_distance, execution_time, stopped_generation, lowerbound):
    """
    Helper function to print the results of the TSP algorithm in a formatted way.
    
    Args:
        poptype (str): The type of initial population used ('random' or 'nearest_neighbor').
        best_path (list): The best path found by the algorithm.
        best_distance (float): The distance of the best path.
        execution_time (float): The time taken to complete the run.
        stopped_generation (int): The generation at which the algorithm stopped.
    """
    print(f"Results using {poptype} initialization:")
    print(f"Best Path: {best_path}")
    print(f"Best Distance: {best_distance:.2f}")
    print(f"Percentage of lowerbound: {lowerbound*100/best_distance:.2f}%")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Stopped at Generation: {stopped_generation}")
    print("="*50)  # Divider line for better readability


NUMBER_OF_NODES = 20
EDGE_WEIGHT_SPAN = (10, 100)

g = Graph(NUMBER_OF_NODES, EDGE_WEIGHT_SPAN)

graph = g.generate_random_graph(seed=1)

#print(graph)

#shortest_path, weight = g.solve_bf(graph, 0)

lowerbound1 = g.max_one_tree_lower_bound(graph)

#print(f"Shortest path: {shortest_path}, {weight} meters, Lowerbound_1_tree: {lowerbound1}")

#g.show_graph(graph)

#create dictionary to set parameters instead
solver = TSPGeneticSolver(
    graph,
    population_size_range=(10, 50),
    mutation_rate=0.01,
    bounds=(lowerbound1, None))


# Run the solver with random initial population
best_path_random, best_distance_random, time_random, gen_random = solver.run(generations=300, poptype="random", patience=10)
print_results("random", best_path_random, best_distance_random, time_random, gen_random, lowerbound1)

# Run the solver with nearest neighbor initial population
best_path_nn, best_distance_nn, time_nn, gen_nn = solver.run(generations=300, poptype="nearest_neighbor", patience=10)
print_results("nearest_neighbor", best_path_nn, best_distance_nn, time_nn, gen_nn, lowerbound1)

best_path_gt, best_distance_gt, time_gt, gen_gt = solver.run(generations=300, poptype="greedy_tour", patience=10)
print_results("greedy_tour", best_path_gt, best_distance_gt, time_gt, gen_gt, lowerbound1)

best_path_ch, best_distance_ch, time_ch, gen_ch = solver.run(generations=300, poptype="christofides", patience=10)
print_results("christofides", best_path_ch, best_distance_ch, time_ch, gen_ch, lowerbound1)

best_path_rn, best_distance_rn, time_rn, gen_rn = solver.run(generations=300, poptype="random_and_neighbor", patience=10)
print_results("random_and_neighbor", best_path_rn, best_distance_rn, time_rn, gen_rn, lowerbound1)



#Example usage to run the 1-tree on 10 different graphs
#for i in range(10):
#    graph = g.generate_random_graph(i)
#
#    print(graph)

#    shortest_path, weight = g.solve_bf(graph, 0)

    #lowerbound = g.lower_bound(graph)
#    lowerbound1 = g.max_one_tree_lower_bound(graph)

#    print(f"Shortest path: {shortest_path}, {weight} meters, Lowerbound_1_tree: {lowerbound1}")


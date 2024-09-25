"""main program code here"""

from main_classes.graph import Graph
from evolutionary_classes.tsp_gen_solver import TSPGeneticSolver

NUMBER_OF_NODES = 100
EDGE_WEIGHT_SPAN = (10, 100)

def main():
    g = Graph(NUMBER_OF_NODES, EDGE_WEIGHT_SPAN)

    graph = g.generate_random_graph(seed=1)

    print(graph)

    #shortest_path, weight = g.solve_bf(graph, 0)

    lowerbound = g.max_one_tree_lower_bound(graph)
    #print(f"Lowerbound estimated: {lowerbound1}")

    #print(f"Shortest path: {shortest_path}, {weight} meters, Lowerbound_1_tree: {lowerbound1}")
    print(f"Lowerbound: {lowerbound}")
    #g.show_graph(graph)

    #create dictionary to set parameters instead
    solver = TSPGeneticSolver(
        graph,
        population_size_range=(100, 101),
        mutation_rate=0.05,
        bounds=(lowerbound, None),
        crossover_method="OX"
        )

    best_path, best_distance = solver.run(generations=10000, population_size=100)

    print(f"\nBest path found: {best_path} with distance: {best_distance}")
    print(f"Score: {lowerbound / best_distance}")


if __name__ == "__main__":
    main()
#Example usage to run the 1-tree on 10 different graphs
#for i in range(10):
#    graph = g.generate_random_graph(i)
#
#    print(graph)

#    shortest_path, weight = g.solve_bf(graph, 0)

    #lowerbound = g.lower_bound(graph)
#    lowerbound1 = g.max_one_tree_lower_bound(graph)

#    print(f"Shortest path: {shortest_path}, {weight} meters, Lowerbound_1_tree: {lowerbound1}")



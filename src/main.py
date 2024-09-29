"""main program code here"""

import time
import pandas as pd
from main_classes.graph import Graph
from evolutionary_classes.tsp_gen_solver import TSPGeneticSolver

NUMBER_OF_NODES = 100
EDGE_WEIGHT_SPAN = (10, 50)

def main():
    """Main program execution"""
    g = Graph(NUMBER_OF_NODES, EDGE_WEIGHT_SPAN)

    graph = g.generate_random_graph(seed=1)

    print(graph)

    #shortest_path, weight = g.solve_bf(graph, 0)

    lowerbound = g.max_one_tree_lower_bound(graph)
    #print(f"Lowerbound estimated: {lowerbound1}")

    #print(f"Shortest path: {shortest_path}, {weight} meters, Lowerbound_1_tree: {lowerbound1}")
    print(f"Lowerbound: {lowerbound}")
    #g.show_graph(graph)

    crossover_methods = ["SCX", "OX", "CX", "PMX"]
    selection_methods = ["elitism", "tournament", "roulette_wheel", "rank_selection"]
    selection_methods1 = [("elitism", 0.1), ("rank_selection", 0.9)]
    for crossover_method in crossover_methods:
        print(f"\nTesting crossover method: {crossover_method} with {selection_methods1}")
        solver = TSPGeneticSolver(
            graph,
            mutation_rate=0.02,
            bounds=(lowerbound, None),
            crossover_method=crossover_method,
            selection_methods=selection_methods1
            )

        #best_path, best_distance = solver.run(generations=400, population_size=100)
        _, best_distance, convergence_start, convergence_duration = solver.run(generations=100, population_size=100)
        print(f"\nBest path distance: {best_distance}")
        #print(f"\nBest path found: {best_path} with distance: {best_distance}")
        print(f"Score: {lowerbound / best_distance}")


def run_tests():
    crossover_methods = ["SCX", "OX", "CX"]
    selection_methods = ["elitism", "tournament", "roulette_wheel", "rank_selection"]
    selection_methods1 = [("elitism", 0.1), ("tournament", 0.9)]
    num_generations = 300
    # Create a DataFrame to store the results
    results = []

    for i in range(3):
        print(f"\nGenerating graph {i + 1}...")
        g = Graph(NUMBER_OF_NODES, EDGE_WEIGHT_SPAN)
        graph = g.generate_random_graph(seed=i)

        lowerbound = g.max_one_tree_lower_bound(graph)

        for crossover_method in crossover_methods:
            for selection_method in selection_methods:
                print(f"\nTesting crossover method: {crossover_method} with {selection_methods1}")

                start_time = time.time()

                solver = TSPGeneticSolver(
                    graph,
                    mutation_rate=0.02,
                    bounds=(lowerbound, None),
                    crossover_method=crossover_method,
                    selection_methods=selection_methods1
                )

                _, best_distance, convergence_start, convergence_duration = solver.run(num_generations, population_size=100)
                end_time = time.time()

                elapsed_time = end_time - start_time
                score = lowerbound / best_distance

                # Append the results to the list
                results.append({
                    "Graph": i + 1,
                    "Crossover Method": crossover_method,
                    "Selection Method": selection_method,
                    "Best Distance": best_distance,
                    "Score": score,
                    "Time (s)": elapsed_time,
                    "Number of Nodes": NUMBER_OF_NODES,
                    "Edge Weight Span": EDGE_WEIGHT_SPAN,
                    "Convergence Start": convergence_start,
                    "Convergence Duration": convergence_duration,
                    "Total number of generations": num_generations
                })

    # Convert results into a DataFrame for easy viewing
    results_df = pd.DataFrame(results)

    # Print the results nicely
    print("\nTest Results:")
    print(results_df)

if __name__ == "__main__":
    #run_tests()
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

"""main program code here"""
# pylint: disable=too-many-arguments,line-too-long
import time
import os
import pandas as pd
from main_classes.graph import Graph
from evolutionary_classes.tsp_gen_solver import TSPGeneticSolver

#CHANGE THESE PARAMETERS TO TRY DIFFERENT GRAPHS
NUMBER_OF_NODES = 100
EDGE_WEIGHT_SPAN = (10, 50)
NUM_GRAPHS = 3


def main():
    """Main program execution"""
    run_tests()
    #run_tests_w_preprocess()

def run_tests():
    """
    Simple method to test some combinations of crossovers with selections
    Prints the results in a tabular format when the run is done.
    """
    crossover_methods = ["SCX", "OX", "CX"]
    #examples on how to set selection methods
    #[("elitism", 0.1), ("roulette_wheel", 0.9)]
    #[("elitism", 0.1), ("tournament", 0.9)], [("roulette_wheel", 1.0)], [("rank_selection", 1.0)]]
    selection_methods_list = [[("elitism", 1.0)]]
    num_generations = 300
    results = []

    for i in range(NUM_GRAPHS):
        print(f"\nGenerating graph {i + 1}...")
        g = Graph(NUMBER_OF_NODES, EDGE_WEIGHT_SPAN)
        graph = g.generate_random_graph(seed=i)
        lowerbound = g.max_one_tree_lower_bound(graph)

        for crossover_method in crossover_methods:
            for selection_methods in selection_methods_list:
                print(f"\nTesting crossover method: {crossover_method} with {selection_methods}")

                start_time = time.time()

                #adjust the parameters in the TSPGeneticSolver to test as you like
                solver = TSPGeneticSolver(
                            graph=graph,
                            mutation_rate=0.02,
                            bounds=(lowerbound, None),
                            crossover_method=crossover_method,
                            selection_methods=selection_methods,
                            survive_rate = 0.1,
                            tournament_size=2)

                _, best_distance, convergence_start, convergence_duration = solver.run(num_generations, population_size=100)
                end_time = time.time()

                elapsed_time = end_time - start_time
                score = lowerbound / best_distance

                results.append({
                    "Graph": i + 1,
                    "Crossover Method": crossover_method,
                    "Selection Method": selection_methods,
                    "Best Distance": best_distance,
                    "Score": score,
                    "Time (s)": elapsed_time,
                    "Number of Nodes": NUMBER_OF_NODES,
                    "Edge Weight Span": EDGE_WEIGHT_SPAN,
                    "Convergence Start": convergence_start,
                    "Convergence Duration": convergence_duration,
                    "Total number of generations": num_generations
                })


    results_df = pd.DataFrame(results)
    #print the results nicely :^)
    print("\nTest Results:")
    print(results_df)
    file_path = "results.txt"

    if os.path.exists(file_path):
        results_df.to_csv(file_path, mode='a', header=False, index=False, sep='\t')
    else:
        results_df.to_csv(file_path, mode='w', header=True, index=False, sep='\t')

    print(f"Data written to {file_path}")

def run_tests_w_preprocess():
    """
    Makes use of the preprocessing of the graph.
    Runs the Genetic Algorithm with the parameters of the preprocessing
    """
    results = []
    for i in range(NUM_GRAPHS):
        print(f"\nGenerating graph {i + 1}...")
        g = Graph(NUMBER_OF_NODES, EDGE_WEIGHT_SPAN)
        graph = g.generate_random_graph(seed=i)

        #preprocess the graph and get statistics and parameters
        statistics, parameters = g.preprocess_graph(graph)

        print(f"\nGraph {i + 1} Statistics:")
        stats_df = pd.DataFrame(statistics.items(), columns=["Metric", "Value"])
        print(stats_df)

        print(f"\nGraph {i + 1} Parameters:")
        for key, value in parameters.items():
            print(f"{key}: {value}")

        lowerbound = g.max_one_tree_lower_bound(graph)
        selection_methods = parameters["selection_methods"]
        crossover_method = parameters["crossover_method"]
        mutation_rate = parameters["mutation_rate"]
        tournament_size = parameters.get("tournament_size", None)

        start_time = time.time()

        #adjust the parameters in the TSPGeneticSolver to test as you like
        solver = TSPGeneticSolver(
            graph=graph,
            mutation_rate=mutation_rate,
            bounds=(lowerbound, None),
            crossover_method=crossover_method,
            selection_methods=selection_methods,
            survive_rate=0.5,
            tournament_size=tournament_size)
        num_generations = 300
        _, best_distance, convergence_start, convergence_duration = solver.run(generations=num_generations, population_size=200)

        #print(f"Best path distance: {best_distance} Score: {lowerbound / best_distance} with crossover method {crossover_method}, selection methods: {selection_methods}, convergence started at: {convergence_start}, duration: {convergence_duration}")

        end_time = time.time()

        elapsed_time = end_time - start_time
        score = lowerbound / best_distance

        results.append({
            "Graph": i + 1,
            "Crossover Method": crossover_method,
            "Selection Method": selection_methods,
            "Best Distance": best_distance,
            "Score": score,
            "Time (s)": elapsed_time,
            "Number of Nodes": NUMBER_OF_NODES,
            "Edge Weight Span": EDGE_WEIGHT_SPAN,
            "Convergence Start": convergence_start,
            "Convergence Duration": convergence_duration,
            "Total number of generations": num_generations
        })

    results_df = pd.DataFrame(results)
    print("\nTest Results:")
    print(results_df)

    file_path = "results.txt"

    if os.path.exists(file_path):
        results_df.to_csv(file_path, mode='a', header=False, index=False, sep='\t')
    else:
        results_df.to_csv(file_path, mode='w', header=True, index=False, sep='\t')

    print(f"Data written to {file_path}")

if __name__ == "__main__":
    main()

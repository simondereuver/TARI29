"""main program code here"""


# example code for generating random graphs
from main_classes.graph import Graph
from evolutionary_classes.initialPopulation import InitialPopulation
from evolutionary_classes.selection import Selection

NUMBER_OF_NODES = 5
EDGE_WEIGHT_SPAN = (1, 25)
INITIAL_POPULATION_SIZE = 10

g = Graph(NUMBER_OF_NODES, EDGE_WEIGHT_SPAN)

graph = g.generate_random_graph(seed=1)

print(graph)

"""
shortest_path, weight = g.solve_bf(graph, 0)
print(f"Shortest path: {shortest_path}, {weight} meters")
""" 
pop = InitialPopulation.generate_initial_population(NUMBER_OF_NODES, INITIAL_POPULATION_SIZE)

for i in range(INITIAL_POPULATION_SIZE):
    totalDist = Selection.calculate_distance(graph, pop[i])
    print(pop[i])
    print(totalDist)
    
print("---------------")

survivors = Selection.pick_survivor(graph, pop)
for survivor in survivors:
    totalDist = Selection.calculate_distance(graph, survivor)
    print(survivor)
    print(totalDist)

g.show_graph(graph)

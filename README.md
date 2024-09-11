# Warehouse Item Collection Problem

## Dependencies and Recommendations

- Python 3.x, you can specify in the python venv which pyhton version you want to use, if you have multiple installations on your computer.
- Create a Python virtual environment by running: `python -m venv nameofenv` (note: you can name the enviroment whatever you want and don't forget to activate it when you are working on the project).
- To activate the environment:
- On **Linux** and **Mac**:
  `source nameofenv/bin/activate`
- On **Windows**:
  `nameofenv\Scripts\activate`
- To deactivate the environment, run:
    `deactivate`
- Once the environment is activated, install the dependencies by running: `pip install -r requirements.txt`
- If you add a new dependency to the project, make sure to add it to the list of the `requirments.txt` file.

## Problem Definition

Find the shortest route to collect items for a order in a warehouse and return to the starting point (drop off zone).

### Steps Involved

1. **Generate Warehouse Graph**:
   Create a graph that represents the layout of the warehouse. Nodes represent locations of items, and edges represent the distances between them.

2. **Generate Sub-Graphs**:
   Extract sub-graphs from the main warehouse graph that represent only the locations and distances relevant to the current order.

3. **Compute Lower Bound**:
   Compute a lower bound on the length of the shortest route in the graph.

4. **Apply Evolutionary Algorithm**:
   Use a genetic algorithm to evolve and search for an optimal or near-optimal route through the selected nodes (items to be collected).

## Current Focus

As of now, we are prioritizing:

1. Computing the lower bound using methods like Minimum Spanning Tree (MST).
2. Applying the evolutionary algorithm to randomly generated graphs to achieve an optimal or near-optimal solution.

Future work will involve refining the graph generation to match a real warehouse layout and handling sub-graphs.

## Functions

### `create_warehouse_graph(num_rows: int, num_cols: int) -> Tuple[Graph, Tuple[int, int]]`

Generates a graph representing the warehouse with weighted edges between nodes.

- **Input**:
  - `num_rows`: The number of rows in the warehouse layout.
  - `num_cols`: The number of columns in the warehouse layout.

- **Output**:
  - `graph`: A graph where nodes are locations and edges are distances between them.
  - `start_end_point`: The start/end point (drop off zone).

### `generate_sub_graph(initial_graph: Graph, nodes_to_visit: List[int]) -> Graph`

Generates a sub-graph from the initial warehouse graph, including only the nodes and edges necessary for visiting the specified items.

- **Input**:
  - `initial_graph`: The full warehouse graph.
  - `nodes_to_visit`: A list of nodes representing items to be collected (including the start/end point).

- **Output**:
  - `sub_graph`: A sub-graph representing only the items and paths relevant to the order.

### `compute_lower_bound(graph: Graph) -> float`

Computes a lower bound for the shortest route, which could be based on the Minimum Spanning Tree (MST) or another heuristic.

- **Input**:
  - `graph`: The graph (or sub-graph) for which to compute the lower bound.

- **Output**:
  - `lower_bound`: The computed lower bound for the shortest route.

---

## Evolutionary Algorithm Class

### Methods

- **`crossover()`**: Combines solutions from the population to generate new offspring.
- **`fitness()`**: Evaluates the quality of a solution based on the route distance.
- **`mutation()`**: Introduces random changes to a solution to maintain diversity in the population.
- **`populate()`**: Initializes a population of potential solutions.
- **`selection()`**: Selects the best solutions to pass on their genes to the next generation.

---

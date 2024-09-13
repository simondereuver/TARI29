# Warehouse Item Collection Problem

## Dependencies and Recommendations

### Makefile commands and usage

This project uses a Makefile to automate common tasks such as setting up a virtual environment, installing dependencies, running linting with `pylint`, and cleaning up the environment.

#### Available Commands

1. **`make start`**:
   - **Purpose**: Sets up the environment by creating a virtual environment (if it doesn't already exist), installs the dependencies, opens VS Code, and prints instructions for manually activating the virtual environment.
   - **Steps**:
     1. Creates a virtual environment called `venv` (if necessary).
     2. Installs the project dependencies from `requirements.txt`.
     3. Opens the project in VS Code.
     4. Provides instructions to manually activate the virtual environment. (There is no workaround this, so the enviroment has to be manually activated after running `make start`).

   - **Usage**:
    `make start`

2. **`make lint`**:
    - **Purpose**: Runs pylint on the source code to check for code style issues and errors. This requires the virtual environment to be activated first, usefull to catch errors before you push and create a merge request (so it passes the pipeline).
    Note: You must manually activate the virtual environment before running this command, as the subshell created by Makefile doesn't persist the environment activation.

    - **Usage**:
    Activate the virtual enviroment:
    On Windows:
    `venv\Scripts\activate`
    On Unix-like systems (Linux, macOS):
    `source nameofenv/bin/activate`
    Then run: `make lint`.

3. **`make clean`**:

    - Purpose: Removes the virtual environment directory, cleaning up the project.
    - Steps:
    Removes the virtual environment folder **venv**.

    - **`Usage`**:
    `make clean`

4. **`IMPORTANT NOTE`**:
    - You need to have make installed to use it, makefile has been created with make version 4.4.

### Do it manually

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
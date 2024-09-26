"""
Module containing crossover methods
"""

import random
import numpy as np

class Crossover:
    """Class for generating children from parents"""

    def __init__(self, cross_over_method="Simple", graph=None):
        """
        Initialize Crossover class.

        Args:
            cross_over_method (str): Method to use for crossover.
            graph (optional): Graph data required for SCX method.
        """

        # Check if the cross_over_method is valid
        if cross_over_method not in ["Simple", "OX", "CX", "PMX", "SCX"]:
            raise ValueError(
                "Invalid cross_over_method. Choose from 'Simple', 'OX', 'CX', 'PMX', 'SCX'"
            )

        self.cross_over_method = cross_over_method

        if cross_over_method == "SCX":
            if graph is None:
                raise ValueError("Graph must be provided for SCX crossover method.")
            self.distance_matrix = graph
        else:
            self.distance_matrix = None

    def order(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        """Creates a child using Order Crossover (OX) with numpy arrays."""
        size = parent_a.shape[0]
        child = np.full(size, -1, dtype=parent_a.dtype)

        start, end = sorted(random.sample(range(size), 2))

        child[start:end + 1] = parent_a[start:end + 1]

        child_set = set(child[start:end + 1])

        b_index = (end + 1) % size
        c_index = (end + 1) % size
        filled_positions = np.count_nonzero(child != -1)

        while filled_positions < size:
            gene = parent_b[b_index]
            if gene not in child_set:
                child[c_index] = gene
                child_set.add(gene)
                c_index = (c_index + 1) % size
                filled_positions += 1
            b_index = (b_index + 1) % size

        return child

    def cycle(self, parent_a: list, parent_b: list) -> list:
        """Create a child using Cycle Crossover (CX)."""
        size = len(parent_a)
        child = [None] * size
        indices = list(range(size))
        cycle = 0
        while None in child:
            if cycle % 2 == 0:
                index = next(iter(indices))
                start_gene = parent_a[index]
                while True:
                    child[index] = parent_a[index]
                    indices.remove(index)
                    index = parent_a.index(parent_b[index])
                    if parent_a[index] == start_gene:
                        break
            else:
                for index in indices:
                    child[index] = parent_b[index]
                indices.clear()
            cycle += 1

        return child

    def pmx(self, parent_a: list, parent_b: list) -> list:
        """Creates a child using Partially Mapped Crossover (PMX)."""
        size = len(parent_a)
        child = [None] * size

        # Randomly select a subset
        start, end = sorted(random.sample(range(size), 2))

        # Copy the subset from parent A to child
        child[start:end + 1] = parent_a[start:end + 1]

        # Mapping from parent B to parent A
        mapping = {}
        for i in range(start, end + 1):
            mapping[parent_b[i]] = parent_a[i]

        # Track used genes
        used_genes = set(child[start:end + 1])

        # Fill the remaining positions
        for i in range(size):
            if child[i] is None:
                gene = parent_b[i]
                visited = set()
                while gene in mapping and gene not in visited:
                    visited.add(gene)
                    gene = mapping[gene]
                # Ensure the gene is not already used
                if gene in used_genes:
                    for g in parent_b:
                        if g not in used_genes:
                            gene = g
                            break
                child[i] = gene
                used_genes.add(gene)

        return child

    def simple(self, parent_a: list, parent_b: list) -> list:
        """Creates a child out of a pair of parents."""

        child = []
        start = random.randint(0, len(parent_a) - 1)
        end = random.randint(start, len(parent_a))
        sub_path_a = parent_a[start:end]
        sub_path_b = [item for item in parent_b if item not in sub_path_a]
        for i in range(len(parent_a)):
            if start <= i < end:
                child.append(sub_path_a.pop(0))
            else:
                child.append(sub_path_b.pop(0))
        return child

    def scx1(self, parent_a: list, parent_b: list) -> list:
        """Creates a child using Sequential Constructive Crossover (SCX)."""
        size = len(parent_a)
        child = []
        visited = set()

        # Start with the first node of parent_a
        current_node = parent_a[0]
        child.append(current_node)
        visited.add(current_node)

        while len(child) < size:
            # Find the next legitimate node from parent_a
            next_a = self._find_next_legitimate1(current_node, parent_a, visited)

            # Find the next legitimate node from parent_b
            next_b = self._find_next_legitimate1(current_node, parent_b, visited)

            # If both are None, select a random unvisited node
            if next_a is None and next_b is None:
                remaining_nodes = set(parent_a) - visited
                chosen_node = random.choice(list(remaining_nodes))
            else:
                # Compare distances and choose the better edge
                if next_a is None:
                    chosen_node = next_b
                elif next_b is None:
                    chosen_node = next_a
                else:
                    distance_a = self.distance_matrix[current_node][next_a]
                    distance_b = self.distance_matrix[current_node][next_b]

                    if distance_a < distance_b:
                        chosen_node = next_a
                    else:
                        chosen_node = next_b

            # Add the chosen node to the child
            child.append(chosen_node)
            visited.add(chosen_node)
            current_node = chosen_node

        return child

    def scx(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        """Creates a child using Sequential Constructive Crossover (SCX) with NumPy arrays."""
        size = parent_a.shape[0]
        child = np.empty(size, dtype=parent_a.dtype)
        visited = np.zeros(size, dtype=bool)  # Assuming city indices range from 0 to size - 1

        # Create mappings from city indices to their positions in parents
        positions_a = np.empty(size, dtype=int)
        positions_a[parent_a] = np.arange(size)

        positions_b = np.empty(size, dtype=int)
        positions_b[parent_b] = np.arange(size)

        # Start with the first node of parent_a
        current_node = parent_a[0]
        child[0] = current_node
        visited[current_node] = True

        position = 1  # Next position to fill in child

        while position < size:
            # Find the next legitimate node from parent_a
            next_a = self._find_next_legitimate(current_node, parent_a, visited, positions_a)

            # Find the next legitimate node from parent_b
            next_b = self._find_next_legitimate(current_node, parent_b, visited, positions_b)

            # If both are None, select a random unvisited node
            if next_a is None and next_b is None:
                remaining_nodes = np.where(~visited)[0]
                chosen_node = np.random.choice(remaining_nodes)
            else:
                # Compare distances and choose the better edge
                if next_a is None:
                    chosen_node = next_b
                elif next_b is None:
                    chosen_node = next_a
                else:
                    distance_a = self.distance_matrix[current_node, next_a]
                    distance_b = self.distance_matrix[current_node, next_b]

                    if distance_a < distance_b:
                        chosen_node = next_a
                    else:
                        chosen_node = next_b

            # Add the chosen node to the child
            child[position] = chosen_node
            visited[chosen_node] = True
            current_node = chosen_node
            position += 1

        return child

    def _find_next_legitimate1(
            self,
            current_node: int,
            parent: list,
            visited:set)->int:
        size = len(parent)
        index = parent.index(current_node)

        for i in range(1, size):
            candidate = parent[(index + i) % size]
            if candidate not in visited:
                return candidate
        return None

    def _find_next_legitimate(self, current_node: int, parent: np.ndarray, visited: np.ndarray, positions: np.ndarray) -> int:
        size = parent.shape[0]
        index = positions[current_node]

        # Generate candidate indices in the parent, wrapping around using modulo
        candidate_indices = (index + np.arange(1, size)) % size
        candidate_cities = parent[candidate_indices]

        # Find the first unvisited candidate
        unvisited_mask = ~visited[candidate_cities]

        if np.any(unvisited_mask):
            # Return the first unvisited candidate
            first_unvisited_city = candidate_cities[np.argmax(unvisited_mask)]
            return first_unvisited_city
        else:
            return None

    def create_children(self, parent_a: list, parent_b: list) -> list:
        """
        Creates a child out of a pair of parents.

        Args:
            parent_a (list): The first parent.
            parent_b (list): The second parent.

        Returns:
            list: The generated child.
        """
        switcher = {
            "Simple": self.simple,
            "OX": self.order,
            "CX": self.cycle,
            "PMX": self.pmx,
            "SCX": self.scx
        }

        return switcher[self.cross_over_method](parent_a, parent_b)

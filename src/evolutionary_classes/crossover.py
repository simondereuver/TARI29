"""Module containing OX, CX, SCX crossover methods"""
# pylint: disable=too-many-arguments,line-too-long,too-many-function-args

import random
import numpy as np

class Crossover:
    """Class for generating children from parents"""

    def __init__(self, cross_over_method="SCX", graph=None):
        """
        Initialize Crossover class.

        Args:
            cross_over_method (str): Method to use for crossover.
            graph (optional): Graph data required for SCX method.
        """

        #Check if the cross_over_method is valid
        if cross_over_method not in ["OX", "CX", "SCX"]:
            raise ValueError(
                "Invalid cross_over_method. Choose from 'OX', 'CX' or 'SCX'"
            )

        self.cross_over_method = cross_over_method

        if cross_over_method == "SCX":
            if graph is None:
                raise ValueError("Graph must be provided for SCX crossover method.")
            self.distance_matrix = graph
        else:
            self.distance_matrix = None

    def order(self,
              parent_a: np.ndarray,
              parent_b: np.ndarray) -> np.ndarray:
        """
        Creates a child using Order Crossover (OX).

        Args:
            parent_a (np.ndarray): The first parent.
            parent_b (np.ndarray): The second parent.

        Returns:
            np.ndarray: The child generated from the two parents.
        """
        #get size of parent
        size = parent_a.shape[0]
        #initialize child with -1s
        child = np.full(size, -1, dtype=parent_a.dtype)
        
        #select the two crossover points
        start, end = sorted(random.sample(range(size), 2))

        #add sequence from parent_a to child
        child[start:end + 1] = parent_a[start:end + 1]
        child_set = set(child[start:end + 1])

        #set indices to fill from parent b
        b_index = (end + 1) % size
        c_index = (end + 1) % size

        filled_positions = np.count_nonzero(child != -1)

        #add genes from parent b to child
        while filled_positions < size:
            gene = parent_b[b_index]
            if gene not in child_set:
                child[c_index] = gene
                child_set.add(gene)
                c_index = (c_index + 1) % size
                filled_positions += 1
            b_index = (b_index + 1) % size

        return child

    def cycle(self,
              parent_a: np.ndarray,
              parent_b: np.ndarray) -> np.ndarray:
        """
        Create a child using Cycle Crossover (CX) with NumPy.
        

        Args:
            parent_a (np.ndarray): The first parent.
            parent_b (np.ndarray): The second parent.

        Returns:
            np.ndarray: The child generated from the two parents.
        """
        #get size of parent
        size = len(parent_a)
        #initialize child with none
        child = np.full(size, None)
        indices = np.arange(size)
        cycle = 0
        #add genes to child
        while None in child:
            if cycle % 2 == 0:
                index = indices[0]
                start_gene = parent_a[index]
                while True:
                    child[index] = parent_a[index]
                    indices = indices[indices != index]
                    index = np.where(parent_a == parent_b[index])[0][0]
                    if parent_a[index] == start_gene:
                        break
            else:
                child[indices] = parent_b[indices]
                indices = np.array([])
            cycle += 1

        return child

    def scx(self,
            parent_a: np.ndarray,
            parent_b: np.ndarray) -> np.ndarray:
        """
        Creates a child using Sequential Constructive Crossover (SCX).
        
        Args:
            parent_a (np.ndarray): The first parent.
            parent_b (np.ndarray): The second parent.

        Returns:
            np.ndarray: The child generated from the two parents.
        """
        parent_a = parent_a.tolist()
        parent_b = parent_b.tolist()
        size = len(parent_a)
        child = []
        visited = set()

        #start with frist node from parent_a
        current_node = parent_a[0]
        child.append(current_node)
        visited.add(current_node)

        #add genes to child
        while len(child) < size:
            #find the next legitimate node from parent_a
            next_a = self._find_next_legitimate(current_node, parent_a, visited)

            #find the next legitimate node from parent_b
            next_b = self._find_next_legitimate(current_node, parent_b, visited)

            #if both are none, select a random unvisited node
            if next_a is None and next_b is None:
                remaining_nodes = set(parent_a) - visited
                chosen_node = random.choice(list(remaining_nodes))
            else:
                #compare distances and choose the better edge (lower weight)
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

            #add the chosen node to the child
            child.append(chosen_node)
            visited.add(chosen_node)
            current_node = chosen_node

        return np.array([child])

    def _find_next_legitimate(self,
                              current_node: int,
                              parent: list,
                              visited:set)->int:
        """
        Helper method for SCX, finds the next legitimate node from the parent sequence that hasn't been visited.

        Args:
            current_node (int): The current node in the sequence.
            parent (list): The parent's sequence of nodes.
            visited (set): The set of nodes that have already been visited.

        Returns:
            int: The next legitimate node or None if none is found.
        """
        size = len(parent)
        index = parent.index(current_node)

        for i in range(1, size):
            candidate = parent[(index + i) % size]
            if candidate not in visited:
                return candidate
        return None

    def create_children(self,
                        parent_a: np.ndarray,
                        parent_b: np.ndarray) -> np.ndarray:
        """
        Creates a child out of a pair of parents.

        Args:
            parent_a (list): The first parent.
            parent_b (list): The second parent.

        Returns:
            list: The generated child.
        """
        switcher = {
            "OX": self.order,
            "CX": self.cycle,
            "SCX": self.scx
        }

        return switcher[self.cross_over_method](parent_a, parent_b)

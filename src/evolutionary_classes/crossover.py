"""
Module containing crossover methods
"""

import random

class Crossover:
    """Class for generating children from parents"""
    def __init__(self, cross_over_method="Simple"):
        """
        Initialize Crossover class.

        Args:
            cross_over_method (str): Method to use for crossover.
        """

        # Check if the cross_over_method is valid
        if cross_over_method not in ["Simple", "OX", "CX", "PMX"]:
            raise ValueError("Invalid cross_over_method. Choose from 'Simple', 'OX', 'CX', 'PMX'")

        self.cross_over_method = cross_over_method

    def order(self, parent_a: list, parent_b: list) -> list:
        """Creates a child using Order Crossover (OX)."""
        size = len(parent_a)
        child = [None] * size

        # Randomly select a subset
        start, end = sorted(random.sample(range(size), 2))

        # Copy the subset from parent A to child
        child[start:end + 1] = parent_a[start:end + 1]

        # Use a set for faster membership checks
        child_set = set(child[start:end + 1])

        # Fill the remaining positions with genes from parent B in order
        b_index = end + 1
        c_index = end + 1
        while None in child:
            gene = parent_b[b_index % size]
            if gene not in child_set:
                child[c_index % size] = gene
                child_set.add(gene)
                c_index += 1
            b_index += 1

        return child

    def cycle(self, parent_a :list, parent_b:list)-> list:
        """Create a child using Cycle Crossover(CX)"""
        size= len(parent_a)
        child=[None]* size
        indices=list(range(size))
        cycle=0
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
        """Creates a child using Partially Mapped Crossover (PMX)"""
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

    def create_children(self, parent_a: list, parent_b: list) -> list:
        """Creates a child out of a pair of parents."""

        switcher = {
            "Simple": self.simple,
            "OX": self.order,
            "CX": self.cycle,
            "PMX": self.pmx
        }

        return switcher[self.cross_over_method](parent_a, parent_b)

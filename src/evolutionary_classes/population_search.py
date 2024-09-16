"""
Module containing population-based search
"""
import random
from  typing import  List,Optional

class RandomPopulation:
    def __init__(self, nodes: List[int] , size: int, seed:Optional[int]=None) -> None:
        if not nodes:
            raise ValueError("Nodes list cannot be empty.")
        if size <= 0:
            raise ValueError("Size must be positive.")
        if size > factorial(len(nodes)):
            raise ValueError("Size exceed the number of possible permutations.")
        if seed is not None:
            random.seed(seed)
        self.values =self.generate_uniqe(nodes ,size)

    def generate_uniqe(self, nodes: List[int], size:int)->List[List[int]]:
        seen=set()
        permutations = []
        while len(permutations)< size:
            perm=tuple(random.sample(nodes, len(nodes)))
            if perm not in seen:
                seen.add(perm)
                permutations.append(list(perm))
        return permutations

    def get_values(self) -> List[List[int]]:
        return self.values
    

def factorial(n:int) -> int:
    if n ==0 or n==1:
        return 1
    result=1
    for i in range(2, n + 1):
        result *= i
    return result


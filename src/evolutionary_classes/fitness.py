"""
Module containing fitness function(s)
"""
class Fitness: 
    def __init__(self,graph, population,the_best) -> None:
        distance = Distance(graph)
        current_best_route = the_best
        current_best_distance = distance.measure(the_best)

        for item in population:
            d = distance.measure(item)

            if d < current_best_distance:
                    current_best_distance = d
                    current_best_route = item

        return current_best_route
    

class Distance:
    def __init__(self,graph) -> None:
        self.graph = graph
        
    def measure(self, route):
        distance = 0

        for i in range(0, len(route) - 1):
              start_path = route[i]
              end_path = route[i+1]
              distance += self.graph[start_path][end_path]

        return distance
        

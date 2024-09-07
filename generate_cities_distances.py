import random 
import numpy as np 

# Create a list of cities with random coordinates

def create_cities(num_cities):
    cities = {f'City_{i}': (random.uniform(0, 100), random.uniform(0, 100)) for i in range(num_cities)}
    distance_matrix, city_list = create_distance_matrix(cities)
    return cities, distance_matrix, city_list

# Calculate distances between each pair of cities with euclidean_distance

def create_distance_matrix(cities):
    city_list = list(cities.keys())
    n = len(city_list)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = euclidean_distance(cities[city_list[i]], cities[city_list[j]])
    return distance_matrix, city_list

def euclidean_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)


# Test euclidean_distance function
point1 = (1, 2)
point2 = (4, 6)
print("Euclidean Distance between (1, 2) and (4, 6):", euclidean_distance(point1, point2)) 

# Test create_distance_matrix function with predefined cities
test_cities = {
    'City_0': (0, 0),
    'City_1': (3, 4),
    'City_2': (6, 8)
}
distance_matrix, city_list = create_distance_matrix(test_cities)
print("Test Distance Matrix:\n", distance_matrix)

# Test create_cities function
num_cities = 3
cities, distance_matrix, city_list = create_cities(num_cities)
print("Generated Cities:", cities)
print("Distance Matrix:\n", distance_matrix)
print("City List:", city_list)



 
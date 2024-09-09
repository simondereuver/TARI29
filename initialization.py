import numpy as np 

NUM_HORIZONAL_SECTION=2
NUM_VERTICAL_SECTION=5
SHELF_PER_SECTION=3


def create_matrix(num_horizonal_section,num_vertical_section,shelf_per_section):

    if num_horizonal_section <= 1:
        raise ValueError("Number of horizonal section must be 1 or higher")
    
    if num_vertical_section <= 1:
        raise ValueError("Number of vertical section must be 1 or higher")
    
    if shelf_per_section <= 1:
        raise ValueError("Number of shelf per section must be 1 or higher")
    
    if not isinstance(num_horizonal_section, int) or not isinstance(num_vertical_section, int) or not isinstance(shelf_per_section, int):
        raise ValueError("Number of cities must be an integer")

    # Dimensions of the matrix
    n = 4 * num_horizonal_section + 1
    m = 2 * num_vertical_section + 1

    matrix = np.empty((n, m),str)

    counter = 1
    for j in range(m): 
        for i in range(n): 
            if j % 2 == 0 or i % 4 == 0:
                matrix[i][j] = '🌫️'
            else:
                matrix[i][j] = '🗄️'  
                counter += 1     
    return matrix

# Test the function
result_matrix = create_matrix(NUM_HORIZONAL_SECTION, NUM_VERTICAL_SECTION, SHELF_PER_SECTION)
print(result_matrix)

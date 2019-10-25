import numpy as np 

x = np.full((10), 2)
y = np.full((5, 10), 2)

print(np.array([np.dot(row, x) for row in y]))
import numpy as np

a = [[1,2,3],
     [4,5,6],
     [7,8,9]]
b = [[1,2,3],
     [4,5,6],
     [7,8,9]]

a,b = np.array(a),np.array(b)
# dot product
print(np.dot(a,b))
print(np.matmul(a,b))
print(a@b)
# element-wise product
print(np.multiply(a,b))
print(a*b)
import numpy as np

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis = 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    
    return dZ
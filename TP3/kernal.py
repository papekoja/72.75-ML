import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# All from chatGPT though..... 
# So just as a starting point



def linear(x1, x2):
    """
    Compute the linear kernel between two feature vectors.

    Parameters:
    x1, x2: numpy arrays or lists containing feature vectors.

    Returns:
    The dot product of x1 and x2.
    """
    # Convert input to NumPy arrays for easy computation
    x1 = np.array(x1)
    x2 = np.array(x2)
    
    # Compute the dot product (inner product) between x1 and x2
    dot_product = np.dot(x1, x2)
    
    return dot_product

def polynomial(X, Y, degree=2):
    """
    Compute the polynomial kernel matrix for two sets of data points.

    Parameters:
    X : numpy array
        The first set of data points (shape: m1 x n).
    Y : numpy array
        The second set of data points (shape: m2 x n).
    degree : int
        The degree of the polynomial kernel (default is 2).

    Returns:
    K : numpy array
        The polynomial kernel matrix (shape: m1 x m2).
    """
    # Ensure that X and Y have the same number of features (n)
    assert X.shape[1] == Y.shape[1], "Number of features must match."

    # Compute the kernel matrix
    K = (np.dot(X, Y.T) + 1) ** degree

    return K

def rbf(x1, x2, gamma=1.0):
    """
    Compute the Radial Basis Function (RBF) kernel between two vectors x1 and x2.

    Parameters:
    - x1: A numpy array or list representing the first vector.
    - x2: A numpy array or list representing the second vector.
    - gamma: The kernel parameter (bandwidth).

    Returns:
    - The RBF kernel value between x1 and x2.
    """

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    diff = x1 - x2
    squared_norm = np.dot(diff, diff)
    return np.exp(-gamma * squared_norm)



def my_function():
    print("This is a function from script1")
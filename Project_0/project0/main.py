import numpy as np

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    if isinstance(n, int):
        arr = np.random.rand(n,1)
    else: 
        raise TypeError("Only integers are allowed")
    return arr

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    if not isinstance(h, int): 
        raise TypeError("Only integers are allowed")
    if not isinstance(w, int): 
        raise TypeError("Only integers are allowed")
    A = np.random.rand(h,w)
    B = np.random.rand(h,w)
    s = A + B
    return A, B, s


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    if A.shape[1] != 1:
        raise TypeError("Only Column arrays could be entered")
    if B.shape[1] != 1:
        raise TypeError("Only Column arrays could be entered")
    AB = A + B
    s = np.linalg.norm(AB)
    return s


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    if inputs.shape != weights.shape:
        raise TypeError("inputs and weights must have the same shape")
    wh = np.matmul(weights.transpose(), inputs)
    out = np.tanh(wh)
    return out

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    if x <= y:
        f = x * y
    else:
        f = x / y
    return f

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    if x.shape[0] != y.shape[0]:
        raise TypeError("vectors should be of same size")
    f = []
    for i in range(x.shape[0]):
        f.append(scalar_function(x[i],y[i]))
    return f    


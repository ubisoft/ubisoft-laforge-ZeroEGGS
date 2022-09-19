import numpy as np

def mul(x, y):
    return np.matmul(x, y)

def mul_vec(x, y):
    return np.matmul(x, y[...,np.newaxis])[...,0]
    
def transpose(x):
    return x.transpose(list(range(x.ndim-2))+[x.ndim-1, x.ndim-2])
    
# TODO: Should work for multiple matrices at once.
def psolve(x, y, eps=1e-5):
    if x.shape[0] < x.shape[1]:
        return np.linalg.solve(x.T.dot(x) + (eps*eps) * np.eye(x.shape[1]), x.T.dot(y)).T
    elif x.shape[0] > x.shape[1]:
        return x.T.dot(np.linalg.solve(x.dot(x.T) + (eps*eps) * np.eye(x.shape[0]), y)).T
    else:
        return np.linalg.solve(x + (eps*eps) * np.eye(x.shape[0]), y).T
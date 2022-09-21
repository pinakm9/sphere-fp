import numpy as np 
import tensorflow as tf 



def mu_np(X, b=0.208186):
    x, y, z = np.split(X, 3, axis=-1)
    p = np.sin(y) - b*x 
    q = np.sin(z) - b*y 
    r = np.sin(x) - b*z
    return np.concatenate([p, q, r], axis=-1) 



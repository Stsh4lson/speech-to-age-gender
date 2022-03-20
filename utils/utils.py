import numpy as np

def normalize(arr, max=1.0, min=-1.0):
    onezero = (arr-np.min(arr))/(np.max(arr)-np.min(arr))*(max-min)
    return onezero+min
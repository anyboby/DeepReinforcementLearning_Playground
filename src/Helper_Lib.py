import numpy as np


def rgb2gray(rgb, norm):
   
    gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    if norm:
        # normalize
        gray = gray.astype('float32') / 128 - 1 

    return gray 

import numpy as np
import warnings
from scipy.signal import medfilt

def projection(vec, norm):
    '''
    gets the projection of vec onto the plane defined by norm
    '''
    norm_norm = norm.dot(norm)
    proj = vec - ((vec.dot(norm)/norm_norm)*norm)
    return proj
    

def clean_mag(mag, g):
    '''
    Gets the magnetometer vector that is orthogonal to the gravity vector.
    Passes it into a median filter.
    '''
    #error handling
    if mag.shape[0] != g.shape[0]:
        raise TypeError("number of samples of the magnetometer and gravity are not equal")
        
    if mag.shape[1] != 3:
        raise TypeError("magnetometer reading is not in x,y,z form")
        
    if g.shape[1] != 3:
        raise TypeError("gravity vector is not in x,y,z form")
        
    for i in range(mag.shape[0]):
        mag[i,:] = projection(mag[i,:], g[i,:])
    
    mag = np.apply_along_axis(medfilt, 0, mag)
    
    return mag
        
    
    
    
    
    
    
    
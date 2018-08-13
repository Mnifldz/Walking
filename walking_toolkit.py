# Walking Analysis Toolkit
#------------------------------------------------------------------------------
# Last Updated 8/13/2018
# Description: Toolkit for the evaluation and analysis of walking data.  Many
# of these functions are meant to be flexible, but were originally developed for 
# the HAPT data set.

# Libraries
#------------------------------------------------------------------------------
import numpy as np
import scipy.linalg as LA



# Lie Group Operations
#------------------------------------------------------------------------------
# SO(3) Log Map
def SO3_log(Q):
    """Closed form equation for SO(3) log map."""
    R = (Q - Q.transpose())/2
    square = np.matmul(R,R)
    siz = np.sqrt(-square.trace()/2 )
    return (np.arcsin(siz)/siz)*R

# Lie algebra exponential map
def so3_exp(A):
    """ Lie algebra exponential map for so(3)."""
    vec = [A[0,1], A[0,2], A[1,2]]
    theta = LA.norm(vec)
    return np.identity(3) + (np.sin(theta)/theta)*A + ((1-np.cos(theta))/theta**2)*np.matmul(A,A)

# Row to Mat
def row_to_mat(vec):
    """ Takes gyroscope vector and makes it into a Lie algebra element."""
    return np.array([[0, vec[0], -vec[1]], [-vec[0], 0, vec[2]], [vec[1], -vec[2], 0]])

# Mat to Row
def mat_to_row(mat):
    """ Takes Lie algebra-valued matrix and returns corresponding row vector."""
    return np.array([mat[0,1], -mat[0,2], mat[1,2]])

# Gyro to Rotation
def gyro_to_rot(x,y,z):
    """Convert gyroscope data to rotational data.  This function outputs a data
       frame that has each row as a flattened rotation matrix (interpreted as a
       1x9 vector)."""
    L = x.shape[0]
    Rots = np.zeros([L,9])
    Last = np.identity(3)
    Rots[0,:] = Last.reshape([1,9])
    for i in range(1,L):
        Lie_Alg = row_to_mat([x[i-1], y[i-1], z[i-1]])
        New     = np.matmul(Last, so3_exp(Lie_Alg))
        Rots[i,:] = New.reshape([1,9])
        Last = New
    
    return Rots

def stupid_func(vec):
    return np.zeros([128,9])




















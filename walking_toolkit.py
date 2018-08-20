# Walking Analysis Toolkit
#------------------------------------------------------------------------------
# Last Updated 8/13/2018
# Description: Toolkit for the evaluation and analysis of walking data.  Many
# of these functions are meant to be flexible, but were originally developed for 
# the HAPT data set.

# Libraries
#------------------------------------------------------------------------------
import numpy as np
import numpy.linalg as npl
import scipy.linalg as LA
import time



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

# SO(3) Distance Function
def SO3_dist(A,B):
    """ Returns the Lie group distance of two rotation matrices."""
    return LA.norm( SO3_log( np.matmul(A.transpose(), B))  )

# SPD(n) Distance Function
def SPD_dist(A,B):
    """ Returns the affine-invariant distance of two SPD matrices (i.e. covariances)."""
    pre_dist = npl.solve( LA.sqrtm(A), B)
    return LA.norm( LA.logm( npl.solve( LA.sqrtm(A), pre_dist.transpose()   )  )  )

# Averaging/Functions for Minimizing Mean-Squared Distance
#------------------------------------------------------------------------------
# SO(3) mean via mean-squared distance
def SO3_geo_mean(Rotations,thresh):
    """ This function will find the barycenter of N rotations via a gradient 
        descent method and Lie group-valued computations.  The input 'Rotations'
        is expected to be an Nx9 data frame where each row is a flattened (i.e.
        vectorized) rotation matrix."""
    L = Rotations.shape[0]
    Rots = [Rotations.loc[i,:].values.reshape([3,3]) for i in range(L)]
    
    # Gradient Descent Algorithm
    current = SO3_lin_mean(Rotations) # Initial guess
    tan     = np.array([[0,1,0], [-1,0,0], [0,0,0]]) # Initial Guess for tangent vector
    siz     = 0.01 # step size of the algorithm 
    iters   = 0
    
    start = time.time()
    while LA.norm(np.matmul(current.transpose(), tan)) > thresh:
        if time.time() - start >= 60:
            break
        current = np.matmul(current, so3_exp(-siz* tan))
        Log_Mats = [SO3_log(np.matmul(x.transpose(), current )) for x in Rots]
        Log_Mats = [A for A in Log_Mats if ~np.isnan(A).any() and ~np.isinf(A).any()]
        tan =  sum(Log_Mats)
        iters += 1       
     
    return current,LA.norm(np.matmul(current.transpose(), tan)), iters+1, time.time() - start

# SO(3) mean via Log-Euclidean averaging
def SO3_lin_mean(Rotations):
    """Finds the average of rotations matrices via the Log-Euclidean framework."""
    L    = Rotations.shape[0]
    Rots = [Rotations.loc[i,:].values.reshape([3,3]) for i in range(L)]
    Logs = [SO3_log(A) for A in Rots]
    Logs = [A for A in Logs if ~np.isinf(A).any()]
    Logs = [A for A in Logs if ~np.isnan(A).any()]
    L    = len(Logs)
    return so3_exp((1/L)*sum(Logs))

# Find Lie Group Covariance from Gaussian model
def SO3_cov(Rotations,mean):
    """ This function finds the Lie group covariance of the rotations.  This
        requires the input of 'mean' as found in the function 'find_mean' as 
        given above."""
        
    L = Rotations.shape[0]
    Rots = [Rotations.loc[i,:].values.reshape([3,3]) for i in range(L)]
    
    Vecs = [mat_to_row(SO3_log( np.matmul(mean.transpose(), x ))) for x  in Rots]
    Mats = [np.outer(Vecs[i], Vecs[i]) for i in range(L)]
    return (1/L)*sum(Mats)

# Find the Mean Covariance
def cov_geo_mean(cov_seq, thresh):
    """ This function finds the SPD-valued mean of the covariances, interpreting
        the mean of these matrice as the minimizer of the mean-squared distance
        in the manifold SPD(n) with the affine-invariant metric (coinciding
        with the Fisher information metric)."""
    
    # Gradient Descent Algorithm
    L       = cov_seq.shape[0]
    Covs    = [cov_seq.loc[i,:].values.reshape([3,3]) for i in range(L)]
    current = cov_lin_mean(cov_seq)
    tan     = np.array([[0,1,0], [1,0,0], [0,0,0]])
    siz     = 0.01
    iters   = 0
    
    start = time.time()
    while LA.norm(tan) > thresh:
        if time.time() - start > 15:
            break
        pre_tan = npl.solve(LA.sqrtm(current), tan)
        tan     = npl.solve(LA.sqrtm(current), pre_tan.transpose())
        current = np.matmul( LA.sqrtm(current), np.matmul( LA.expm(-siz*tan), LA.sqrtm(current)  ))
        Log_Mats = [LA.logm( npl.solve(C, current )) for C in Covs]
        tan      = np.matmul(current, sum(Log_Mats))
        iters += 1
    
    return current, LA.norm(tan), iters+1, time.time() - start
        
# Find linear average of covariance matrices
def cov_lin_mean(cov_seq):
    """This function finds the linear mean of covariance matrices."""
    L = cov_seq.shape[0]
    Covs = [cov_seq.loc[i,:].values.reshape([3,3]) for i in range(L)]
    return(1/L)*sum(Covs)
    




















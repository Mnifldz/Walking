# Soft Gyroscope Toolkit
#------------------------------------------------------------------------------
# Last Updated 8/1/2018
# Description: These are all of the tools used for a soft gyroscope from gravity 
# and magnetometer data.  This includes functions pertaining to gravity extraction
# from accelerometer data, Lie group computations for gyroscope reconstruction, 
# and functions associated for error analysis of our resulting approximation.

# Libraries
#------------------------------------------------------------------------------
import numpy as np
from scipy.signal import butter, lfilter, medfilt
from numpy import linalg as LA
from pyquaternion import Quaternion
#import warnings

# Gravity Extraction
#------------------------------------------------------------------------------
def butter_lowpass_filter(data, corner=0.02, fs=50, order=3):
    """
    Uses signal.butter to perform the filter on data
    Arguments:
        <data> (numpy array) --- one dimensional timeseries data
        which the filter is to be applied to
        <corner> (float) --- critical (corner) frequency for LPF
        <fs> (float) --- sampling rate
        <order> (int) --- represents the order of the filter
    Returns:
        <y> (numpy array) --- data which the filter has been applied to
    """

    # scipy.signal.butter accepts an argument "Wn is normalized from 0 to 1,
    # where 1 is the Nyquist frequency, pi radians/sample"
    # the nyquist frequency is fs/2, so we must express Wn as corner / (0.5 * fs)

    Wn = corner / (0.5 * fs)
    b, a = butter(order, Wn, 'low')
    
    return lfilter(b, a, data)


def get_gravity(data):
    '''
    Takes in the raw accelerometer data as a n x 3 numpy array.
    Columns x,y,z
    Implements 2/3 of the technique shown in the following paper:
    https://hal-upec-upem.archives-ouvertes.fr/hal-00826243/document
    
    last 1/3 may be needed
    
    https://github.com/KalebKE/AccelerationExplorer/wiki/Low-Pass-Filter-Linear-Acceleration
    
    '''
    data = np.apply_along_axis(butter_lowpass_filter, 0, data)
    data = np.apply_along_axis(medfilt, 0, data)
    return data

# Magnetometer Filtering/Cleaning
#------------------------------------------------------------------------------
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

# Angular and Quaternion Functions
#------------------------------------------------------------------------------
# Find Euler Angles from Gravity and Magnetometer
def Euler_Angs(B,g):
    """Input magnetometer vector B and gravity vector g both in the device frame."""
    phi = np.arctan2(2*g[:,1], g[:,2])
    theta = np.arctan2(-g[:,0], np.multiply(g[:,1],np.sin(phi)) + np.multiply(g[:,2],np.cos(phi)) ) 
    psi = np.arctan2( np.multiply(B[:,2],np.sin(phi)) - np.multiply(B[:,1],np.cos(phi)), np.multiply(B[:,0],np.cos(theta)) + np.multiply(np.multiply(B[:,1],np.sin(phi)),np.sin(theta)) + np.multiply(np.multiply(B[:,2],np.sin(theta)),np.cos(phi)) )
    return [phi,theta,psi]

# Quaternion from Euler Angles
def Ang2Qt(phi,theta,psi):
    """Takes Euler angles and outputs their quaternion representation."""
    L = phi.shape[0]
    Q_x = [Quaternion(np.cos(phi[i]/2), np.sin(phi[i]/2),0,0) for i in range(L)] 
    Q_y = [Quaternion(np.cos(theta[i]/2), 0, np.sin(theta[i]/2), 0) for i in range(L)]
    Q_z = [Quaternion(np.cos(psi[i]/2),0,0, np.sin(psi[i]/2)) for i in range(L)]
    return [Q_x[i]*Q_y[i]*Q_z[i] for i in range(L)]

# Quaternion Derivative
def Q_Deriv(q_old, q_new, samp):
    """Takes old and new quaternion values and approximates their derivative with 'samp'
    as the sample rate (i.e. time difference between observations)."""
    return (q_new - q_old)/samp

        
# Lie Group Exponential and Logarithm
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

# Soft Gyroscope Function
#------------------------------------------------------------------------------
# Find time step from global time stamp
def time_diffs(times):
    """ This function takes the absolute time stamps and converts them into 
        relative time stamps (in seconds)."""
    L = times.shape[0]
    time_stamp = []
    for i in range(L):
        if i == 0:
            time_stamp.append(0)
        else:
            time_stamp.append( 0.001*(times[i] - times[i-1]) )
    new_stamp = [x if x!=0 else 0.01 for x in time_stamp]
    return new_stamp
            
# SOFT GYROSCOPE FUNCTION (Lie group)
def soft_gyro_Lie(Gravity, Magnet):
    """ Returns a soft gyroscope approximation based on gravity and magnetometer
        data.  Constructs the matrix and uses a Lie algebra approximation for the 
        final reconstruction."""
    L = Gravity.shape[0]
    C_vec = np.cross(Gravity, Magnet)
    C_norms = LA.norm(C_vec, axis =1)
    C_N = np.divide(C_vec, C_norms[:,None])
    
    G_norms = LA.norm(Gravity, axis=1)
    G_N     = np.divide(Gravity, G_norms[:,None])
    T_G     = np.cross(G_N, C_N)
    soft_gyro_Lie = np.zeros([L,3])
    MATS_Lie = [None]*2
    for i in range(L):
        if i == 0:    
            MATS_Lie[0] = np.array([-T_G[i,:], C_N[i,:], G_N[i,:]]).transpose()
        if i > 0:
            MATS_Lie[1] = np.array([-T_G[i,:], C_N[i,:], G_N[i,:]]).transpose()
            Lie_Alg = SO3_log( np.matmul(MATS_Lie[0].transpose(), MATS_Lie[1]) )
            soft_gyro_Lie[i,:] = mat_to_row(Lie_Alg)
            MATS_Lie[0] = MATS_Lie[1]  
    return soft_gyro_Lie

# SOFT GYROSCOPE FUNCTION (quaternion)
def soft_gyro_quat(Gravity, Magnet, Times):
    """ Returns a soft gyroscope approximation based on gravity and magnetometer
        data.  Constructs the matrix and uses a quaternion approach for the final
        reconstruction."""
    L = Gravity.shape[0]
    SoftGyro_Quat = np.zeros([L-1,3])
    [phi, theta, psi] = Euler_Angs(Magnet, Gravity)
    Quats = Ang2Qt(phi,theta,psi)
    new_T = time_diffs(Times)
    Q_dots = [Q_Deriv(Quats[i+1],Quats[i],new_T[i+1]) for i in range(L-1)]
    for i in range(L-1):
        vec = 2*Quats[i].inverse*Q_dots[i]
        SoftGyro_Quat[i,:] = vec.imaginary
    return SoftGyro_Quat
    

# Error Analysis
#------------------------------------------------------------------------------
def calculate_error(a, b):
    '''
    a is predicted vectors
    b are actual vectors
    '''
    error_list = []
    for i in range(a.shape[0]):
        #Euclidean distance
        error = LA.norm(a[i, :]-b[i, :])/2
        error = error/(LA.norm(a[i, :])+1/2)
        error_list += [error]
    return sum(error_list)/len(error_list)

def calculate_rmse(a, b):
    return np.sqrt(np.mean((a-b)**2, axis = 0))
    
def calculate_angle_error(a, b):
    a_n = np.apply_along_axis(lambda row:np.linalg.norm(row,ord=1), 1, a)
    b_n = np.apply_along_axis(lambda row:np.linalg.norm(row,ord=1), 1, b)
    a = a/a_n[:,None]
    b = b/b_n[:,None]
    return calculate_rmse(a,b)




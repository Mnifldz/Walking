# HMOG Walking Data Analysis
#------------------------------------------------------------------------------
# Author: Paul David
# Last Updated: 7/25/2018
# Description: We analyze walking data from the HMOG data set abd perform data
# analysis based on the gyroscopic and rottional data from different users and 
# accross all behaviors.

# Imported Libraries
#------------------------------------------------------------------------------
import numpy as np
from numpy import genfromtxt
from os import listdir
from numpy import linalg as npl


# Supporting Functions
#------------------------------------------------------------------------------
# so(3) Exponential Map
def so3_exp(Mat):
    """Closed form equation for so(3) exponential map."""
    theta = npl.norm([Mat[0,1], Mat[0,2], Mat[1,2]] )
    return np.identity(3) + (np.sin(theta)/theta)*Mat + ((1-np.cos(theta))/theta**2)*np.matmul(Mat, Mat)
    
# SO(3) Logarithmic Map
def SO3_log(Q):
    """Closed form equation for SO(3) log map."""
    R = (Q - Q.transpose())/2
    square = np.matmul(R,R)
    siz = np.sqrt(-square.trace()/2 )
    return (np.arcsin(siz)/siz)*R








# Rotational Reconstruction for Users
#------------------------------------------------------------------------------

# Root Folder
Root = "/Users/pauldavid/Documents/My_Files/Applications/Jobs/2018_UNIFYID/Data/HMOG/public_dataset"
big_list = listdir(Root)

# Initialize Behaviors
read_sit = [1,7,13,19]
read_walk = [2,8,14,20]
write_sit = [3,9,15,21]
write_walk = [4,10,16,22]
map_sit = [5,11,17,23]
map_walk = [6,12,18,24]

# Rotational Reconstruction for Single User
user_dir = Root + "/100669/100669_session_1/"
gyro = genfromtxt(user_dir + "Gyroscope.csv", delimiter = ",")
L = gyro.shape[0]
time_steps = [None]*L
time_steps[0] = 0
for i in range(1,L):
    time_steps[i] = gyro[i,0] - gyro[i-1,0]















# Soft Gyro and Rotations for Aggressive Data
#------------------------------------------------------------------------------
# Last Updated 7/30/2018
# Description: This script outputs the gyroscope and rotational data for aggressive
# walking cycle data.

# Libraries
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
from numpy import linalg as LA
from isolate_g import get_gravity
from clean_mag import clean_mag


# Small Functions
#------------------------------------------------------------------------------
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

# Import Data and run Soft Gyro
#------------------------------------------------------------------------------
Root = "/Users/pauldavid/Documents/My_Files/Applications/Jobs/2018_UNIFYID/Data/HMOG/public_dataset"
List = os.listdir(Root)
L = len(List)

for i in range(13,L):
    if os.path.isdir(Root + "/" + List[i]):
        for j in range(24):
            print("Processing " + List[i] + " session " + str(j+1))
            name = Root + "/" + List[i] + "/" + List[i] + "_session_" + str(j+1) + "/Gyroscope.csv"
            
            Data = np.genfromtxt(name, delimiter = ",")
            n = Data.shape[0]
            Rots = [None]*2
            t_step = [None]*2
            rotation_data = np.zeros([n, 13])
            for k in range(n):
                if k == 0:
                    Rots[0] = np.identity(3)
                if k > 0:
                    t_step = 0.001*(Data[k,0] - Data[k-1,0])
                    Rots[1] = Rots[0]*so3_exp(t_step*row_to_mat(Data[k, 3:6]))
                    rotation_data[k,:] = np.append(Data[k,0:3], np.append(Rots[1].reshape([1,9]), Data[k,6])  )
                    Rots[0] = Rots[1]
            save_name = Root + "/" + List[i] + "/" + List[i] + "_session_" + str(j+1) + "/Rotations.csv"
            np.savetxt(save_name, rotation_data, delimiter = ",")
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
        
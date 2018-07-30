# Rotations for HMOG Data
#------------------------------------------------------------------------------
# Last Updated 7/30/2018
# Description: This script outputs rotation matrices from the HMOG data set 
# taking the gyroscope data as an input.

# Libraries
#------------------------------------------------------------------------------
import numpy as np
import os
from numpy import linalg as LA


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

for i in range(138,L):
    sub_dir = Root + "/" + List[i]
    if os.path.isdir(sub_dir):
        sub_list = os.listdir(sub_dir)
        for j in range(len(sub_list)):
            sub_string = sub_list[j].split("_")
            if "session" in sub_string:
                print("Processing " + List[i] + " session " + sub_string[2])
                name = Root + "/" + List[i] + "/" + List[i] + "_session_" + sub_string[2] + "/Gyroscope.csv"
                
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
                save_name = Root + "/" + List[i] + "/" + List[i] + "_session_" + sub_string[2] + "/Rotations.csv"
                np.savetxt(save_name, rotation_data, delimiter = ",")
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
        
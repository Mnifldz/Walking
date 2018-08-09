# Cross Points for Gyroscope HAPT Data Set
#------------------------------------------------------------------------------
# Last Updated: 8/8/2018
# Description: This code finds cross points, developed for the GUPR algorithm, 
# for tracking activity changes in the HAPT data set.  We intend to use this for
# rotational activity recognition.

# Libraries
#------------------------------------------------------------------------------
import pandas as pd
import find_crossings as fc
import matplotlib.pyplot as plt
import numVector
import os

# Crossing Function (Adapted from GUPR)
#------------------------------------------------------------------------------
def example(x,y,z,rate,save_name):
    time = [rate*i for i in range(x.shape[0])] # generate vector of time stamps

    #get ready to collect the matrices
    matrices = []

    #loop through each axis
    for axis in [x,y,z]:
        #look at the documentation for getVectorsNum: try different thresholds and such for the specific data
        _, matrix, _ = numVector.getVectorsNum(axis, time, curveThreshold = 0.1, distThreshold = 0.1)
        matrices.append(matrix)

    #put all the matrices as input to the naiveCross
    crossingsDict = fc.naiveCross(*matrices)

    #I'm plotting here just to demonstrate
    for axis in [x,y,z]:
        plt.plot(time, axis, color = 'black')
    
    for key, color in zip(crossingsDict.keys(), ['red','blue', 'green']):
        for x in crossingsDict[key]:
            plt.axvline(x = x, color = color)

    plt.savefig(save_name)
    plt.close()
    return  crossingsDict


# Find Crossings of HAPT Gyro Data
#------------------------------------------------------------------------------
Root_train = "/Volumes/Seagate Backup Plus Drive/UnifyID/HAPT/UCI HAR Dataset/train/Inertial Signals/"
Files = [Root_train + "body_gyro_" + i + "_train.txt" for i in ["x", "y", "z"]]

gyro_train_x = pd.read_fwf(Files[0], header = None)
gyro_train_y = pd.read_fwf(Files[1], header = None)
gyro_train_z = pd.read_fwf(Files[2], header = None)

Dicts = []
save_dir = Root_train + "crossings"
if os.path.isdir(save_dir) == 0:
    os.makedirs(save_dir)
    
for ind in range(gyro_train_x.shape[0]):
     
    save_name = save_dir + "/gyro_train_ind=" + str(ind) + ".eps"
    Dicts += [example(gyro_train_x.loc[ind,:], gyro_train_y.loc[ind,:], gyro_train_z.loc[ind,:], 0.02, save_name)]



























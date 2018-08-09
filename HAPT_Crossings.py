# Cross Points for Gyroscope HAPT Data Set
#------------------------------------------------------------------------------
# Last Updated: 8/8/2018
# Description: This code finds cross points, developed for the GUPR algorithm, 
# for tracking activity changes in the HAPT data set.  We intend to use this for
# rotational activity recognition.

# Libraries
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
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

Root_test = "/Volumes/Seagate Backup Plus Drive/UnifyID/HAPT/UCI HAR Dataset/test/Inertial Signals/"

Files_test = [Root_test + "body_gyro_" + i + "_test.txt" for i in ["x", "y", "z"]]

gyro_test_x = pd.read_fwf(Files_test[0], header = None)
gyro_test_y = pd.read_fwf(Files_test[1], header = None)
gyro_test_z = pd.read_fwf(Files_test[2], header = None)

Dicts_test = []
save_dir_test = Root_test + "crossings"
if os.path.isdir(save_dir_test) == 0:
    os.makedirs(save_dir_test)
    
for inds in range(gyro_test_x.shape[0]):
    save_test = save_dir_test + "/gyro_test_ind=" + str(inds) + ".eps"
    Dicts_test += [example(gyro_test_x.loc[inds,:], gyro_test_y.loc[inds,:], gyro_test_z.loc[inds,:], 0.02, save_test)]


# Convert and save Cross Times
#------------------------------------------------------------------------------
cross_times_train = []
Cross_Times_Train = []
cross_times_test = []
Cross_Times_Test = []

# Training Cross Times
for i in range(gyro_train_x.shape[0]):
    cross_times_train += [Dicts[i][('x', 'y')] + Dicts[i][('z', 'x')] + Dicts[i][('y', 'z')]]

Cross_Times_Train = np.zeros([len(cross_times_train), len(max(cross_times_train,key = lambda x: len(x)))])
for i,j in enumerate(cross_times_train):
    Cross_Times_Train[i][0:len(j)] = j
    
np.savetxt(save_dir + "/_train_cross_times.csv", Cross_Times_Train, delimiter = ",")

# Testing Cross Times
for i in range(gyro_test_x.shape[0]):
    cross_times_test += [Dicts_test[i][('x', 'y')] + Dicts_test[i][('z', 'x')] + Dicts_test[i][('y', 'z')]]
    
Cross_Times_Test = np.zeros([len(cross_times_test), len(max(cross_times_test,key = lambda x: len(x)))])
for i,j in enumerate(cross_times_test):
    Cross_Times_Test[i][0:len(j)] = j
    
np.savetxt(save_dir_test + "/_test_cross_times.csv", Cross_Times_Test, delimiter = ",")




















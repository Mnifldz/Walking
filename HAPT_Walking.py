# HAPT Walking Analysis on Training and Test Data
#------------------------------------------------------------------------------
# Last Updated 8/13/2018
# Description: This script trains and classifies gyroscope walking data in the
# HAPT data set.  The procedure works as follows: 1. Convert gyroscope data to
# Lie group rotation data using differential geometric methods, 2. cluster 
# rotational data into collections on the manifold SO(3), 3. fit a Lie group-valued
# normal distribution to the data for each rotational sequence, 4. Average the
# distributions using Fisher information geometry corresponding to the same type
# of data sequence (training), 5. Use testing data to find distributions of test
# sequences and measure the Fisher distance from the training distributions, 
# 6. Classify the testing data by assigning it the label corresponding to the
# minimum distribution distance.

# Libraries
#------------------------------------------------------------------------------
import walking_toolkit as wt
import pandas as pd
import numpy as np
import os


# Import HAPT Data Set
#------------------------------------------------------------------------------
Root_train = "/Volumes/Seagate Backup Plus Drive/UnifyID/HAPT/UCI HAR Dataset/train/Inertial Signals/"
Root_test = "/Volumes/Seagate Backup Plus Drive/UnifyID/HAPT/UCI HAR Dataset/test/Inertial Signals/"

Files_train = [Root_train + "body_gyro_" + i + "_train.txt" for i in ["x", "y", "z"]]
Files_test  = [Root_test + "body_gyro_" + i + "_test.txt" for i in ["x", "y", "z"]]

gyro_train_x = pd.read_fwf(Files_train[0], header = None)
gyro_train_y = pd.read_fwf(Files_train[1], header = None)
gyro_train_z = pd.read_fwf(Files_train[2], header = None)

gyro_test_x = pd.read_fwf(Files_test[0], header = None)
gyro_test_y = pd.read_fwf(Files_test[1], header = None)
gyro_test_z = pd.read_fwf(Files_test[2], header = None)

L_train = gyro_train_x.shape[0]
L_test  = gyro_test_x.shape[0]

# Convert/Import Rotations
#------------------------------------------------------------------------------
MODE = "Write"  # Modes should either be "Write" or "Import"

Rots_train_dir = Root_train + "Rotations"
Rots_test_dir  = Root_test + "Rotations"

if MODE == "Write":
    if os.path.isdir(Rots_train_dir) == 0:
        os.mkdir(Rots_train_dir)
        
    for i in range(L_train):
        print("Processing training rotations " + str(i) + " of " + str(L_train))
        Rots_train = wt.gyro_to_rot(gyro_train_x.loc[i,:], gyro_train_y.loc[i,:], gyro_train_z.loc[i,:])
        np.savetxt(Rots_train_dir + "/rots_train_ind=" + str(i) + ".csv", Rots_train, delimiter = ",")
        
    if os.path.isdir(Rots_test_dir) == 0:
        os.mkdir(Rots_test_dir)
        
    for i in range(L_test):
        print("Processing testing rotations " + str(i) + " of " + str(L_test))
        Rots_test = wt.gyro_to_rot(gyro_test_x.loc[i,:], gyro_test_y.loc[i,:], gyro_test_z.loc[i,:])
        np.savetxt(Rots_test_dir + "/rots_test_ind=" + str(i) + ".csv", Rots_test, delimiter = ",")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
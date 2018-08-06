# HMOG Gyroscope Cross Points (Extraction for Visualization)
#------------------------------------------------------------------------------
# Last Updated: 8/3/2018
# Description: This script takes functions from the GUPR algorithm (getVectorsNum
# and findCrossings) in order to get the cross points for gyroscope data.  We 
# will subsequently take these cross points and plot them against tripod time-
# series visualization we get from SymmLab in order to verify if we are able to 
# visualize a change in behavior.  

# Libraries
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import os
from numVector import getVectorsNum
import find_crossings as fc
from soft_gyro_toolkit import time_diffs

# Find gyro crosses for HMOG data
#------------------------------------------------------------------------------
Root = "/Volumes/Seagate Backup Plus Drive/UnifyID/HMOG/public_dataset"
List = os.listdir(Root)
L = len(List)

test_name = Root + "/100669/100669_session_1/Gyroscope.csv"
test = np.genfromtxt(test_name, delimiter = ",")

gyro = pd.DataFrame(test[:,3:6])
time_orig = test[:,0]
time_stamp = pd.DataFrame(time_diffs(time_orig))

_, g_x, _  = getVectorsNum(gyro[0], time_stamp)
_, g_y, _  = getVectorsNum(gyro[1], time_stamp)
_, g_z, _  = getVectorsNum(gyro[2], time_stamp)
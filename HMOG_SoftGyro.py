# Soft Gyroscope Reconstruction for HMOG Data
#------------------------------------------------------------------------------
# Last Updated: 8/1/2018
# Description: We used the accelerometer and magnetometer data from HMOG to 
# estimate the gyroscope as well as compare it to the actual measured gyroscope 
# data taken in the HMOG data set.


# Libraries
#------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import time
from numpy import genfromtxt
import soft_gyro_toolkit as sg

# Main Loop for Gyro Reconstruction and Error Analysis
#------------------------------------------------------------------------------
Root = "/Volumes/Seagate Backup Plus Drive/UnifyID/HMOG/public_dataset"
List = os.listdir(Root)
L = len(List)

master_stats = pd.DataFrame()
inds = [x for x in range(118) ]  + [x for x in range(119,L)]

for i in inds:
    sub_dir = Root + "/" + List[i]
    if os.path.isdir(sub_dir):
        sub_list = os.listdir(sub_dir)
        for j in range(len(sub_list)):
            sub_string = sub_list[j].split("_")
            if "session" in sub_string:
                print("Processing " + List[i] + " session " + sub_string[2])
                
                # Import and Pre-Process Data
                gyro = Root + "/" + List[i] + "/" + List[i] + "_session_" + sub_string[2] + "/Gyroscope.csv"
                acc  = Root + "/" + List[i] + "/" + List[i] + "_session_" + sub_string[2] + "/Accelerometer.csv"
                mag  = Root + "/" + List[i] + "/" + List[i] + "_session_" + sub_string[2] + "/Magnetometer.csv"
                
                Gyro = genfromtxt(gyro, delimiter = ",")[:,3:6]
                Acc  = genfromtxt(acc, delimiter = ",")
                pre_mag  = genfromtxt(mag, delimiter = ",")
                least = min(Gyro.shape[0], Acc.shape[0], pre_mag.shape[0])
                
                Grav = sg.get_gravity(Acc[0:least,3:6])
                Mag  = sg.clean_mag(pre_mag[0:least,3:6], Grav)
                Gyro = Gyro[0:least,:]
                #time_stamp = sg.time_diffs(Acc[0:least,0])
                
                # Run Soft Gyroscope and Error Analysis
                start = time.time()
                soft_Gyro = sg.soft_gyroscope(Grav, Mag)
                stop = time.time()
                
                error = sg.calculate_error(soft_Gyro, Gyro)
                rmse  = sg.calculate_rmse(soft_Gyro, Gyro)
                
                approx_dat = np.concatenate((Acc[0:least,0:3], soft_Gyro, np.reshape(Acc[0:least,6], [least,1])), axis = 1)
                
                new_stats_row = pd.DataFrame([[sub_string[0], sub_string[2], str(stop-start)] + [str(error)] + [str(num) for num in rmse]], columns = ["user", "session", "runtime", "error", "rmse_1", "rmse_2", "rmse_3"])
                
                master_stats = pd.concat([master_stats, new_stats_row])
                
                # Save Soft Gyro Data
                save_name = Root + "/" + List[i] + "/" + List[i] + "_session_" + sub_string[2] + "/Soft_Gyroscope.csv"
                np.savetxt(save_name, approx_dat, delimiter = ",")
                    
# Save stats
save_stats = Root + "/Soft_Gyro_Stats.csv"
master_stats.to_csv(save_stats, sep = ",")

            
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Root = "/Volumes/Seagate Backup Plus Drive/UnifyID/HMOG/public_dataset"
file_Lie = Root + "/Soft_Gyro_Stats_Lie.csv"
file_quat = Root + "/Soft_Gyro_Stats_Quat.csv"

master_stats = pd.read_csv(file_Lie, delimiter = ",")
master_stats_Q = pd.read_csv(file_quat, delimiter = ",")
        

to_hist = ["runtime", "error", "rmse_1", "rmse_2", "rmse_3"]
Titles  = ["Run Times", "Error", "RMSE (Roll)", "RMSE (Pitch)", "RMSE (Yaw)"]
x_labs  = ["Time (s)"] + ["Error"]*4



for i in range(len(to_hist)):
    vec = [float(x) for x in master_stats[to_hist[i]].dropna()]
    vec_Q = [float(x) for x in master_stats_Q[to_hist[i]].dropna()] 
    L = len(vec)
    L_Q = len(vec_Q)
    if L != L_Q:
        if L > L_Q:
            vec = np.random.choice(vec, replace = False, size = L_Q)
        else:
            vec_Q = np.random.choice(vec_Q, replace = False, size = L)
        
    plt.hist([vec,vec_Q], bins = 15, label = ["Lie", "Quaternion"], weights = 100*np.ones_like([vec, vec_Q])/len(vec) )


    plt.title("HMOG Soft Gyro " + Titles[i])
    plt.ylabel("Percentage of Trials")
    plt.xlabel(x_labs[i])
    #plt.savefig(Root + "/SG_Lie_Hist_" + to_hist[i] + ".eps")
    
    #plt.title("Soft Gyro (Quaternion) - " + Titles[i])
    #plt.ylabel("Number of Users")
    plt.legend()
    plt.savefig(Root + "/SG_quat_Hist_" + to_hist[i] + ".eps")
    
    plt.cla()
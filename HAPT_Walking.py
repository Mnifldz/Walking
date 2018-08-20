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
MODE = ""  # Modes should either be "Write" or "Import"

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
        
# Train Data
#------------------------------------------------------------------------------
def TRAIN(Dir):
    """Here we train the HAPT data by finding normal distribution corresponding
       to each data sequence, and then use Fisher information geometry to find
       the representative normal distribution by averaging each distribution 
       corresponding to a particular class (i.e. walking, sitting, etc.).  We
       will subsequently use these meta distributions on the testing data."""
    
    rots_dir = Dir + "/Inertial Signals/Rotations"
    Dir_List = os.listdir(rots_dir)
    Dir_List = [f for f in Dir_List if os.path.isfile(os.path.join(rots_dir, f))]
    L = len(Dir_List)
    
    Means = np.zeros([L,9])
    Covs  = np.zeros([L,9])
    mean_list = []
    cov_list  = []
    Labels    = []
    
    save_dir = Dir + "/Distributions"
    if os.path.isdir(save_dir) == 0:
        os.mkdir(save_dir)
        
    
    # Find Seperate Distributions
    for i in range(L):
        print("Finding distribution " + str(i+1) + " of " + str(L))
        rots = pd.read_csv(rots_dir + "/" + Dir_List[i], delimiter = ",", header = None)
        
        new_mean,_,_,_ = wt.SO3_geo_mean(rots, 0.05)
        new_cov    = wt.SO3_cov(rots, new_mean)
        Means[i,:] = new_mean.reshape([1,9])
        Covs[i,:]  = new_cov.reshape([1,9])
        
        mean_list += [new_mean]
        cov_list  += [new_cov]
        
        lab_1 = Dir_List[i].find("=")
        lab_2 = Dir_List[i].find(".")
        Labels += [ int(Dir_List[i][(lab_1+1):lab_2]) ]
        
    # Save individual distributions
    np.savetxt(save_dir + "/Rots_Individual_Means.csv", Means, delimiter = ",")
    np.savetxt(save_dir + "/Rots_Individual_Covs.csv", Covs, delimiter = ",")
    np.savetxt(save_dir + "/Rots_Individual_Labels.csv", Labels, delimiter = ",")
    
    
    # Find Meta-Distributions:
    train_labels = pd.read_fwf(Dir + "/y_train.txt", header = None)
    LL = train_labels.shape[0]
    #LABS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
    MEAN_GEO = np.zeros([6,9])
    COV_GEO  = np.zeros([6,9])
    MEAN_LIN = np.zeros([6,9])
    COV_LIN  = np.zeros([6,9])
    for j in range(6):
        print("Training Class " + str(j+1) + " of 6")
        sub_inds = [k for k in range(LL) if j == train_labels.loc[k,0]-1]
        inds     = [int(Labels[k]) for k in sub_inds]
        
        mean_class = pd.DataFrame(data = Means[inds,:]).reset_index(drop = True)
        cov_class  = pd.DataFrame(data = Covs[inds,:]).reset_index(drop = True)
        
        print("Finding Geodesic Mean")
        New_Mean_geo = wt.SO3_geo_mean(mean_class, 0.05)
        print("Finding Geodesic Covariance")
        New_Cov_geo = wt.cov_geo_mean(cov_class, 0.05)
        
        print("FInding Linear Mean and Covariance")
        New_Mean_lin = wt.SO3_lin_mean(mean_class)
        New_Cov_lin  = wt.cov_lin_mean(cov_class)
        
        MEAN_GEO[j,:] = New_Mean_geo[0].reshape([1,9])
        COV_GEO[j,:]  = New_Cov_geo[0].reshape([1,9])
        MEAN_LIN[j,:] = New_Mean_lin.reshape([1,9])
        COV_LIN[j,:]  = New_Cov_lin.reshape([1,9])
        
        
    np.savetxt(save_dir + "/Rots_GeoClass_Means.csv", MEAN_GEO, delimiter = ",")
    np.savetxt(save_dir + "/Rots_GeoClass_Covs.csv", COV_GEO, delimiter = ",")
    np.savetxt(save_dir + "/Rots_LinClass_Means.csv", MEAN_LIN, delimiter = ",")
    np.savetxt(save_dir + "/Rots_LinClass_Covs.csv", COV_LIN, delimiter = ",")
    return MEAN_GEO, COV_GEO, MEAN_LIN, COV_LIN


# Classification
#------------------------------------------------------------------------------
def CLASSIFY(MEAN_GEO, COV_GEO, MEAN_LIN, COV_LIN, Dir):
    """This function classifies the data using the classifier found from the 
       function 'TRAIN(...)'.  A dataframe with the statistics of the data is 
       found showing the best linear and geodesic averaging as well as the 
       actual classification label."""
       
    rots_dir = Dir + "/Inertial Signals/Rotations"
    Dir_List = os.listdir(rots_dir)
    Dir_List = [f for f in Dir_List if os.path.isfile(os.path.join(rots_dir, f))]
    L = len(Dir_List)
    
    Labels = pd.read_fwf(Dir + "/y_test.txt", header = None)
    
    # Reshape All Class Parameters
    Mean_Lin = [MEAN_LIN[i,:].reshape([3,3]) for i in range(6)]
    Cov_Lin  = [COV_LIN[i,:].reshape([3,3]) for i in range(6)]
    Mean_Geo = [MEAN_GEO[i,:].reshape([3,3]) for i in range(6)]
    Cov_Geo  = [COV_GEO[i,:].reshape([3,3]) for i in range(6)]
    
    # Classification Data Frame
    geo_cols = ["Geo_dist_" + str(i) for i in range(1,7)]
    lin_cols = ["Lin_dist_" + str(i) for i in range(1,7)]
    Cols     = geo_cols + lin_cols + ["Lin_Class", "Geo_Class", "Actual_Class", ]
    CLASSIFIED = pd.DataFrame(columns = Cols)
    
    # Classify Testing Data
    for i in range(L):
        print("Finding and classifying distribution " + str(i+1) + " of " + str(L))
        rots = pd.read_csv(rots_dir + "/" + Dir_List[i], delimiter = ",", header = None)
        
        
        
        new_mean,_,_,_ = wt.SO3_geo_mean(rots,0.05)
        new_cov        = wt.SO3_cov(rots, new_mean)
        new_row        = []
        
        for s in range(6):
            new_row += [np.sqrt(wt.SO3_dist(new_mean,Mean_Geo[s])**2 + wt.SPD_dist(new_cov, Cov_Geo[s])**2 ) ]
        for s in range(6):
            new_row += [ np.sqrt(wt.SO3_dist(new_mean, Mean_Lin[s])**2 + wt.SPD_dist(new_cov, Cov_Lin[s])**2 ) ]
            
        new_row += [new_row[:6].index(min(new_row[:6]))]
        new_row += [new_row[6:].index(min(new_row[6:]))]
        new_row += [Labels.loc[i,0]]
        
        new_row = pd.DataFrame(data = [new_row], columns = Cols)
        CLASSIFIED = pd.concat([CLASSIFIED, new_row], ignore_index = True )
        
    CLASSIFIED.to_csv(Dir + "/Classified_Data.csv", index = False)
    return CLASSIFIED
        
MEAN_GEO, COV_GEO, MEAN_LIN, COV_LIN = TRAIN("/Volumes/Seagate Backup Plus Drive/UnifyID/HAPT/UCI HAR Dataset/train")    
STATS                                = CLASSIFY(MEAN_GEO, COV_GEO, MEAN_LIN, COV_LIN, "/Volumes/Seagate Backup Plus Drive/UnifyID/HAPT/UCI HAR Dataset/test")           
        
        
        
        

    
    


        
                
        
                
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
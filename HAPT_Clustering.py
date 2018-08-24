# HAPT Clustering Analysis
#------------------------------------------------------------------------------
# Last Updated: 8/22/2018
# Description: We perform three kinds of clustering on the HAPT rotational data:
# Euclidean clustering of rotations viewed as vectors obtained from their Lie 
# algebra representation, Riemannian clustering of rotations viewed directly in
# terms of their Riemannian distance (flatten matrices), 


# Libraries
#------------------------------------------------------------------------------
import walking_toolkit as wt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
#from sklearn.cluster import AgglomerativeClustering as Agl

# Import Data
#------------------------------------------------------------------------------
Root_train = "/Volumes/Seagate Backup Plus Drive/UnifyID/HAPT/UCI HAR Dataset/train"

train_means = pd.read_csv(Root_train + "/Distributions/Rots_Individual_Means.csv", delimiter = ",", header = None)
LieAlg_vecs = [wt.mat_to_row( wt.SO3_log(train_means.loc[i,:].values.reshape([3,3]) ) ) for i in range(train_means.shape[0]) ]

Labels = pd.read_fwf(Root_train + "/y_train.txt", header = None)

# Scatter Plot with Colored Labelings
#------------------------------------------------------------------------------
Cols = ["r", "b", "g", "m", "y", "c"]
for i in range(6):
    inds  = [k for k in range(Labels.shape[0]) if Labels.loc[k,0] == i+1]
    x_Lie = [LieAlg_vecs[k][0] for k in inds]
    y_Lie = [LieAlg_vecs[k][1] for k in inds]
    z_Lie = [LieAlg_vecs[k][2] for k in inds]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_Lie, y_Lie, z_Lie, c = Cols[i])
    plt.title("HAPT Lie Algebra Clusters, Class " + str(i+1))
    plt.savefig(Root_train + "/Distributions/LieAlgClust_Class=" + str(i+1) + ".eps")

# Plot Individual Sequences 
#------------------------------------------------------------------------------
Plot_Seqs = ""

if Plot_Seqs == True:
    Rots_Dir = Root_train + "/Inertial Signals/Rotations"
    Dir_List = os.listdir(Rots_Dir)
    Dir_List = [f for f in Dir_List if os.path.isfile(os.path.join(Rots_Dir, f))]
    L = len(Dir_List)
    
    for i in range(L):
        rots = pd.read_csv(Rots_Dir + "/" + Dir_List[i], delimiter = ",", header = None)
        lab_1 = Dir_List[i].find("=") + 1
        lab_2 = Dir_List[i].find(".")
        ind = Dir_List[i][lab_1:lab_2]
        
        Lie_Vecs = [wt.mat_to_row( wt.SO3_log( rots.loc[k,:].values.reshape([3,3]) )  )  for k in range(rots.shape[0])   ]
        
        X_Lie = [Lie_Vecs[k][0] for k in range(rots.shape[0])]
        Y_Lie = [Lie_Vecs[k][1] for k in range(rots.shape[0])]
        Z_Lie = [Lie_Vecs[k][2] for k in range(rots.shape[0])]
        
        Fig = plt.figure()
        AX = Fig.add_subplot(111, projection='3d')
        AX.scatter(X_Lie, Y_Lie, Z_Lie, c = "r")
        plt.title("Lie Algebra Sequence, User #" + str(ind))
        plt.savefig(Rots_Dir + "/" + Dir_List[i][:lab_2] + "_LieAlg_plot.eps")


# Find minimal ellipsoid
#------------------------------------------------------------------------------
# Training Algorithm using Training Ellipsoids and testing data.
TEST = True

if TEST == True:
   Root_train = "/Volumes/Seagate Backup Plus Drive/UnifyID/HAPT/UCI HAR Dataset/train"
   Root_test = "/Volumes/Seagate Backup Plus Drive/UnifyID/HAPT/UCI HAR Dataset/test"
   train_Labels = pd.read_fwf(Root_train + "/y_train.txt", header = None)
   test_Labels = pd.read_fwf(Root_test + "/y_test.txt", header = None)
   
   # Training Directory:
   Rots_Train = Root_train + "/Inertial Signals/Rotations"
   Dir_List_Train = os.listdir(Rots_Train)
   Dir_List_Train = [f for f in Dir_List_Train if f.endswith(".csv")]
   L_Train = len(Dir_List_Train)
   
   # Testing Directory:
   Rots_Test = Root_test + "/Inertial Signals/Rotations"
   Dir_List_Test = os.listdir(Rots_Test)
   Dir_List_Test = [f for f in Dir_List_Test if f.endswith(".csv")]
   L_Test = len(Dir_List_Test)
   
   
   # TRAIN ELLIPSOIDS
   ###################
   Ell_Centers = []
   Ell_Radii   = []
   Ell_Rots    = []
   
   for i in range(6):
       print("Training class " + str(i+1) + " of 6")
       train_inds = [k for k in range(L_Train) if train_Labels.loc[k,0] == i+1]
       L_sub      = len(train_inds)
       
       Lie_Vecs = []
       for j in range(L_sub):
           dat = pd.read_csv(Root_train + "/Inertial Signals/Rotations/rots_train_ind=" + str(train_inds[j]) + ".csv", delimiter = ",", header = None)
           rand_inds = np.random.choice(dat.shape[0], 10, replace = False)
           Lie_Vecs += [wt.mat_to_row( wt.SO3_log( dat.loc[k,:].values.reshape([3,3]) )  )  for k in rand_inds  ]
           
          
       Lie_Vecs = np.asarray(Lie_Vecs)
       Lie_Vecs = Lie_Vecs[~np.isnan(Lie_Vecs).any(axis=1)]
       Lie_Vecs = Lie_Vecs[~np.isinf(Lie_Vecs).any(axis=1)]

       
       new_center, new_rads, new_rot = wt.getMinVolEllipse( Lie_Vecs)
       
       Ell_Centers += [new_center]
       Ell_Radii   += [new_rads]
       Ell_Rots    += [new_rot]
       
       
   # TEST ELLIPSOIDS
   ###################
   Cols = ["In_Ell_1", "In_Ell_2", "In_Ell_3", "In_Ell_4", "In_Ell_5", "In_Ell_6", "Ell_Class", "Actual_Class"]
   Ell_Stats = pd.DataFrame(data = None, columns = Cols)
   Count = 0
   walk = [1,2,3]
   lay = [4,5,6]
   for i in range(L_Test):
       print("Testing User " + str(i) + " of " + str(L_Test))
       Rots = pd.read_csv(Rots_Test + "/" + Dir_List_Test[i], header = None)
       
       test_seq = [wt.mat_to_row( wt.SO3_log( Rots.loc[k,:].values.reshape([3,3]) )  )  for k in range(Rots.shape[0]) ]
       
       new_row = []
       for r in range(6):
           seq_count = [wt.In_Ellipsoid(test_seq[k], Ell_Centers[r], Ell_Radii[r], Ell_Rots[r]) for k in range(len(test_seq))]
           new_row  += [sum(seq_count)]
          
       new_row += [new_row.index(max(new_row)) + 1]
       new_row += [test_Labels.loc[i,0]]
       
       if new_row[-2] in walk:
          Class = walk
       else:
          Class = lay
              
       if new_row[-1] in Class:
           Count += 1
       new_row = pd.DataFrame(data = [new_row], columns = Cols)
       Ell_Stats = pd.concat([Ell_Stats, new_row], axis = 0)
       
       
   Ell_Stats.to_csv(Rots_Test + "_Ellipsoid_Stats.csv", index = False) 
    
       
       
 

















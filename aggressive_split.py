# Split Aggressive Data into Individual Users
#------------------------------------------------------------------------------
# Last Updated: 7/30/2018
# Description: Split the Aggressive Data into seperate user files

# Libraries
#------------------------------------------------------------------------------
import pandas as pd
import os
#import csv


# Import Data and Split
#------------------------------------------------------------------------------
Root = "/Users/pauldavid/Documents/My_Files/Applications/Jobs/2018_UNIFYID/Data/"
File = Root + "aggressive_processed.csv"

new_dir = Root + "aggressive_split_data"
if os.path.isdir(new_dir) == 0:
    os.makedirs(new_dir)


Data = pd.read_csv(File)
L = Data.shape[0]

user_data = []
file_names = []
user_names = []
count = 0
j     = 0
names = [None]*2
for i in range(L):
    if i == 0:
        names[0] = Data["user"][0]
        user_names.append(names[0])
    if i > 0:
        names[1] = Data["user"][i]
        if names[0] != names[1]:
            user_data.append(Data[j:i])
            j = i
            file_names.append("aggressive_" + names[0] + ".csv")
            count += 1
            user_names.append(names[0])
        names[0] = names[1]
    if i == L-1:
        user_names.append(names[1])
        user_data.append(Data[j:i])
        file_names.append("aggressive_" + names[0] + ".csv")
        count += 1

# Save Data
#------------------------------------------------------------------------------

for n in range(count):
    user_data[n].to_csv(new_dir + "/" + file_names[n])





        
                
            
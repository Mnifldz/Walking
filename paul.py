import pandas as pd
import find_crossings as fc
import numpy as np
import matplotlib.pyplot as plt
import numVector

def example():
    #example file
    filePath = "/Volumes/Seagate Backup Plus Drive/UnifyID/HMOG/public_dataset/100669/100669_session_15/Gyroscope.csv"


    #read in the important columns of the CSV
    df = pd.read_csv(filePath, header = None, usecols = [0,3, 4, 5], names = ['time', 'x','y','z'])

    #get the time axis
    time = df['time']

    #get ready to collect the matrices
    matrices = []

    #loop through each axis
    for axis in ['x','y','z']:
        #look at the documentation for getVectorsNum: try different thresholds and such for the specific data
        _, matrix, _ = numVector.getVectorsNum(df[axis], time, curveThreshold = 0.1, distThreshold = 0.1)
        matrices.append(matrix)

    #put all the matrices as input to the naiveCross
    crossingsDict = fc.naiveCross(*matrices)


    #I'm plotting here just to demonstrate
    for axis in ['x','y','z']:
        plt.plot(time, df[axis], color = 'black')
    
    for key, color in zip(crossingsDict.keys(), ['red','blue', 'green']):
        for x in crossingsDict[key]:
            plt.axvline(x = x, color = color)

    #plt.show()
    plt.savefig("sample_crossings_HMOG.eps")
    plt.close()
    return  crossingsDict

'''
    radarTestWaves - play with the distance-data from preProcessing
'''

import os
import numpy as np
from radarLib import *
from radarTools import *
from radarTestDefs import *

PLOT_RANGE      = 30    #how much data to plot in s
USE_ALT_DIST    = False
def plotDistData(dataPath:str, savePlot:bool = False):
    if USE_ALT_DIST:
        dataFilename    = dataPath+"/radar_distance_alt.npy" # os.path.join(path,"radar_distance.py")
    else:
        dataFilename    = dataPath+"/radar_distance.npy" # os.path.join(path,"radar_distance.py")

    print("Processing : ", dataFilename)
    distData        = np.load(dataFilename)
    distCount       = distData.shape[0]
    print("> Distance values : ", distCount)

    radarConfig     = importRadarConfig(dataPath)
    framesPerMinute    = radarConfig.frame_rate_Hz * 60
    print("> Frames-per-minute : ", framesPerMinute)

    plotLen = int(radarConfig.frame_rate_Hz * PLOT_RANGE)        # plot data for 2 min
    ofs =  (distCount - plotLen) //2    # for testing get data from the middle
    plotData = distData[ofs:ofs+plotLen,:]

    frames_per_min = radarConfig.frame_rate_Hz * 60
    plotDataLow = plotData.copy()
    plotDataLow[:,1]  = filterData(plotDataLow[:,1], None, 25, frames_per_min, order=4 )

    plotDataHigh = plotData.copy()
    plotDataHigh[:,1] = filterData(plotData[:,1], 30, 70, frames_per_min, order=4 )

    if savePlot:
        # plotDistances(plotData, radarConfig.frame_rate_Hz, 'unfiltered', dataPath)
        plotDistances(plotDataLow, radarConfig.frame_rate_Hz, 'low-rate (1-25 BPM)', dataPath)
        plotDistances(plotDataHigh, radarConfig.frame_rate_Hz, 'high-rate (25-100 BPM)', dataPath)
    else:
        # plotDistances(plotData, radarConfig.frame_rate_Hz, 'unfiltered', None)
        plotDistances(plotDataLow, radarConfig.frame_rate_Hz, 'low-rate (1-25 BPM)', None)
        plotDistances(plotDataHigh, radarConfig.frame_rate_Hz, 'high-rate (25-100 BPM)', None)

if __name__ == '__main__':
    plotDistData(radarFilePathes[4], False)



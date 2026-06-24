'''
    radarTestWaves - play with the distance-data from preProcessing
'''

import os
import numpy as np
import pandas as pd
from radarLib import *
from radarTools import *
from radarTestDefs import *

PLOT_RANGE      = 600    #how much data to plot in s
USE_ALT_PHASE   = True
USE_ALT_DIST    = True
USE_FFT_FILT    = False

# filter frequencies for hibernating hedgehog
MIN_LOW_FREQ_HIB   = 0.5 #for breathing
MAX_LOW_FREQ_HIB    = 10 #should decrease from 14
MIN_HIGH_FREQ_HIB  = 2 #for heart rate
MAX_HIGH_FREQ_HIB  = 25 

#normal, for arousals
NORMAL_LOW_FREQ_MIN   = 10 #for breathing
NORMAL_LOW_FREQ_MAX = 50
NORMAL_HIGH_FREQ_MIN  = 60 #for heart rate
NORMAL_HIGH_FREQ_MAX   = 280

HIB_BREATH_BAND = [MIN_LOW_FREQ_HIB, MAX_LOW_FREQ_HIB]     
HIB_HEART_BAND = [MIN_HIGH_FREQ_HIB, MAX_HIGH_FREQ_HIB]   

AWAKE_BREATH_BAND  = [NORMAL_LOW_FREQ_MIN, NORMAL_LOW_FREQ_MAX] 
AWAKE_HEART_BAND  = [NORMAL_HIGH_FREQ_MIN, NORMAL_HIGH_FREQ_MAX]  

# Define an arousal threshold (variance of displacement)
AROUSAL_THRESHOLD = 0.05 

def plotDistDataChunk(radarConfig:radarConfigData, plotData:np.ndarray, savePlot:bool = False, dataPath:str=None):
    if USE_ALT_DIST:
        name_suffix = ' - alt'
    else:
        name_suffix = ''

    frames_per_min = radarConfig.frame_rate_Hz * 60 
    print(frames_per_min)

    plotDataLow = plotData.copy()
    # bandpass-filter for breathrate
    plotDataLow[:,1]  = filterData(plotDataLow[:,1], HIB_BREATH_BAND[0], HIB_BREATH_BAND[1], frames_per_min, order=4 )
    breathRate = calcRate( plotDataLow[:,1], radarConfig.frame_rate_Hz )
    print("Breathrate : ", breathRate)

    plotDataHigh = plotData.copy()
    # remove harmonics of breathrate
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*2, Q=5, fs=frames_per_min)
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*3, Q=5, fs=frames_per_min)
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*4, Q=5, fs=frames_per_min)
    # filter for heartRate
    plotDataHigh[:,1] = filterData(plotDataHigh[:,1], HIB_HEART_BAND[0], HIB_HEART_BAND[1], frames_per_min, order=4 )
    heartRate = calcRate( plotDataHigh[:,1], radarConfig.frame_rate_Hz )
    print('Heartrate : ', heartRate)

    # plot all data
    if savePlot:
        plotDistances(plotData, radarConfig.frame_rate_Hz, 'full-spectrum'+name_suffix, hibernate = True, data_path = dataPath)
        plotDistances(plotDataLow, radarConfig.frame_rate_Hz, 'low-rate'+name_suffix, hibernate = True, data_path = dataPath)
        plotDistances(plotDataHigh, radarConfig.frame_rate_Hz, 'high-rate'+name_suffix, hibernate = True, data_path = dataPath)
    else:
        plotDistances(plotData, radarConfig.frame_rate_Hz,'full-spectrum', hibernate = True)
        plotDistances(plotDataLow, radarConfig.frame_rate_Hz, 'low-rate', hibernate = True)
        plotDistances(plotDataHigh, radarConfig.frame_rate_Hz, 'high-rate', hibernate = True)


def FFTidx(freq:float, dataLen:int, fs:float):
    return int(freq * dataLen * (1/fs) ) +1


LOW_RATE_MAX    = 60
HIGH_RATE_MIN   = LOW_RATE_MAX
HIGH_RATE_MAX   = 150
MAX_PLOT_BPM    = 200

def plotDistDataChunkFFT(radarConfig:radarConfigData, plotData:np.ndarray, savePlot:bool = False, dataPath:str=None):
    if USE_ALT_DIST:
        name_suffix = ' - alt'
    else:
        name_suffix = ''

    dataWindowed    = plotData[:,1] * sp.signal.get_window(window='blackman', Nx=len(plotData) )
    spectrum        = np.abs(sp.fft.rfft(dataWindowed))
    specFreq        = sp.fft.rfftfreq(len(plotData), 1/(radarConfig.frame_rate_Hz*60))

    maxFFTidx   = FFTidx(MAX_PLOT_BPM, len(plotData), radarConfig.frame_rate_Hz*60)
    spectrum = spectrum[:maxFFTidx]
    specFreq = specFreq[:maxFFTidx]

    spectrumLow     = spectrum.copy()
    minLowIdx          = 2
    maxLowIdx          = FFTidx(LOW_RATE_MAX, len(plotData), radarConfig.frame_rate_Hz*60)
    spectrumLow[:minLowIdx] = 0
    spectrumLow[maxLowIdx:] = 0
    breathIdx        = np.argmax(spectrumLow)
    breathRate       = specFreq[ breathIdx ]
    print("Breathrate = ",breathRate)

    spectrumHigh     = spectrum.copy()
    minIdx          = FFTidx(HIGH_RATE_MIN, len(plotData), radarConfig.frame_rate_Hz*60)+1
    maxIdx          = FFTidx(HIGH_RATE_MAX, len(plotData), radarConfig.frame_rate_Hz*60)
    spectrumHigh[:minIdx] = 0
    spectrumHigh[maxIdx:]  = 0
    # clear harmonics of breathrate
    for i in [1,2,3,4,5,6,7,8]:
        harmIdx = breathIdx*i
        if harmIdx <len(spectrumHigh):
            spectrumHigh[harmIdx] = 0

    heartIdx        = np.argmax(spectrumHigh)
    heartRate       = specFreq[ heartIdx ]
    print("Heartrate = ", heartRate)

    fig, (ax_low, ax_high) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    ax_low.plot(specFreq, spectrumLow, color='purple')
    ax_low.grid(True)
    ax_low.set_title("low-spectrum / breath")
    ax_low.axvline(breathRate, color='purple', linestyle=":")
    ax_low.annotate(f'breathrate:{breathRate:6.2f}', (breathRate,spectrumLow[breathIdx]) )

    ax_high.plot(specFreq, spectrumHigh, color='orange')
    ax_high.grid(True)
    ax_high.set_title("high-spectrum / heart")
    ax_high.axvline(heartRate, color='orange', linestyle=":")
    ax_high.annotate(f'heartrate:{heartRate:6.2f}', (heartRate,spectrumHigh[heartIdx]) )
    fig.suptitle(dataPath)

    if savePlot:
        plt.tight_layout()
        plt.savefig(os.path.join(dataPath, f"spectrum{name_suffix}.png"), dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plotDistData(dataPath:str, savePlot:bool = False):
    if USE_ALT_DIST:
        if USE_ALT_PHASE:
            dataFilename = dataPath + "/radar_distance2_alt.npy"  # os.path.join(path,"radar_distance.py")
        else:
            dataFilename    = dataPath+"/radar_distance_alt.npy" # os.path.join(path,"radar_distance.py")
    else:
        if USE_ALT_PHASE:
            dataFilename = dataPath + "/radar_distance2.npy"  # os.path.join(path,"radar_distance.py")
        else:
            dataFilename    = dataPath+"/radar_distance.npy" # os.path.join(path,"radar_distance.py")

    print("Processing : ", dataFilename)
    distData        = np.load(dataFilename)
    distCount       = distData.shape[0]
    print("> Distance values : ", distCount)

    radarConfig     = importRadarConfig(dataPath)
    framesPerMinute    = radarConfig.frame_rate_Hz * 60
    print("> Frames-per-minute : ", framesPerMinute)

    plotLen = int(radarConfig.frame_rate_Hz * PLOT_RANGE)        # plot data for 30sec
    ofs =  (distCount - plotLen) //2    # for testing get data from the middle
    
    for ofs in range(0, distCount-plotLen, plotLen//2):
        plotData = distData[ofs:ofs+plotLen,:]
        if USE_FFT_FILT:
            plotDistDataChunkFFT(radarConfig, plotData, savePlot, dataPath)
        else:
            plotDistDataChunk(radarConfig, plotData, savePlot, dataPath)



def calcRates(radarConfig:radarConfigData, distData:np.ndarray, breathRange:list, heartRange:list ) :
    """
    calculate breathRate and heartrate from the DistData
    :param radarConfig:
    :param distData:
    :return:    a dict with breathrate and heartrate
    """
    frames_per_min = radarConfig.frame_rate_Hz * 60

    plotDataLow = distData.copy()
    # bandpass-filter for breathrate
    plotDataLow[:,1]  = filterData(plotDataLow[:,1], breathRange[0], breathRange[1], frames_per_min, order=4 )
    breathRate = calcRate( plotDataLow[:,1], radarConfig.frame_rate_Hz )

    plotDataHigh = distData.copy()
    # remove harmonics of breathrate
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*2, Q=5, fs=frames_per_min)
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*3, Q=5, fs=frames_per_min)
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*4, Q=5, fs=frames_per_min)
    # filter for heartRate
    plotDataHigh[:,1] = filterData(plotDataHigh[:,1],heartRange[0], heartRange[1], frames_per_min, order=4 )
    heartRate = calcRate( plotDataHigh[:,1], radarConfig.frame_rate_Hz )
    
    return {'Breathrate':breathRate, 'Heartrate':heartRate }

def calcRatesOverTime(dataPath:str, plotRange:int):
    """
    calculate the breath- and heartrate for different parts of the distanceData
    :param dataPath:    where to find the data
    :param plotRange:   how much time to use for the calculation
    :return:    a list with breathrates and heartrates
    """
    dataFilename = dataPath + "/radar_distance_alt.npy"  # os.path.join(path,"radar_distance.py")
    distData        = np.load(dataFilename)
    distCount       = distData.shape[0]
    radarConfig     = importRadarConfig(dataPath)
    
    rates = []
    rateLen = int(radarConfig.frame_rate_Hz * plotRange)    # how much data to use for calculation
    for ofs in range(0, distCount-rateLen, rateLen//2):
        # select part of data to calculate rates
        rateData = distData[ofs:ofs+rateLen,:]
        #grab timestamps
        window_timestamp = distData[ofs, 0]
        displacement = rateData[:, 1]

        # AROUSAL DETECTION FOR THIS WINDOW 
        window_std = np.std(displacement)
        is_aroused = window_std > AROUSAL_THRESHOLD

        if is_aroused:
            status = "awake/arousal state"
        else:
            status = "hibernating"
            
        # calculate rates from selected data
        breath_heart_rate = calcRates(radarConfig, rateData, HIB_BREATH_BAND, HIB_HEART_BAND)
        # append calculated rates to list
        rates.append({
            "Timestamp_s": window_timestamp,
            "Breathing_Rate_BPM": breath_heart_rate['Breathrate'],
            "Heart_Rate_BPM": breath_heart_rate['Heartrate'],
            "Arousal_Flag": status,
            "Displacement_Std": window_std
            })
    # return list of calculated rates
    return rates


def calcRatesOverTimeAwake(dataPath:str, plotRange:int):
    """
    calculate the breath- and heartrate for different parts of the distanceData
    :param dataPath:    where to find the data
    :param plotRange:   how much time to use for the calculation
    :return:    a list with breathrates and heartrates
    """
    dataFilename = dataPath + "/radar_distance_alt.npy"  # os.path.join(path,"radar_distance.py")
    distData        = np.load(dataFilename)
    distCount       = distData.shape[0]
    radarConfig     = importRadarConfig(dataPath)
    
    awake_rates = []
    rateLen = int(radarConfig.frame_rate_Hz * plotRange)    # how much data to use for calculation
    for ofs in range(0, distCount-rateLen, rateLen//2):
        # select part of data to calculate rates
        rateData = distData[ofs:ofs+rateLen,:]

        #grab timestamps
        window_timestamp = distData[ofs, 0]

        # calculate rates from selected data
        breath_heart_rate = calcRates(radarConfig, rateData, AWAKE_BREATH_BAND, AWAKE_HEART_BAND)
        # append calculated rates to list
        awake_rates.append({
            "Timestamp_s": window_timestamp,
            "Breathing_Rate_BPM": breath_heart_rate['Breathrate'],
            "Heart_Rate_BPM": breath_heart_rate['Heartrate']
       })
    # return list of calculated rates
    return awake_rates


if __name__ == '__main__':
    rates = calcRatesOverTime(radarFilePaths[1], PLOT_RANGE) 
    rates_df = pd.DataFrame.from_dict(rates) 
    rates_df.to_csv(radarFilePaths[1] + "rates.csv") #save rates to a csv

    awake_rates = calcRatesOverTimeAwake(radarFilePaths[1], PLOT_RANGE) 
    awake_rates_df = pd.DataFrame.from_dict(awake_rates) 
    awake_rates_df.to_csv(radarFilePaths[1] + "awake_rates.csv") #save rates to a csv 
    #plotDistData(radarFilePaths[1], True) 
    #plotDistData(radarFilePaths[4], True) #which of the two should be plotted
   # plotDistData(radarFilePaths[6], True)
    # for path in radarFilePaths: plotDistData(path, True)




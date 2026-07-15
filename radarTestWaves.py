'''
    radarTestWaves - play with the distance-data from preProcessing
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from radarLib import *
from radarTools import *
from radarTestDefs import *

PLOT_RANGE      = 30   #how much data to plot in s
PLOT_RANGE_HIB = 180
USE_ALT_PHASE   = True
USE_ALT_DIST    = True
USE_FFT_FILT    = False 

# filter frequencies for hibernating hedgehog
MIN_LOW_FREQ_HIB   = 0.2 #for breathing
MAX_LOW_FREQ_HIB    = 5 #should decrease from 14
MIN_HIGH_FREQ_HIB  = 2 #for heart rate
MAX_HIGH_FREQ_HIB  = 20 

#normal, for dog or arousals
NORMAL_LOW_FREQ_MIN   = 10 #for breathing
NORMAL_LOW_FREQ_MAX = 80
NORMAL_HIGH_FREQ_MIN  = 120 #for heart rate
NORMAL_HIGH_FREQ_MAX   = 250 

HIB_BREATH_BAND = [MIN_LOW_FREQ_HIB, MAX_LOW_FREQ_HIB]     
HIB_HEART_BAND = [MIN_HIGH_FREQ_HIB, MAX_HIGH_FREQ_HIB]   

AWAKE_BREATH_BAND  = [NORMAL_LOW_FREQ_MIN, NORMAL_LOW_FREQ_MAX] 
AWAKE_HEART_BAND  = [NORMAL_HIGH_FREQ_MIN, NORMAL_HIGH_FREQ_MAX]  


def plotDistDataChunk(radarConfig:radarConfigData, plotData:np.ndarray, savePlot:bool = False, dataPath:str=None):
    if USE_ALT_DIST:
        name_suffix = ' - alt'
    else:
        name_suffix = ''

    frames_per_min = int(radarConfig.frame_rate_Hz * 60)

    plotDataLow = plotData.copy()
    # bandpass-filter for breathrate
    plotDataLow[:,1]  = filterData(plotDataLow[:,1], AWAKE_BREATH_BAND[0], AWAKE_BREATH_BAND[1], frames_per_min, order=4 )
    breathRate = calcRate( plotDataLow[:,1], radarConfig.frame_rate_Hz )
    print("Breathrate : ", breathRate)

    plotDataHigh = plotData.copy()
    # remove harmonics of breathrate
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*2, Q=5, fs=frames_per_min)
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*3, Q=5, fs=frames_per_min)
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*4, Q=5, fs=frames_per_min)
    # filter for heartRate
    plotDataHigh[:,1] = filterData(plotDataHigh[:,1], AWAKE_HEART_BAND[0], AWAKE_HEART_BAND[1], frames_per_min, order=4 )
    heartRate = calcRate(plotDataHigh[:,1], radarConfig.frame_rate_Hz )
    print('Heartrate : ', heartRate)

    combinedData = np.column_stack((plotData[:, 0], plotDataLow[:, 1], plotDataHigh[:, 1])) #to plot both together
    
    # plot all data
    #
    if savePlot:
       plotDistances(plotData, radarConfig.frame_rate_Hz, 'full-spectrum'+name_suffix,  data_path = dataPath)
       plotDistances(plotDataLow, radarConfig.frame_rate_Hz, 'low-rate'+name_suffix,  data_path = dataPath)
       plotDistances(plotDataHigh, radarConfig.frame_rate_Hz, 'high-rate'+name_suffix,  data_path = dataPath)
       plotDistances(combinedData, radarConfig.frame_rate_Hz,"Arousal State: Hedgehog Heart and Respiration Rates"+name_suffix, dataColumns=[1, 2], legend=["Respiration", "Heart"], data_path=dataPath)

    else:
        plotDistances(plotData, radarConfig.frame_rate_Hz,'full-spectrum')
        plotDistances(plotDataLow, radarConfig.frame_rate_Hz, 'low-rate')
        plotDistances(plotDataHigh, radarConfig.frame_rate_Hz, 'high-rate')
        plotDistances(combinedData, radarConfig.frame_rate_Hz, 'heart-respiration', dataColumns=[1,2], legend=["Respiration","Heart"])
    

def FFTidx(freq:float, dataLen:int, fs:float):
    return int(freq * dataLen * (1/fs) ) +1


LOW_RATE_MAX    = 60
HIGH_RATE_MIN   = LOW_RATE_MAX
HIGH_RATE_MAX   = 150

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



def plotDistData(dataPath: str, savePlot: bool = False):
    if USE_ALT_DIST:
        if USE_ALT_PHASE:
            dataFilename = os.path.join(dataPath, "radar_distance2_alt.npy")
        else:
            dataFilename = os.path.join(dataPath, "radar_distance_alt.npy")
    else:
        if USE_ALT_PHASE:
            dataFilename = os.path.join(dataPath, "radar_distance2.npy")

        else:
            dataFilename = os.path.join(dataPath, "radar_distance.npy")

    print("Processing : ", dataFilename)
    distData = np.load(dataFilename)

    
    radarConfig = importRadarConfig(dataPath)

    frames_to_skip = int(10.0 * radarConfig.frame_rate_Hz)
    
    if distData.shape[0] > frames_to_skip:
        distData = distData[frames_to_skip:, :] # Slice off the start
    else:
        print(f"Warning: File is shorter than 10s. Plotting as-is.")
    
    distCount = distData.shape[0] #9565
    print("> Distance values : ", distCount)

    plotLen = int(radarConfig.frame_rate_Hz * PLOT_RANGE)        
    ofs =  (distCount - plotLen) //2    # for testing get data from the middle

    # Run the sliding window loop safely
    for ofs in range(ofs, distCount - plotLen, plotLen // 2):
        plotData = distData[ofs : ofs + plotLen, :]
        
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
    frames_per_min = int(radarConfig.frame_rate_Hz * 60)

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
    dataFilename = dataPath + "/radar_distance_alt.npy"  # os.path.join(path,"radar_distance_alt.npy")
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

        window_std = np.std(displacement)
        
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
    dataFilename = os.path.join(dataPath,"radar_distance_alt.npy")
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
        displacement = rateData[:, 1]

        #DETECTION FOR THIS WINDOW 
        window_std = np.std(displacement)

        status = "awake/arousal state"
        
        # calculate rates from selected data
        breath_heart_rate = calcRates(radarConfig, rateData, AWAKE_BREATH_BAND, AWAKE_HEART_BAND)

        # append calculated rates to list
        awake_rates.append({
            "Timestamp_s": window_timestamp,
            "Breathing_Rate_BPM": breath_heart_rate['Breathrate'],
            "Heart_Rate_BPM": breath_heart_rate['Heartrate'],
            "Arousal_Flag": status,
            "Displacement_Std": window_std
       })
    # return list of calculated rates
    return awake_rates



#doggie so no hibernating rates
# if __name__ == '__main__':

#     awake_rates = calcRatesOverTimeAwake(radarFilePaths3[0], PLOT_RANGE) 
#     awake_rates_df = pd.DataFrame.from_dict(awake_rates) 
#     awake_rates_csv_path = os.path.join(radarFilePaths3[0], "awake_rates.csv")
#     awake_rates_df.to_csv(awake_rates_csv_path)

#     plotDistData(radarFilePaths3[0], True) 
    #plotDistData(radarFilePaths[4], True) #which of the two should be plotted
    # plotDistData(radarFilePaths[6], True)
    # for path in radarFilePaths: plotDistData(path, True)



##hedgehog is awake so no hibernating rates (and no timestamps)
##Loop through every directory path in list
for data_path in radarFilePaths2:
#     print(f"\nAnalyzing awake rates for: {data_path}")
        
    awake_rates = calcRatesOverTimeAwake(data_path, PLOT_RANGE) 
    awake_rates_df = pd.DataFrame.from_dict(awake_rates) 
    awake_rates_csv_path = os.path.join(data_path, "awake_rates.csv")
    awake_rates_df.to_csv(awake_rates_csv_path, index=False) 
    plotDistData(data_path, True)

# loop for hibernating hedgehogs, includes timestamps

# for data_path in radarFilePaths1:
#     rates = calcRatesOverTime(data_path, PLOT_RANGE_HIB) 
#     rates_df = pd.DataFrame.from_dict(rates) 
#     rates_df["Timestamp_UTC"] = pd.to_datetime(rates_df["Timestamp_s"], unit="s", utc=True)
#     rates_df.to_csv(data_path + "rates.csv") 
#     plotDistData(data_path, True)


#arousal loop
for data_path in radarFilePaths1:
    awake_rates = calcRatesOverTimeAwake(data_path, PLOT_RANGE) 
    awake_rates_df = pd.DataFrame.from_dict(awake_rates) 
    awake_rates_df["Timestamp_UTC"] = pd.to_datetime(awake_rates_df["Timestamp_s"], unit="s", utc=True)
    awake_rates_df.to_csv(data_path + "awake_rates.csv")  
    plotDistData(data_path, True)


## plot both lines at the same time
#30 seconds for dog ()
#representative data - 3 arousals, 3 hibernating, 1 awake (hedgehog)
#model includes all ()



# if __name__ == '__main__':

#     rates = calcRatesOverTime(radarFilePaths4[0], PLOT_RANGE) 
#     rates_df = pd.DataFrame.from_dict(rates) 
#     rates_csv_path = os.path.join(radarFilePaths4[0], "rates.csv")
#     rates_df.to_csv(rates_csv_path)

#     plotDistData(radarFilePaths4[0], True)


# if __name__ == '__main__':

#     awake_rates = calcRatesOverTimeAwake(radarFilePaths4[0], PLOT_RANGE) 
#     awake_rates_df = pd.DataFrame.from_dict(rates) 
#     awake_rates_csv_path = os.path.join(radarFilePaths4[0], "awake_rates.csv")
#     awake_rates_df.to_csv(awake_rates_csv_path)

#     plotDistData(radarFilePaths4[0], True)



# if __name__ == '__main__':

#     awake_rates = calcRatesOverTimeAwake(radarFilePaths4[1], PLOT_RANGE) 
#     awake_rates_df = pd.DataFrame.from_dict(awake_rates) 
#     awake_rates_csv_path = os.path.join(radarFilePaths4[1], "awake_rates.csv")
#     awake_rates_df.to_csv(awake_rates_csv_path)
   
#     plotDistData(radarFilePaths4[1], True)


    
#updated processing ??

# Loop for hibernating hedgehogs (5-minute averaging)

for data_path in radarFilePaths1:
    rates = calcRatesOverTime(data_path, PLOT_RANGE_HIB) 
    rates_df = pd.DataFrame.from_dict(rates) 
    
    # Create the Datetime column
    rates_df["Timestamp_UTC"] = pd.to_datetime(rates_df["Timestamp_s"], unit="s", utc=True)
    
    # Set the Datetime column as the index (required for resampling)
    rates_df.set_index("Timestamp_UTC", inplace=True)
    
    # Specify only the numeric columns to average 
    numeric_cols = ["Breathing_Rate_BPM", "Heart_Rate_BPM", "Displacement_Std"]
    
    # Resample to 5 minutes and calculate  mean
    rates_5min_df = rates_df[numeric_cols].resample('5min').mean()
    
    # Save 
    rates_5min_df.to_csv(data_path + "rates_5min.csv") 
    
    plotDistData(data_path, True)


# Arousal loop 

for data_path in radarFilePaths2:
    awake_rates = calcRatesOverTimeAwake(data_path, PLOT_RANGE) 
    awake_rates_df = pd.DataFrame.from_dict(awake_rates) 
    
    # Create the Datetime column
    awake_rates_df["Timestamp_UTC"] = pd.to_datetime(awake_rates_df["Timestamp_s"], unit="s", utc=True)
    
    # Set Datetime as index
    awake_rates_df.set_index("Timestamp_UTC", inplace=True)
    
    # Specify numeric columns
    numeric_cols = ["Breathing_Rate_BPM", "Heart_Rate_BPM", "Displacement_Std"]
    
    # Resample to 1 minute for awake data (to catch faster variations)
    awake_rates_1min_df = awake_rates_df[numeric_cols].resample('1min').mean()
    
    # Save
    awake_rates_1min_df.to_csv(data_path + "awake_rates_1min.csv")  
    
    plotDistData(data_path, True)





    
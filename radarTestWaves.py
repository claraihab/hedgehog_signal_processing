'''
    radarTestWaves - play with the distance-data from preProcessing
'''

import os
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from radarLib import *
from radarTools import *
from radarTestDefs import *

PLOT_RANGE      = 30   #how much data to plot in s (only for arousals, dog)
PLOT_RANGE_HIB = 300
PLOT_RANGE_HIB_Heart = 60

USE_ALT_PHASE   = True
USE_ALT_DIST    = True
USE_FFT_FILT    = False 

# filter frequencies for hibernating hedgehog
MIN_LOW_FREQ_HIB   = 0.2 #for breathing
MAX_LOW_FREQ_HIB    = 3 #should decrease from 14
MIN_HIGH_FREQ_HIB  = 4 #for heart rate
MAX_HIGH_FREQ_HIB  = 17

#normal, for dog or arousals
NORMAL_LOW_FREQ_MIN   = 25 #for breathing
NORMAL_LOW_FREQ_MAX = 50
NORMAL_HIGH_FREQ_MIN  = 180 #for heart rate
NORMAL_HIGH_FREQ_MAX   = 280 

HIB_BREATH_BAND = [MIN_LOW_FREQ_HIB, MAX_LOW_FREQ_HIB]     
HIB_HEART_BAND = [MIN_HIGH_FREQ_HIB, MAX_HIGH_FREQ_HIB]   

AWAKE_BREATH_BAND  = [NORMAL_LOW_FREQ_MIN, NORMAL_LOW_FREQ_MAX] 
AWAKE_HEART_BAND  = [NORMAL_HIGH_FREQ_MIN, NORMAL_HIGH_FREQ_MAX]  

# Plot-spectrum x-axis cutoff (in 1/min). HIB bands top out at 20 bpm so the default
# MAX_PLOT_BPM=70 (from radarTools) is fine there. Awake heart rate goes up to 280 bpm,
# so the awake run below passes a higher max_plot_bpm explicitly.
AWAKE_MAX_PLOT_BPM = 300


def plotDistDataChunk(radarConfig:radarConfigData, plotData:np.ndarray, breathBand:list, heartBand:list,
                      savePlot:bool = False, dataPath:str=None, max_plot_bpm:float=None, title:str="Heart and Respiration Rates"):
    if USE_ALT_DIST:
        name_suffix = ' - alt'
    else:
        name_suffix = ''

    frames_per_min = int(radarConfig.frame_rate_Hz * 60)

    plotDataLow = plotData.copy()
    # bandpass-filter for breathrate
    plotDataLow[:,1]  = filterData(plotDataLow[:,1], breathBand[0], breathBand[1], frames_per_min, order=4 )
    breathRate = calcRate( plotDataLow[:,1], radarConfig.frame_rate_Hz )
    print("Breathrate : ", breathRate)

    plotDataHigh = plotData.copy()
    # remove harmonics of breathrate
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*2, Q=5, fs=frames_per_min)
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*3, Q=5, fs=frames_per_min)
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*4, Q=5, fs=frames_per_min)
    # filter for heartRate
    plotDataHigh[:,1] = filterData(plotDataHigh[:,1], heartBand[0], heartBand[1], frames_per_min, order=4 )
    heartRate = calcRate(plotDataHigh[:,1], radarConfig.frame_rate_Hz )
    print('Heartrate : ', heartRate)

    combinedData = np.column_stack((plotData[:, 0], plotDataLow[:, 1], plotDataHigh[:, 1])) #to plot both together
    
    # plot all data
    
    if savePlot:
       plotDistances(plotData, radarConfig.frame_rate_Hz, 'full-spectrum'+name_suffix,  data_path = dataPath, max_plot_bpm=max_plot_bpm)
       plotDistances(plotDataLow, radarConfig.frame_rate_Hz, 'low-rate'+name_suffix,  data_path = dataPath, max_plot_bpm=max_plot_bpm)
       plotDistances(plotDataHigh, radarConfig.frame_rate_Hz, 'high-rate'+name_suffix,  data_path = dataPath, max_plot_bpm=max_plot_bpm)
       #plotDistances(combinedData, radarConfig.frame_rate_Hz, title+name_suffix, dataColumns=[1, 2], legend=["Respiration", "Heart"], data_path=dataPath, max_plot_bpm=max_plot_bpm) #will not work

    else:
        plotDistances(plotData, radarConfig.frame_rate_Hz,'full-spectrum', max_plot_bpm=max_plot_bpm)
        plotDistances(plotDataLow, radarConfig.frame_rate_Hz, 'low-rate', max_plot_bpm=max_plot_bpm)
        plotDistances(plotDataHigh, radarConfig.frame_rate_Hz, 'high-rate', max_plot_bpm=max_plot_bpm)
        #plotDistances(combinedData, radarConfig.frame_rate_Hz, 'heart-respiration', dataColumns=[1,2], legend=["Respiration","Heart"], max_plot_bpm=max_plot_bpm)
    

def FFTidx(freq:float, dataLen:int, fs:float):
    return int(freq * dataLen * (1/fs) ) +1


def plotDistDataChunkFFT(radarConfig:radarConfigData, plotData:np.ndarray, breathRange:list, heartRange:list,
                          savePlot:bool = False, dataPath:str=None, max_plot_bpm:float=None):
    if max_plot_bpm is None:
        max_plot_bpm = MAX_PLOT_BPM
    if USE_ALT_DIST:
        name_suffix = ' - alt'
    else:
        name_suffix = ''

    dataWindowed    = plotData[:,1] * sp.signal.get_window(window='blackman', Nx=len(plotData) )
    spectrum        = np.abs(sp.fft.rfft(dataWindowed))
    specFreq        = sp.fft.rfftfreq(len(plotData), 1/(radarConfig.frame_rate_Hz*60))

    maxFFTidx   = FFTidx(max_plot_bpm, len(plotData), radarConfig.frame_rate_Hz*60)
    spectrum = spectrum[:maxFFTidx]
    specFreq = specFreq[:maxFFTidx]

    spectrumLow     = spectrum.copy()
    minLowIdx          = 2
    maxLowIdx          = FFTidx(breathRange[1], len(plotData), radarConfig.frame_rate_Hz*60)
    spectrumLow[:minLowIdx] = 0
    spectrumLow[maxLowIdx:] = 0
    breathIdx        = np.argmax(spectrumLow)
    breathRate       = specFreq[ breathIdx ]
    print("Breathrate = ",breathRate)

    spectrumHigh     = spectrum.copy()
    minIdx          = FFTidx(heartRange[0], len(plotData), radarConfig.frame_rate_Hz*60)+1
    maxIdx          = FFTidx(heartRange[1], len(plotData), radarConfig.frame_rate_Hz*60)
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



def plotDistData(dataPath: str, breathBand:list, heartBand:list, plotRange:int, savePlot: bool = False, max_plot_bpm:float=None, title:str="Heart and Respiration Rates"):
    dataFilename = getDistFilename(dataPath, USE_ALT_PHASE, USE_ALT_DIST)

    print("Processing : ", dataFilename)
    distData = np.load(dataFilename)

    
    radarConfig = importRadarConfig(dataPath)

    frames_to_skip = int(15.0 * radarConfig.frame_rate_Hz)
    
    if distData.shape[0] > frames_to_skip:
        distData = distData[frames_to_skip:, :] # Slice off the start
    else:
        print(f"Warning: File is shorter than 10s. Plotting as-is.")
    
    distCount = distData.shape[0] #9565
    print("> Distance values : ", distCount)

    plotLen = int(radarConfig.frame_rate_Hz * plotRange)
    ofs =  (distCount - plotLen) //2    # for testing get data from the middle

    # Run the sliding window loop safely
    for ofs in range(ofs, distCount - plotLen, plotLen // 2):
        plotData = distData[ofs : ofs + plotLen, :]
        
        if USE_FFT_FILT:
            plotDistDataChunkFFT(radarConfig, plotData, breathBand, heartBand, savePlot, dataPath, max_plot_bpm=max_plot_bpm)
        else:
            plotDistDataChunk(radarConfig, plotData, breathBand, heartBand, savePlot, dataPath, max_plot_bpm=max_plot_bpm, title=title)



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
    breathResult = calcRateSNR( plotDataLow[:,1], radarConfig.frame_rate_Hz, n_harmonics=4,
                                 min_freq=max(0.01, breathRange[0]*0.5), max_freq=breathRange[1]*1.5 )
    breathRate = breathResult["rate"]
    breathStd = np.std(plotDataLow[:,1]) #Standard Deviation of Respiration Signal

    plotDataHigh = distData.copy()
    # remove harmonics of breathrate
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*2, Q=5, fs=frames_per_min)
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*3, Q=5, fs=frames_per_min)
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*4, Q=5, fs=frames_per_min)
    # filter for heartRate
    plotDataHigh[:,1] = filterData(plotDataHigh[:,1],heartRange[0], heartRange[1], frames_per_min, order=4 )
    heartResult = calcRateSNR( plotDataHigh[:,1], radarConfig.frame_rate_Hz, n_harmonics=4,
                                min_freq=heartRange[0]*0.5, max_freq=heartRange[1]*1.5 )
    heartRate = heartResult["rate"]
    heartStd = np.std(plotDataHigh[:,1]) #Standard Deviation of Heart Signal

    return {
        'Breathrate': breathRate, 
        'Heartrate': heartRate,
        'BreathStd': breathStd,
        'HeartStd': heartStd,
        'Breath_SNR_dB': breathResult["snr_dB"],
        'Heart_SNR_dB': heartResult["snr_dB"],
        'Breath_Prominence': breathResult["prominence_ratio"],
        'Heart_Prominence': heartResult["prominence_ratio"],
    }


def getDistFilename(dataPath:str, useAltPhase:bool, useAltDist:bool) -> str:
    '''
    build the distance-data filename using the same naming convention as plotDistData(),
    so every consumer of these files (plots, rates.csv) agrees on which phase/distance
    calculation method the numbers came from.
    '''
    if useAltDist:
        if useAltPhase:
            name = "radar_distance2_alt.npy"
        else:
            name = "radar_distance_alt.npy"
    else:
        if useAltPhase:
            name = "radar_distance2.npy"
        else:
            name = "radar_distance.npy"
    return os.path.join(dataPath, name)


def calcRatesOverTime(dataPath:str, plotRange:int):
    """
    calculate the breath- and heartrate for different parts of the distanceData, using the
    hibernating-torpor bands (HIB_BREATH_BAND/HIB_HEART_BAND) for every window.
    :param dataPath:    where to find the data
    :param plotRange:   how much time to use for the calculation
    :return:    a list with breathrates and heartrates
    """
    dataFilename    = getDistFilename(dataPath, USE_ALT_PHASE, USE_ALT_DIST)
    distData        = np.load(dataFilename)
    distCount       = distData.shape[0]
    radarConfig     = importRadarConfig(dataPath)

    rates = []
    rateLen = int(radarConfig.frame_rate_Hz * plotRange)    # how much data to use for calculation
    if distCount < rateLen:
        print(f"   WARNING: recording has {distCount} frames, shorter than one "
              f"{plotRange}s window ({rateLen} frames needed) - skipping, 0 rate windows produced.")
        return rates
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
            "Breathing_Std": breath_heart_rate['BreathStd'],      # Added 
            "Heart_Std": breath_heart_rate['HeartStd'],          
            "Breath_SNR_dB": breath_heart_rate['Breath_SNR_dB'],
            "Heart_SNR_dB": breath_heart_rate['Heart_SNR_dB'],
            "Breath_Prominence": breath_heart_rate['Breath_Prominence'],
            "Heart_Prominence": breath_heart_rate['Heart_Prominence'],
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
    dataFilename    = getDistFilename(dataPath, USE_ALT_PHASE, USE_ALT_DIST)
    distData        = np.load(dataFilename)
    distCount       = distData.shape[0]
    radarConfig     = importRadarConfig(dataPath)
    
    awake_rates = []
    rateLen = int(radarConfig.frame_rate_Hz * plotRange)    # how much data to use for calculation
    if distCount < rateLen:
        print(f"   WARNING: recording has {distCount} frames, shorter than one "
              f"{plotRange}s window ({rateLen} frames needed) - skipping, 0 rate windows produced.")
        return awake_rates
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
            "Breathing_Std": breath_heart_rate['BreathStd'],     
            "Heart_Std": breath_heart_rate['HeartStd'],        
            "Breath_SNR_dB": breath_heart_rate['Breath_SNR_dB'],
            "Heart_SNR_dB": breath_heart_rate['Heart_SNR_dB'],
            "Breath_Prominence": breath_heart_rate['Breath_Prominence'],
            "Heart_Prominence": breath_heart_rate['Heart_Prominence'],
            "Arousal_Flag": status,
            "Displacement_Std": window_std
       })
    # return list of calculated rates
    return awake_rates


def runHibernatingAnalysis(data_paths:list = None):
    '''
    run rate-extraction + plotting for hibernating recordings, using HIB_BREATH_BAND/
    HIB_HEART_BAND and PLOT_RANGE_HIB (5-min windows). Writes rates.csv + plots into
    each recording's own folder.
    '''
    if data_paths is None:
        data_paths = radarFilePaths1

    for data_path in data_paths:
        print(f"\nProcessing (hibernating config): {data_path}")
        try:
            rates = calcRatesOverTime(data_path, PLOT_RANGE_HIB)
            if len(rates) == 0:
                print(f"   Skipping {data_path} - no rate windows produced (recording too short).")
                continue
            rates_df = pd.DataFrame.from_dict(rates)
            rates_df["Timestamp_UTC"] = pd.to_datetime(rates_df["Timestamp_s"], unit="s", utc=True)
            rates_df.to_csv(os.path.join(data_path, "rates.csv"), index=False)
            plotDistData(data_path, HIB_BREATH_BAND, HIB_HEART_BAND, PLOT_RANGE_HIB, savePlot=True,
                         title="Hibernating Hedgehog Heart and Respiration Rates")
        except Exception:
            print("Error processing dir:", data_path)
            traceback.print_exc()


def runAwakeAnalysis(data_paths:list = None):
    '''
    run rate-extraction + plotting for awake/active recordings, using AWAKE_BREATH_BAND/
    AWAKE_HEART_BAND, PLOT_RANGE (30s windows), and a wider spectrum-plot cutoff so the
    180-280 bpm heart-rate peak isn't clipped off the plot. Writes awake_rates.csv + plots
    into each recording's own folder.
    '''
    if data_paths is None:
        data_paths = radarFilePaths1 #changed to run the arousal

    for data_path in data_paths:
        print(f"\nProcessing (awake config): {data_path}")
        try:
            awake_rates = calcRatesOverTimeAwake(data_path, PLOT_RANGE)
            if len(awake_rates) == 0:
                print(f"   Skipping {data_path} - no rate windows produced (recording too short).")
                continue
            awake_rates_df = pd.DataFrame.from_dict(awake_rates)
            awake_rates_df["Timestamp_UTC"] = pd.to_datetime(awake_rates_df["Timestamp_s"], unit="s", utc=True)
            awake_rates_df.to_csv(os.path.join(data_path, "awake_rates.csv"), index=False)
            plotDistData(data_path, AWAKE_BREATH_BAND, AWAKE_HEART_BAND, PLOT_RANGE, savePlot=True,
                         max_plot_bpm=AWAKE_MAX_PLOT_BPM, title="Awake Hedgehog Heart and Respiration Rates")
        except Exception:
            print("Error processing dir:", data_path)
            traceback.print_exc()


def buildComparisonCsv(data_path:str):
    '''
    merge the hibernating-band (rates.csv) and awake-band (awake_rates.csv) estimates for
    the same recording into one CSV, aligned by nearest timestamp. Both
    candidate heart/breath-rate estimates for any given moment side by side, then use 
    CO2-confirmed arousal timestamps to decide - per window - which estimate is the real one.
    '''
    hib_path   = os.path.join(data_path, "rates.csv")
    awake_path = os.path.join(data_path, "awake_rates.csv")
    if not (os.path.exists(hib_path) and os.path.exists(awake_path)):
        print(f"Skipping comparison csv for {data_path} - need both rates.csv and awake_rates.csv present")
        return

    hib_df   = pd.read_csv(hib_path).add_prefix("Hib_").rename(columns={"Hib_Timestamp_s": "Timestamp_s"})
    awake_df = pd.read_csv(awake_path).add_prefix("Awake_").rename(columns={"Awake_Timestamp_s": "Timestamp_s"})

    hib_df   = hib_df.sort_values("Timestamp_s")
    awake_df = awake_df.sort_values("Timestamp_s")

    # awake windows (30s) are much finer than hibernating windows (5min), so align each
    # awake row to its nearest hibernating window rather than trying to force equal timestamps
    merged = pd.merge_asof(awake_df, hib_df, on="Timestamp_s", direction="nearest")
    merged["Timestamp_UTC"] = pd.to_datetime(merged["Timestamp_s"], unit="s", utc=True)
    merged.to_csv(os.path.join(data_path, "rates_comparison.csv"), index=False)
    print(f"Wrote rates_comparison.csv for {data_path}")


def runBothAnalyses(data_paths:list = None):
    '''
    for recordings that may contain both torpor and arousal bouts (e.g. radarFilePaths1),
    run BOTH the hibernating-band and awake-band analyses on the same files, then merge
    the two into rates_comparison.csv per recording. Use  CO2 log afterward to pick
    which set of numbers (Hib_* vs Awake_*) is correct for each time window.
    '''
    if data_paths is None:
        data_paths = radarFilePaths1

    runHibernatingAnalysis(data_paths)
    runAwakeAnalysis(data_paths)

    for data_path in data_paths:
        try:
            buildComparisonCsv(data_path)
        except Exception:
            print("Error building comparison csv for dir:", data_path)
            traceback.print_exc()



#doggie so no hibernating rates
# if __name__ == '__main__':

#     awake_rates = calcRatesOverTimeAwake(radarFilePaths3[0], PLOT_RANGE) 
#     awake_rates_df = pd.DataFrame.from_dict(awake_rates) 
#     awake_rates_csv_path = os.path.join(radarFilePaths3[0], "awake_rates.csv")
#     awake_rates_df.to_csv(awake_rates_csv_path)
#     plotDistData(radarFilePaths3[0], True) 



if __name__ == '__main__':
    # runs both band configs on radarFilePaths1 (files may contain both torpor and arousal
    # bouts), producing rates.csv, awake_rates.csv, and rates_comparison.csv per recording.
    # Use runHibernatingAnalysis()/runAwakeAnalysis() individually.

    runAwakeAnalysis()




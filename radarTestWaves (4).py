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

USE_ALT_PHASE   = True
USE_ALT_DIST    = True
USE_FFT_FILT    = False 

# filter frequencies for hibernating hedgehog
MIN_LOW_FREQ_HIB   = 0.4 #for breathing 
MAX_LOW_FREQ_HIB    = 3
MIN_HIGH_FREQ_HIB  = 2 #for heart rate 
MAX_HIGH_FREQ_HIB  = 30


#for arousals
NORMAL_LOW_FREQ_MIN   = 25 #for breathing 
NORMAL_LOW_FREQ_MAX = 50
NORMAL_HIGH_FREQ_MIN  = 180 #for heart rate
NORMAL_HIGH_FREQ_MAX   = 280 

#for dog 
DOG_LOW_MIN = 10
DOG_LOW_MAX = 30
DOG_HIGH_MIN = 60
DOG_HIGH_MAX = 120

#for active hedgehog 
ACTIVE_LOW_MIN = 10
ACTIVE_LOW_MAX = 60
ACTIVE_HIGH_MIN = 90
ACTIVE_HIGH_MAX = 250

HIB_BREATH_BAND = [MIN_LOW_FREQ_HIB, MAX_LOW_FREQ_HIB]     
HIB_HEART_BAND = [MIN_HIGH_FREQ_HIB, MAX_HIGH_FREQ_HIB]   

#arousal
AWAKE_BREATH_BAND  = [NORMAL_LOW_FREQ_MIN, NORMAL_LOW_FREQ_MAX] 
AWAKE_HEART_BAND  = [NORMAL_HIGH_FREQ_MIN, NORMAL_HIGH_FREQ_MAX]

#for  active hedgehog 
ACTIVE_BREATH_BAND  = [ACTIVE_LOW_MIN, ACTIVE_LOW_MAX] 
ACTIVE_HEART_BAND  = [ACTIVE_HIGH_MIN, ACTIVE_HIGH_MAX]

#for dog 
DOG_BREATH_BAND  = [DOG_LOW_MIN, DOG_LOW_MAX] 
DOG_HEART_BAND  = [DOG_HIGH_MIN, DOG_HIGH_MAX]

##plot spectrum maximum
MAX_PLOT_BPM   = 35
AWAKE_MAX_PLOT_BPM = 300
ACTIVE_MAX_PLOT_BPM = 300 
DOG_MAX_PLOT_BPM = 150 


def plotDistDataChunk(radarConfig:radarConfigData, plotData:np.ndarray, breathBand:list, heartBand:list,
                      savePlot:bool = False, dataPath:str=None, max_plot_bpm:float=None, title:str="Heart and Respiration Rates"):
    if USE_ALT_DIST:
        name_suffix = ' - alt'
    else:
        name_suffix = ''

    frames_per_min = int(radarConfig.frame_rate_Hz * 60)

    plotDataLow = plotData.copy()
    # bandpass-filter for breathrate
    plotDataLow[:,1] = filterData(plotDataLow[:,1], breathBand[0], breathBand[1], frames_per_min, order=4)
    breathResult = calcRateSNR3(
        plotDataLow[:,1], radarConfig.frame_rate_Hz, n_harmonics=4,
        min_freq= breathBand[0], max_freq=breathBand[1] 
    )
    breathRate = breathResult["rate"]
    print(f"Breathrate : {breathRate:.2f} bpm  SNR={breathResult['snr_dB']:.1f}dB  prom={breathResult['prominence_ratio']:.2f}")

    plotDataHigh = plotData.copy()
    # remove harmonics of breathrate
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*2, Q=5, fs=frames_per_min)
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*3, Q=5, fs=frames_per_min)
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*4, Q=5, fs=frames_per_min)
    # filter for heartRate
    plotDataHigh[:,1] = filterData(plotDataHigh[:,1], heartBand[0], heartBand[1], frames_per_min, order=4)
    heartResult = calcRateSNR3(
        plotDataHigh[:,1], radarConfig.frame_rate_Hz, n_harmonics=4,
        min_freq=heartBand[0], max_freq=heartBand[1]
    )
    heartRate = heartResult["rate"]
    print(f"Heartrate : {heartRate:.2f} bpm  SNR={heartResult['snr_dB']:.1f}dB  prom={heartResult['prominence_ratio']:.2f}")

    combinedData = np.column_stack((plotData[:, 0], plotDataLow[:, 1], plotDataHigh[:, 1]))  # to plot both together


    # labels now carry rate + SNR + prominence so the panel titles show them
    low_label  = f"low-rate{name_suffix} - {breathRate:.2f}bpm SNR={breathResult['snr_dB']:.1f}dB prom={breathResult['prominence_ratio']:.2f}"
    high_label = f"high-rate{name_suffix} - {heartRate:.2f}bpm SNR={heartResult['snr_dB']:.1f}dB prom={heartResult['prominence_ratio']:.2f}"


    if savePlot:
       plotDistances(plotData, radarConfig.frame_rate_Hz, 'full-spectrum'+name_suffix, data_path=dataPath, max_plot_bpm=max_plot_bpm)
       plotDistances(plotDataLow, radarConfig.frame_rate_Hz, low_label, data_path=dataPath, max_plot_bpm=max_plot_bpm)
       plotDistances(plotDataHigh, radarConfig.frame_rate_Hz, high_label, data_path=dataPath, max_plot_bpm=max_plot_bpm)
       plotDistances(combinedData, radarConfig.frame_rate_Hz, f"{title}{name_suffix}", dataColumns=[1, 2], legend=["Respiration", "Heart"], data_path=dataPath, max_plot_bpm=max_plot_bpm)
    else:
        plotDistances(plotData, radarConfig.frame_rate_Hz, 'full-spectrum', max_plot_bpm=max_plot_bpm)
        plotDistances(plotDataLow, radarConfig.frame_rate_Hz, low_label, max_plot_bpm=max_plot_bpm)
        plotDistances(plotDataHigh, radarConfig.frame_rate_Hz, high_label, max_plot_bpm=max_plot_bpm)
        plotDistances(combinedData, radarConfig.frame_rate_Hz, f"{title}{name_suffix}", dataColumns=[1,2], legend=["Respiration","Heart"], max_plot_bpm=max_plot_bpm)

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
    breathResult = calcRateSNR3( plotDataLow[:,1], radarConfig.frame_rate_Hz, n_harmonics=4,
                                 min_freq= breathRange[0], max_freq=breathRange[1])
    breathRate = breathResult["rate"]
    breathStd = np.std(plotDataLow[:,1]) #Standard Deviation of Respiration Signal

    plotDataHigh = distData.copy()
    # remove harmonics of breathrate
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*2, Q=5, fs=frames_per_min)
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*3, Q=5, fs=frames_per_min)
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*4, Q=5, fs=frames_per_min)
    # filter for heartRate
    plotDataHigh[:,1] = filterData(plotDataHigh[:,1],heartRange[0], heartRange[1], frames_per_min, order=4 )
    heartResult = calcRateSNR3( plotDataHigh[:,1], radarConfig.frame_rate_Hz, n_harmonics=4,
                                min_freq=heartRange[0], max_freq=heartRange[1])
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


def calcRatesOverTimeAwake(dataPath:str, plotRange:int, breathBand:list=AWAKE_BREATH_BAND,
                            heartBand:list=AWAKE_HEART_BAND, status_label:str="awake/arousal state"):
    """
    calculate the breath- and heartrate for different parts of the distanceData
    """
    dataFilename    = getDistFilename(dataPath, USE_ALT_PHASE, USE_ALT_DIST)
    distData        = np.load(dataFilename)
    distCount       = distData.shape[0]
    radarConfig     = importRadarConfig(dataPath)

    awake_rates = []
    rateLen = int(radarConfig.frame_rate_Hz * plotRange)
    if distCount < rateLen:
        print(f"   WARNING: recording has {distCount} frames, shorter than one "
              f"{plotRange}s window ({rateLen} frames needed) - skipping, 0 rate windows produced.")
        return awake_rates
    for ofs in range(0, distCount-rateLen, rateLen//2):
        rateData = distData[ofs:ofs+rateLen,:]
        window_timestamp = distData[ofs, 0]
        displacement = rateData[:, 1]
        window_std = np.std(displacement)

        # calculate rates from selected data, using whichever band the caller passed in
        breath_heart_rate = calcRates(radarConfig, rateData, breathBand, heartBand)

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
            "Arousal_Flag": status_label,
            "Displacement_Std": window_std
       })
    return awake_rates


def runAwakeAnalysis(data_paths:list = None, breathBand:list=AWAKE_BREATH_BAND,
                      heartBand:list=AWAKE_HEART_BAND, max_plot_bpm:float=None,
                      status_label:str="awake/arousal state", title:str="Awake Hedgehog Heart and Respiration Rates2"):
    '''
    run rate-extraction + plotting for awake/active recordings. Band defaults to
    AWAKE_BREATH_BAND/AWAKE_HEART_BAND (hedgehog arousal); pass ACTIVE_BREATH_BAND/
    ACTIVE_HEART_BAND for dog or active-hedgehog recordings.
    '''
    if data_paths is None:
        data_paths = radarFilePaths1

    if max_plot_bpm is None:
        max_plot_bpm = AWAKE_MAX_PLOT_BPM

    for data_path in data_paths:
        print(f"\nProcessing (awake config): {data_path}")
        try:
            awake_rates = calcRatesOverTimeAwake(data_path, PLOT_RANGE, breathBand, heartBand, status_label)
            if len(awake_rates) == 0:
                print(f"   Skipping {data_path} - no rate windows produced (recording too short).")
                continue
            awake_rates_df = pd.DataFrame.from_dict(awake_rates)
            awake_rates_df["Timestamp_UTC"] = pd.to_datetime(awake_rates_df["Timestamp_s"], unit="s", utc=True)
            awake_rates_df.to_csv(os.path.join(data_path, "awake_rates.csv"), index=False)
            plotDistData(data_path, breathBand, heartBand, PLOT_RANGE, savePlot=True,
                         max_plot_bpm=max_plot_bpm, title=title)
        except Exception:
            print("Error processing dir:", data_path)
            traceback.print_exc()


# if __name__ == '__main__':
#     runAwakeAnalysis(radarFilePaths1, breathBand=AWAKE_BREATH_BAND, heartBand=AWAKE_HEART_BAND,
#                         max_plot_bpm=ACTIVE_MAX_PLOT_BPM, status_label="active",
#                         title="Hedgehog Heart and Respiration Rates2")  # hedgehog station (60-240)


# if __name__ == '__main__':
#     runAwakeAnalysis(radarFilePaths2, breathBand=ACTIVE_BREATH_BAND, heartBand=ACTIVE_HEART_BAND,
#                         max_plot_bpm=ACTIVE_MAX_PLOT_BPM, status_label="active",
#                         title="Hedgehog Station Heart and Respiration Rates2")  # hedgehog station (60-240)


# if __name__ == '__main__':
#     runAwakeAnalysis(radarFilePaths3, breathBand=DOG_BREATH_BAND, heartBand=DOG_HEART_BAND,
#                     max_plot_bpm=DOG_MAX_PLOT_BPM, status_label="active",
#                     title="Dog Heart and Respiration Rates")  # dog (60-120)


def load_awake_rates(paths, filename="awake_rates.csv"):
    dfs = []
    for p in paths:
        f = os.path.join(p, filename)
        if not os.path.exists(f):
            continue
        if os.path.getsize(f) == 0:
            print(f"empty, skipped: {f}")
            continue
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            print(f"no rows, skipped: {f}")
            continue
        df["Recording"] = os.path.basename(os.path.dirname(p))  
        dfs.append(df)
    if not dfs:
        raise ValueError("no valid awake_rates.csv found")
    return pd.concat(dfs, ignore_index=True)


awake_rates = load_awake_rates(radarFilePaths2) 

cols = ["Heart_Rate_BPM", "Breathing_Rate_BPM"]

print("\n=== 1. Overall Dataset Summary (All Windows) ===")
summary = awake_rates[cols].describe(percentiles=[.05, .25, .5, .75, .95]).T
print(summary.round(1))

print("\n=== 2. Per-Hedgehog Summary (Min, Mean, Max) ===")
# Aggregate min, mean, and max for each hedgehog recording folder
rec_summary = awake_rates.groupby("Recording")[cols].agg(['min', 'mean', 'max'])
print(rec_summary.round(1))
# To output directly as a Markdown table to copy into notes/reports:
# print(rec_summary.round(1).to_markdown())


print("\n=== 3. Per-Hedgehog Summary (Filtered for SNR >= 10 dB) ===")
# Filter for high-confidence windows to get cleaner physiological ranges
valid_awake = awake_rates[
    (awake_rates["Heart_SNR_dB"] >= 10) & 
    (awake_rates["Breath_SNR_dB"] >= 10)
]

if not valid_awake.empty:
    filtered_rec_summary = valid_awake.groupby("Recording")[cols].agg(['min', 'mean', 'max'])
    print(filtered_rec_summary.round(1))
else:
    print("No windows met the >= 10 dB SNR threshold across these recordings.")

print(awake_rates["Displacement_Std"].agg(['min', 'mean', 'max']).round(4))


def generateAwakeSnrSummary(awake_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Computes overall and filtered SNR summary statistics (N, median, mean, SD, min, max, range) 
    for breathing and heart rate SNR from the awake/active recordings.
    """
    snr_rows = []
    metrics = [
        ("Breathing Rate", "Breath_SNR_dB"),
        ("Heart Rate", "Heart_SNR_dB")
    ]
    
    # 1. All Windows
    for metric_name, snr_col in metrics:
        if snr_col not in awake_rates.columns:
            continue
        sub = awake_rates[snr_col].dropna()
        if len(sub) > 0:
            snr_rows.append({
                "State/Condition": "Awake Station (All)",
                "Metric": metric_name,
                "N": len(sub),
                "Median": round(sub.median(), 2),
                "Mean ± SD": f"{sub.mean():.2f} ± {sub.std():.2f}",
                "Min": round(sub.min(), 2),
                "Max": round(sub.max(), 2),
                "Range": f"{sub.min():.2f} – {sub.max():.2f}"
            })
            
    # 2. Filtered Windows (High Confidence: SNR >= 10 dB)
    for metric_name, snr_col in metrics:
        if snr_col not in awake_rates.columns:
            continue
        valid_sub = awake_rates.loc[awake_rates[snr_col] >= 10.0, snr_col].dropna()
        if len(valid_sub) > 0:
            snr_rows.append({
                "State/Condition": "Awake Station (SNR >= 10 dB)",
                "Metric": metric_name,
                "N": len(valid_sub),
                "Median": round(valid_sub.median(), 2),
                "Mean ± SD": f"{valid_sub.mean():.2f} ± {valid_sub.std():.2f}",
                "Min": round(valid_sub.min(), 2),
                "Max": round(valid_sub.max(), 2),
                "Range": f"{valid_sub.min():.2f} – {valid_sub.max():.2f}"
            })
            
    return pd.DataFrame(snr_rows)

def generateAwakePerRecordingSnrSummary(awake_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Computes SNR summary statistics broken down individually per recording folder.
    """
    rows = []
    for rec, group in awake_rates.groupby("Recording"):
        for metric_name, snr_col in [("Breathing Rate", "Breath_SNR_dB"), ("Heart Rate", "Heart_SNR_dB")]:
            sub = group[snr_col].dropna()
            if len(sub) > 0:
                rows.append({
                    "Recording": rec,
                    "Metric": metric_name,
                    "N": len(sub),
                    "Median (dB)": round(sub.median(), 2),
                    "Mean (dB)": round(sub.mean(), 2),
                    "SD (dB)": round(sub.std(), 2),
                    "Min (dB)": round(sub.min(), 2),
                    "Max (dB)": round(sub.max(), 2),
                    "Range (dB)": f"{round(sub.min(), 2)} – {round(sub.max(), 2)}"
                })
    return pd.DataFrame(rows)

# # Execution and Export
# if __name__ == '__main__':
#     awake_rates = load_awake_rates(radarFilePaths2)
    
#     # Overall summary table matching your master format
#     awake_snr_summary = generateAwakeSnrSummary(awake_rates)
#     print("\n=== Awake Station Overall SNR Summary ===")
#     print(awake_snr_summary.to_string(index=False))
#     awake_snr_summary.to_csv("awake_station_snr_summary.csv", index=False)
    
#     # Per-recording breakdown table
#     per_rec_snr = generateAwakePerRecordingSnrSummary(awake_rates)
#     print("\n=== Awake Station Per-Recording SNR Summary ===")
#     print(per_rec_snr.to_string(index=False))
#     per_rec_snr.to_csv("awake_per_recording_snr_summary.csv", index=False)

# print(awake_rates[awake_rates["Breath_SNR_dB"].isna() | awake_rates["Heart_SNR_dB"].isna()][["Recording", "Timestamp_s", "Heart_Rate_BPM", "Breath_SNR_dB"]])


from radarTestDefs import radarFilePaths3

# 1. Define the target dog recording path
dog_path = radarFilePaths3[0]
print(f"Processing dog recording: {dog_path}")

# 2. Run the awake analysis using the Dog frequency bands and parameters
runAwakeAnalysis(
    data_paths=[dog_path],
    breathBand=DOG_BREATH_BAND,
    heartBand=DOG_HEART_BAND,
    max_plot_bpm=DOG_MAX_PLOT_BPM,
    status_label="dog",
    title="Dog Heart and Respiration Rates"
)

# 3. Load the resulting awake_rates.csv for this specific recording
dog_rates = load_awake_rates([dog_path])

# 4. Summarize the SNR metrics (Breathing and Heart Rate SNR)
cols = ["Breath_SNR_dB", "Heart_SNR_dB"]

print("\n=== Dog Recording SNR Summary (Statistical Description) ===")
snr_desc = dog_rates[cols].describe(percentiles=[.25, .5, .75]).T
print(snr_desc.round(2))

# Alternatively, using your per-recording summary function:
if 'generateAwakePerRecordingSnrSummary' in globals():
    print("\n=== Detailed SNR Breakdown ===")
    detailed_summary = generateAwakePerRecordingSnrSummary(dog_rates)
    print(detailed_summary.to_string(index=False))
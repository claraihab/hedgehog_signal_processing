'''
radarLib - Library to process Radar-Data from BGT60xx
'''

from scipy import signal
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, detrend, iirnotch, find_peaks

DATA_NAME   = "radar.npy"
CONFIG_NAME = "config.json"
TIME_FILE = "radar_timestamp.csv"

def importRadarData(dataPath : str) -> tuple:
    '''
    import radar-data and configuration
    :param dataPath:
    :return: dictionary with configuration for this recording and numpy-array with raw radar data
    '''
    configFile = dataPath + "\\" + CONFIG_NAME
    dataFile    = dataPath + "\\" + DATA_NAME
    timeFile = dataPath + "\\" + TIME_FILE

    # read JSON-File with configuration
    with open(configFile) as configJson:
        deviceConfig = json.load(configJson)['device_config'] ['fmcw_single_shape']

    # read csv
    df_time = pd.read_csv(timeFile)
    df_timeStamps = df_time.iloc[:, 0].values 

    # read radar data
    data = np.load(dataFile, mmap_mode='r') #import in chunks so it doesn't crash

    return deviceConfig, df_timeStamps, data

def processOneChirp(chirp:np.ndarray, mean:bool =False, window:np.ndarray=None) -> np.ndarray:
    '''
    process radar-data for one chirp
    :param chirp:   1D-ndarray with raw radar data of one chirp
    :param mean:    should the offset removed from the raw-data
    :param window:  window to use before fft
    :return:    fft of one chirp
    '''
    if len(chirp.shape) != 1:
        print("Error: Chirp must have 1 dimension!")
        return None
    # apply window before FFT
    if window is not None:
        if chirp.shape !=window.shape:
            print("Error: Shape of Window does not match chirp data!")
            return None
        chirp = chirp * window
    # remove offset from values
    if mean:
        chirpMean = chirp.mean()
        chirp = chirp -  mean
    # calculate real-FFT from chirp
    fftData = np.fft.rfft(chirp)

    return fftData


def processOneFrame(frame:np.ndarray, mean:bool=False, window:np.ndarray=None) -> np.ndarray:
    '''
    process data from one radar-frame
    :param frame:   2D-ndarray with raw radar data of one frame
    :param mean:    should the offset removed from the raw-data
    :param window:  window to use before fft
    :return:    mean of the FFT-Results of chirps in frame
    '''

    frameShape = frame.shape
    if len(frameShape) !=2:
        print("Error: Framedata have to be 2D!")
        return None

    chirpCnt    = frameShape[0]     # number of chirps in one frame
    chirpSmpls  = frameShape[1]     # number of samples in one chirp

    if (chirpSmpls % 2) ==0:
        fftLen = (chirpSmpls //2) +1
    else:
        fftLen = (chirpSmpls+1) //2

    # prepare result-array
    frameSum = np.zeros(fftLen, dtype=complex)

    # process each chirp and sum into result array
    for i in range(chirpCnt):
        chirpData = frame[i]
        chirpFFT = processOneChirp(chirpData, mean=mean, window=window)
        frameSum = frameSum + chirpFFT

    # calculate mean of all chirps in one frame
    chirpsMean = frameSum / float(chirpCnt)

    return chirpsMean


#Configurations and calculated values

DATA_PATH = r"D:\igel\Radar-Test\recording_2026_02_18_10_17_15\RadarIfxBGT60TR13C_00"

# calculate roundtrip time in s for radar
SPEED_OF_LIGHT = 3e8  # 300.000.000 m/s
TARGET_DISTANCE = 0.4   # distance to target/animal in m
TARGET_RT_TIME  = (TARGET_DISTANCE*2) / SPEED_OF_LIGHT  # time for one roundtrip

print("Import Data from: ", DATA_PATH)
configData, timeStamps, radarData = importRadarData(DATA_PATH)

freq_center = (configData['start_frequency_Hz'] + configData['end_frequency_Hz']) / 2
wavelength = SPEED_OF_LIGHT / freq_center  # Should be ~0.005 meters for BGT60

freqStart       = configData['start_frequency_Hz']
freqStop        = configData['end_frequency_Hz']
smplPerChirp    = configData['num_samples_per_chirp']
smplRate        = configData['sample_rate_Hz']
chirpRepTime    = configData['chirp_repetition_time_s']
chirpPerFramne  = configData['num_chirps_per_frame']

# calculate more values from config
chirpTime       = smplPerChirp / smplRate   # length of one chirp in s
freqRange       = freqStop - freqStart
freqChgSpd      = freqRange / chirpTime     # change-rate of frequency in Hz/s
fftStepHz       = 1 / chirpTime             # step size of FFT result in Hz, the freq of bin[n] is fft_step_size * n
fftStepM        = (SPEED_OF_LIGHT / smplRate) * smplPerChirp

freqChgTarget = TARGET_RT_TIME * freqChgSpd
fftTargetBin  = freqChgTarget / fftStepHz
ffTargetBinIdx  = int(fftTargetBin)         # what FFT-bin has wanted data

print("Data-Shape: ", radarData.shape)


# Definitions for filtering and seperating data

### def butterpass to be added

def apply_notch_filter(data, target_freq, fs, quality_factor=30.0):
    """
    Removes a specific frequency from the signal.
    :param target_freq: The frequency to remove
    :param fs: Sampling frequency 
    :param quality_factor: How narrow the notch is 
    """
    # Design the notch filter
    b_notch, a_notch = iirnotch(target_freq, quality_factor, fs)
    
    # Apply it to the signal
    filtered_data = filtfilt(b_notch, a_notch, data)
    
    return filtered_data


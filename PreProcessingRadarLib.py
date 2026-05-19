'''
First library including directories and functions for preprocessing of raw radar data from BGT60xx

'''

#packages needed 
import numpy as np
import json
import sys, os 
from scipy.signal import get_window, butter, filtfilt, detrend, iirnotch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, detrend


# functions for processing of raw radar-data

DATA_NAME       = "radar.npy"
CONFIG_NAME     = "config.json"
TIMESTAMP_NAME  = "radar_timestamp.csv"
ANGLES_NAME     = "radar_angles.npy"

SPEED_OF_LIGHT = 300e6  # 300.000 km/s = 300.000.000 m/s

def importRadarData(dataPath : str) -> tuple : 
    '''
    import radar-data and configuration
    :param dataPath:
    :return: dictionary with configuration for this recording and numpy-array with raw radar data
    '''
    configFile = dataPath + "\\" + CONFIG_NAME
    dataFile    = dataPath + "\\" + DATA_NAME
    timestampFile = dataPath + "\\" + TIMESTAMP_NAME
    # read JSON-File with configuration
    with open(configFile) as configJson:
        deviceConfig = json.load(configJson)['device_config'] ['fmcw_single_shape']
    # read radar data
    data = np.load(dataFile, mmap_mode = "r")
    # read timestamps
    timestamp = np.loadtxt(timestampFile)

    return deviceConfig, data, timestamp

def processOneChirp(chirp:np.ndarray, mean:bool =True, window:np.ndarray=None) -> np.ndarray:
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


def calcFrequencyFromDistance(radarconfig:dict, distance:float) -> float:
    freqStart   = radarconfig['start_frequency_Hz']
    freqStop    = radarconfig['end_frequency_Hz']
    smplPerChirp= radarconfig['num_samples_per_chirp']
    smplRate    = radarconfig['sample_rate_Hz']
    chirpTime   = smplPerChirp / smplRate       # length of one chirp in s
    freqChgSpeed= (freqStop-freqStart) / chirpTime  # change of frequency in Hz/s

    rtTime = (2*distance) / SPEED_OF_LIGHT  # ToF for&back in s
    freq   = freqChgSpeed * rtTime          # change of frequency in one round trip
    return freq

def calcFFTbinFromFrequency(radarconfig:dict, frequency:float) -> int:
    chirpRepTime = radarconfig['chirp_repetition_time_s']
    chirpPerFrame = radarconfig['num_chirps_per_frame']
    frameTime   = chirpRepTime * chirpPerFrame     # length of one frame in s
    fftStepSize = 1/ frameTime                      # freq-diff between 2 FFT-bins
    fftBin      = frequency / fftStepSize
    return int(fftBin)

def preProcessRadarData(dataPath:str, distance:float|list[float], window:np.ndarray|str=None, frameMax:int=None) -> np.ndarray:
    '''
    import raw radar-data, process frames and return numpy-array with timestamps and angles from FFTbins

    the result is a 2D numpy-array:
        timestamp, phi [, phi, phi]
        timestamp, phi [, phi, phi]
        timestamp, phi [, phi, phi]

    normally the result should be saved as a npy-file [ np.save(dataPath+"\\"+ANGLES_NAME, results) ]
    to avoid time-consuming recalculation at later processing steps

    :param dataPath:    where are the data?
    :param FFTbin:      int|list of int     which bins of the FFT to get
    :param window:      optional, window or name of window-function for FFT of chirps
    :param frameMax:    optional, max number of frames to process
    '''
    TX_ANTENNA = 0
    RX_ANTENNA = 0

    config, data, timestamp = importRadarData(dataPath)
    # create window-array by name
    if window is not None:
        winSize = config['num_samples_per_chirp']
        window = get_window( window, winSize)

    # limit number of frames to process (for testing)
    frameCount = data.shape[0]
    if frameMax is not None:
        if frameCount >frameMax: frameCount = frameMax

    # prepare result-array
    if type(distance) == list:
        binCnt = len(l) #l is not defined check it..
        FFTbin = []
        for d in distance:
            freq = calcFrequencyFromDistance(d)
            bin =  calcFFTbinFromFrequency(freq)
            FFTbin.append(bin)
    else:
        binCnt  = 1
        freq = calcFrequencyFromDistance(distance)
        FFTbin  = calcFFTbinFromFrequency(freq)
        
    results = np.zeros([frameCount,binCnt+1], dtype=float)

    # process frames and fill results array
    for frameNum in range(frameCount):
        frameRaw = data[frameNum, TX_ANTENNA, RX_ANTENNA].copy()  # select frame i, txAnt 0, rxAnt 0
        frameFFT = processOneFrame(frameRaw, mean=True, window=window)  # should use window to get better FFT results

        if type(FFTbin) == list:
            phi = np.angle( frameFFT[FFTbin] )
            results[frameNum] = np.insert(phi, 0, timestamp[frameNum])
        else:
            # select the bin for distance
            phi = np.angle( frameFFT[FFTbin] )
            # save calculated phi in result-array
            results[frameNum] = [timestamp, phi]

    return results

# functions for preprocessed data

def saveRadarAngles( radarAngles:np.ndarray, dataPath:str):
    np.save(dataPath + "\\" + ANGLES_NAME, radarAngles)




'''
Second library includes functions for the filters used on the preprocessed data
'''


def importRadarAngles(dataPath:str):
    '''
    import preprocessed radar-data and radar-configuration from datapath
    :param dataPath:    where to find files
    :return:            deviceConfig, preprocessed radar-data
    '''
    configFile  = dataPath + "\\" + CONFIG_NAME
    anglesFile  = dataPath + "\\" + ANGLES_NAME
    # read JSON-File with configuration
    with open(configFile) as configJson:
        deviceConfig = json.load(configJson)['device_config'] ['fmcw_single_shape']
    # read radar data
    angles = np.load(anglesFile, mmap_mode = "r")
    return deviceConfig, angles


def calculate_peak_rate(signal, start_idx, win_samples, fs, f_min, f_max, n_fft=32768):
    """Slices a signal chunk, applies a Blackman window, and finds the peak frequency in BPM."""
    chunk = signal[start_idx : start_idx + win_samples]
    windowed = chunk * np.blackman(win_samples)
    
    yf = np.abs(fft(windowed, n=n_fft))
    xf = fftfreq(n_fft, 1/fs)
    mask = (xf >= f_min) & (xf <= f_max)
    
    rate = xf[mask][np.argmax(yf[mask])] * 60
    return rate, xf, yf, mask


def save_diagnostic_plot(r_xf, r_yf, r_mask, h_xf, h_yf, h_mask, root_dir):
    """Generates and saves the one-time diagnostic spectrum profile."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    ax1.plot(r_xf[r_mask] * 60, r_yf[r_mask], color='blue')
    ax1.set_title("Respiration Spectrum (Magnitude vs. Frequency)")
    ax1.set_xlabel("Breaths Per Minute (BrPM)")
    
    ax2.plot(h_xf[h_mask] * 60, h_yf[h_mask], color='red')
    ax2.set_title("Heart Spectrum (Magnitude vs. Frequency)")
    ax2.set_xlabel("Beats Per Minute (BPM)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, "diagnostic_fft_window.png"), dpi=150)
    plt.close(fig)
    print(">>> Saved diagnostic spectrum profile to ROOT_DIR.")


from scipy.signal import butter, lfilter, iirnotch, detrend


def apply_filter(data, cutoff, fs, btype='bandpass', order=4):
    nyq = 0.5 * fs
    if btype == 'bandpass':
        low, high = cutoff[0] / nyq, cutoff[1] / nyq
        b, a = butter(order, [low, high], btype=btype)
    else:
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype=btype)
    return filtfilt(b, a, data)





def apply_notch_filter(data, target_freq, fs, quality_factor=30.0):
    """
    Removes a specific frequency from the signal.
    :param target_freq: The frequency to remove (e.g., 0.33 Hz for 20 BPM)
    :param fs: Sampling frequency (your 227 Hz)
    :param quality_factor: How narrow the notch is (higher = narrower)
    """
    # Design the notch filter
    b_notch, a_notch = iirnotch(target_freq, quality_factor, fs)
    
    # Apply it to the signal
    filtered_data = filtfilt(b_notch, a_notch, data)
    
    return filtered_data

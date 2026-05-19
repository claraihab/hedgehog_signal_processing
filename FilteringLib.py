'''
Second library includes functions for the filters used on the preprocessed data
'''

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend, iirnotch
from scipy.fft import fft, fftfreq


DATA_NAME       = "radar.npy"
CONFIG_NAME     = "config.json"
TIMESTAMP_NAME  = "radar_timestamp.csv"
ANGLES_NAME     = "radar_angles.npy"

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



def apply_filter(data, cutoff, fs, btype='bandpass', order=4):
    nyq = 0.5 * fs
    if btype == 'bandpass':
        low, high = cutoff[0] / nyq, cutoff[1] / nyq
        b, a = butter(order, [low, high], btype=btype)
    else:
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype=btype)
    return filtfilt(b, a, data)




'''
    radarTools - collection of additional functions for the analyses of radar data
    
    - plotting data to files
'''

import os
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt


def plotRates(data_path:str, name:str, rate_data:np.ndarray):
    '''
    create a plot with the 2 subplot from the rate_data and save as png-file
    
    rate data must be a numpy-array with 3 columns:
    - timestamp
    - heartrate
    - breathrate
    
    :param data_path:
    :param name:
    :param rate_data:
    :return:
    '''
    # get series from data
    v_timestamps  = rate_data[:0]
    v_heartrate   = rate_data[:1]
    v_breathrate  = rate_data[:2]

    # convert timestamps to date-time
    v_datetime  = [dt.datetime.fromtimestamp(t) for t in v_timestamps]

    fig, (ax_ah, ax_ab) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    ax_ah.plot(v_datetime, v_heartrate, color='purple', marker='o', markersize=2, linestyle='-')
    ax_ah.set_title(f"Heart-Rate in bpm")
    
    ax_ab.plot(v_datetime, v_breathrate, color='orange', linewidth=0.8)
    ax_ab.set_title("Breath-Rate in bpm")

    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"rate_plot_{name}.png"), dpi=150)
    plt.close(fig)

def plotMovements(data_path: str, name: str, distance_data: np.ndarray):
    '''
    create a plot with the 2 subplot from the distance_data and save as png-file

    rate data must be a numpy-array with 3 columns:
    - timestamp
    - distances filtered for heart-rate
    - distances filtered for breath-rate

    :param data_path:
    :param name:
    :param distance_data:
    :return:
    '''
    # get series from data
    v_timestamps = distance_data[:0]
    v_heart_move = distance_data[:1]
    v_breat_move = distance_data[:2]
    
    # convert timestamps to date-time
    v_datetime = [dt.datetime.fromtimestamp(t) for t in v_timestamps]
    
    fig, (ax_ah, ax_ab) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    ax_ah.plot(v_datetime, v_heart_move, color='purple', marker='o', markersize=2, linestyle='-')
    ax_ah.set_title(f"fast movements in mm")
    
    ax_ab.plot(v_datetime, v_breat_move, color='orange', linewidth=0.8)
    ax_ab.set_title("slow movements in mm")
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"move_plot_{name}.png"), dpi=150)
    plt.close(fig)

def plotFFTdata(data_path: str, name: str, FFT_data: np.ndarray, timebase:float):
    '''
    plot the spectrum from an FFT
    
    if FFT_data is 2D plot each singe FFT in gray and mean of all in black
    :param data_path:
    :param name:
    :param FFT_data:
    :param timebase:
    :return:
    '''
    FFT_freq = range(len(FFT_data)) / timebase
    fig, ax = plt.subplots(1, 1, figsize=(15,8) )
    if len(FFT_data.shape) ==1:
        # simple case, only one FFT-data set
        ax.plot(FFT_freq, FFT_data, color='black')
    else:
        FFT_mean = np.mean(FFT_data, axis=1)
        FFT_count = FFT_data.shape(0)
        for i in range(FFT_count):
            ax.plot(FFT_freq, FFT_data[i,:], color=0.5)
        ax.plot(FFT_freq, FFT_mean, color='black')
        
    ax.set_title('Spectrum: '+name)
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"fft_plot_{name}.png"), dpi=150)
    plt.close(fig)

    
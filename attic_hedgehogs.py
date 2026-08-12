'''
    attic - Library of unused functions 
'''

import numpy as np
import matplotlib as plt
import datetime as dt
import os

# unused !!!
# ------------


def plotRates(data_path:str, name:str, rate_data:np.ndarray):
    v_timestamps  = rate_data[:,0]
    v_heartrate   = rate_data[:,1]
    v_breathrate  = rate_data[:,2]

    v_datetime  = [dt.datetime.fromtimestamp(t) for t in v_timestamps]

    fig, (ax_ah, ax_ab) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    ax_ah.plot(v_datetime, v_heartrate, color='purple', marker='o', markersize=2, linestyle='-', linewidth=0.8)
    ax_ah.set_title(f"Heart-Rate in bpm")
    
    ax_ab.plot(v_datetime, v_breathrate, color='orange', linewidth=0.8)
    ax_ab.set_title("Breath-Rate in bpm")

    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"rate_plot_{name}.png"), dpi=150)
    plt.close(fig)


def plotMovements(data_path: str, name: str, distance_data: np.ndarray):
    v_timestamps = distance_data[:,0] 
    v_heart_move = distance_data[:,1]
    v_breath_move = distance_data[:,2]
    
    v_datetime = [dt.datetime.fromtimestamp(t) for t in v_timestamps]
    
    fig, (ax_ah, ax_ab) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    ax_ah.plot(v_datetime, v_heart_move, color='purple', marker='o', markersize=2, linestyle='-')
    ax_ah.set_title(f"fast movements in mm")
    
    ax_ab.plot(v_datetime, v_breath_move, color='orange', linewidth=0.8)
    ax_ab.set_title("slow movements in mm")
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"move_plot_{name}.png"), dpi=150)
    plt.close(fig)



def findBaseFreq(FFT_data:np.ndarray, timebase:float, weighted:bool=True) ->float:
    '''
    find the 'base' frequency from FFT-data,
    
    :param FFT_data:
    :param timebase:   length of the data (in s) used to calculate FFT
    :param weighted:        if True -> FFT-values will be weighted by 1/n to favor low frequencies
    :return:
    '''
    data = np.absolute(FFT_data)    # use absolute values,
    data[0] = 0                     # set DC-component to zero
    
    if weighted:
        for i in range(1, len(data)):
            data[i] = data[i] /i
    peak_idx = np.argmax(data)           # find highest peak
    peak_hz  = peak_idx / timebase  # convert to hz
    return peak_hz

def maskFFTdata(FFT_data:np.ndarray,  timebase:float, min_Hz:float, max_Hz:float) -> float:
    '''
    return a numpy array where the values outside the range min-max are set to zero
    :param FFT_data:
    :param timebase:
    :param min_Hz:
    :param max_Hz:
    :return:
    '''
    min_idx = int(min_Hz * timebase)
    max_idx = int(max_Hz * timebase)

    data = FFT_data.copy()  # reserve original data
    if min_idx >0:
        data[0:min_idx] = 0
    if max_idx >0 and max_idx <=len(data):
        data[max_idx:]  = 0
    return data


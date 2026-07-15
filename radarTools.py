'''
    radarTools - collection of additional functions for the analyses of radar data
    
    - plotting data to files
'''

import os
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as signal


PLOT_COLORS = ['cornflowerblue', 'firebrick']

MAX_PLOT_BPM_HIB   = 70 
MAX_PLOT_BPM    = 300 

WEIGHTED_PEAK_FIND  = False

def normalize(data:np.ndarray) -> np.ndarray:
    return (data - np.mean(data)) / np.std(data)



def plotDistances(distData:np.ndarray, framerate_Hz:float, name:str, dataColumns:int = 1, legend:list=None, data_path:str = None):
    """
    plot a diagram with wave and spectrum from distData
    :param distData:
    :param framerate_Hz:
    :param name:            diagram-title
    :param dataColumns:     datacolumn(s) to plot, if None plot all columns of distData
    :param legend:          list of strings used as legend for plot
    :param data_path:       path to save plot as PNG-file, ==None -> display plot
    :return:
    """
    dataLen = distData.shape[0]
    if dataColumns is None: # plot all dataColumns 1..
        dataColumns = list( range(1,distData.shape[1]) )
    elif type(dataColumns) ==int:
        dataColumns = [dataColumns]

    timestamps  = distData[:,0]
    timestamps  = timestamps - timestamps[0]    # make tiemstamps start with 0

    peakFreq = []
    specs = []

    specFreq = sp.fft.rfftfreq(dataLen, 1 / (framerate_Hz * 60))  # display frequencies in 1/min

    # detect peak frequencies
    distances = distData[:,dataColumns].copy() * 1000

    for col in range(len(dataColumns)):
        distNorm = normalize(distances[:,col])
        distWindowed = distNorm * signal.get_window(window='blackman', Nx=len(distData))
        spectrum = sp.fft.rfft(distWindowed)
        spectrum[0] = 0 # suppress DC component
        specAbs = np.abs(spectrum)
        peakIdx = np.argmax(specAbs)
        # append results to lists
        specs.append(specAbs)
        peakFreq.append(specFreq[peakIdx])

    # scale FFT results into 0..1.0
    fftMax = 1e-20  # very small value greater zero
    for spec in specs: fftMax = max(fftMax, np.max(spec))
    for idx in range(len(specs)):  specs[idx] = specs[idx] / fftMax

    # limit FFT-result to reasonable range 0-200 BPM
    maxFFTidx = int(MAX_PLOT_BPM * len(distances) / (framerate_Hz * 60) ) +1
    print(f"MAX_PLOT_BPM = {MAX_PLOT_BPM} -> maxFFTidx = {maxFFTidx}")
    if maxFFTidx >100: maxFFTidx = 100

    if legend is None:
        labels = [None] * len(dataColumns)
    else:
        labels = legend

    fig, (ax_dist, ax_spec) = plt.subplots(2, 1, figsize=(15, 8), sharex=False)
    fig.suptitle(name)


    ax_dist.grid(True)
    ax_dist.set_title(f"relative distance")

    ax_spec.grid(True)
    ax_spec.set_title("spectrum in 1/min")

    for col in range(len(dataColumns)):
        plotColor = PLOT_COLORS[col % len(PLOT_COLORS)]
        ax_dist.plot(timestamps, distances[:,col], color=plotColor, label=labels[col])
        specAbs = specs[col]
        ax_spec.plot(specFreq[:maxFFTidx], specAbs[:maxFFTidx], color=plotColor, label=labels[col])
        if peakFreq[col] <MAX_PLOT_BPM:
            ax_spec.axvline(peakFreq[col], color=plotColor, linestyle=":")
            ax_spec.annotate(f'{peakFreq[col]:6.2f}', (peakFreq[col], 0))  #important

    if legend is not None:
        ax_dist.legend(loc='upper right')
        ax_spec.legend(loc='upper right')

    if data_path is not None:
        plt.tight_layout()
        plt.savefig(os.path.join(data_path, f"dist_plot_{name}.png"), dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return peakFreq




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
    v_breath_move = distance_data[:2]
    
    # convert timestamps to date-time
    v_datetime = [dt.datetime.fromtimestamp(t) for t in v_timestamps]
    
    fig, (ax_ah, ax_ab) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    ax_ah.plot(v_datetime, v_heart_move, color='purple', marker='o', markersize=2, linestyle='-')
    ax_ah.set_title(f"fast movements in mm")
    
    ax_ab.plot(v_datetime, v_breath_move, color='orange', linewidth=0.8)
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
        FFT_count = FFT_data.shape[0]
        for i in range(FFT_count):
            ax.plot(FFT_freq, FFT_data[i,:], color=0.5)
        ax.plot(FFT_freq, FFT_mean, color='black')
        
    ax.set_title('Spectrum: '+name)
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"fft_plot_{name}.png"), dpi=150)
    plt.close(fig)

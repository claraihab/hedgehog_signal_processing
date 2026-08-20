'''
    radarTools - collection of additional functions for the analyses of radar data
    - plotting data to files
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as signal


PLOT_COLORS = ['blue', 'red']

WEIGHTED_PEAK_FIND  = False

MAX_PLOT_BPM   = 70 
AWAKE_MAX_PLOT_BPM = 300

def normalize(data:np.ndarray) -> np.ndarray:
    return (data - np.mean(data)) / np.std(data)


def plotFFTdata(data_path: str, name: str, FFT_data: np.ndarray, timebase:float, heart_rate_bpm: float = 60.0):
    FFT_freq = np.arange(len(FFT_data)) / timebase
    fig, ax = plt.subplots(1, 1, figsize=(15,8) )
    if len(FFT_data.shape) ==1:
        ax.plot(FFT_freq, FFT_data, color='black')
    else:
        FFT_mean = np.mean(FFT_data, axis=1)
        FFT_count = FFT_data.shape[0]
        for i in range(FFT_count):
            ax.plot(FFT_freq, FFT_data[:,i], color='0.5')
        ax.plot(FFT_freq, FFT_mean, color='black')

    f_hr = heart_rate_bpm / 60.0
    for h in range(1, 5):
        harmonic_freq = h * f_hr
        ax.axvline(
            x=harmonic_freq,
            color="red",
            linestyle="--",
            alpha=0.4,
            label=f"Harmonics ({f_hr:.2f} Hz)" if h == 1 else None,
        )
    
    ax.set_title('Spectrum: '+name)
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"fft_plot_{name}.png"), dpi=150)
    plt.close(fig)



def plotDistances(distData:np.ndarray, framerate_Hz:float, name:str, dataColumns:int = 1, legend:list=None, data_path:str = None, max_plot_bpm:float = None, harmonics_label:str = "Heart"):
    """
    plot a diagram with wave and spectrum from distData
    :param distData:
    :param framerate_Hz:
    :param name:            diagram-title
    :param dataColumns:     datacolumn(s) to plot, if None plot all columns of distData
    :param legend:          list of strings used as legend for plot
    :param data_path:       path to save plot as PNG-file, ==None -> display plot
    :param max_plot_bpm:    upper bound (in 1/min) for the spectrum x-axis; defaults to module MAX_PLOT_BPM.
    :param harmonics_label: only the column whose legend label matches this (case-insensitive)
                             gets harmonic marker lines drawn - e.g. only "Heart", not "Respiration".
                             Set to None to draw harmonics for every column, or pass a label that
                             matches nothing to draw none.
    :return:
    """
    if max_plot_bpm is None:
        max_plot_bpm = MAX_PLOT_BPM
    dataLen = distData.shape[0]
    if dataColumns is None:
        dataColumns = list( range(1,distData.shape[1]) )
    elif type(dataColumns) ==int:
        dataColumns = [dataColumns]

    timestamps  = distData[:,0]
    timestamps  = timestamps - timestamps[0]


    peakFreq = []
    specs = []

    specFreq = sp.fft.rfftfreq(dataLen, 1 / (framerate_Hz * 60)) # display frequencies in 1/min 

    distances = distData[:,dataColumns].copy() * 1000

    for col in range(len(dataColumns)):
        distNorm = normalize(distances[:,col])
        distWindowed = distNorm * signal.get_window(window='blackman', Nx=len(distData))
        spectrum = sp.fft.rfft(distWindowed)
        spectrum[0] = 0 # suppress DC component
        specAbs = np.abs(spectrum)
        peakIdx = np.argmax(specAbs)
        specs.append(specAbs)
        peakFreq.append(specFreq[peakIdx])

    fftMax = 1e-20
    for spec in specs: fftMax = max(fftMax, np.max(spec))
    for idx in range(len(specs)):  specs[idx] = specs[idx] / fftMax

    maxFFTidx = int(max_plot_bpm * len(distances) / (framerate_Hz * 60) ) +1
    print(f"max_plot_bpm = {max_plot_bpm} -> maxFFTidx = {maxFFTidx}")

    if legend is None:
        labels = [None] * len(dataColumns)
    else:
        labels = legend

    fig, (ax_dist, ax_spec) = plt.subplots(2, 1, figsize=(15, 8), sharex=False, linewidth = 0.8)
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

        f0 = peakFreq[col]
        if f0 < max_plot_bpm:
            ax_spec.axvline(f0, color=plotColor, linestyle=":")
            ax_spec.annotate(f'{f0:6.2f}', (f0, 0))

            # only draw the extra harmonic lines for the column matching harmonics_label
            # (e.g. Heart) - not for every column (e.g. Respiration)
            draw_harmonics = (harmonics_label is None) or (
                labels[col] is not None and labels[col].strip().lower() == harmonics_label.strip().lower()
            )
            if draw_harmonics:
                for h in range(2, 5):
                    harmonic_freq = f0 * h
                    if harmonic_freq < max_plot_bpm:
                        ax_spec.axvline(harmonic_freq, color=plotColor, linestyle="--", alpha=0.4)

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

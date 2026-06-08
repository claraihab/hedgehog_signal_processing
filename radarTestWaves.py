'''
    radarTestWaves - play with the distance-data from preProcessing
'''

import os
import numpy as np
from radarLib import *
from radarTools import *
from radarTestDefs import *

PLOT_RANGE      = 30    #how much data to plot in s
USE_ALT_DIST    = True



def plotDistDataChunk(radarConfig:radarConfigData, plotData:np.ndarray, savePlot:bool = False, datapath:str=None):
    if USE_ALT_DIST:
        name_suffix = ' - alt'
    else:
        name_suffix = ''

    frames_per_min = radarConfig.frame_rate_Hz * 60
    plotDataLow = plotData.copy()
    plotDataLow[:,1]  = filterData(plotDataLow[:,1], 5, 50, frames_per_min, order=4 )

    if savePlot:
        breathRate = plotDistances(plotDataLow, radarConfig.frame_rate_Hz, 'low-rate (1-25 BPM)'+name_suffix, dataPath)
    else:
        breathRate = plotDistances(plotDataLow, radarConfig.frame_rate_Hz, 'low-rate (1-25 BPM)', None)

    print("Breathrate : ", breathRate)
    # remove harmonics of breathrate
    plotDataHigh = plotData.copy()

    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*2, Q=5, fs=frames_per_min)
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*3, Q=5, fs=frames_per_min)
    plotDataHigh[:,1] = notchFilter(plotDataHigh[:,1], freq=breathRate*4, Q=5, fs=frames_per_min)

    plotDataHigh[:,1] = filterData(plotDataHigh[:,1],60, 150, frames_per_min, order=4 )

    if savePlot:
        heartrate = plotDistances(plotDataHigh, radarConfig.frame_rate_Hz, 'high-rate (25-100 BPM)'+name_suffix, dataPath)
    else:
        heartrate = plotDistances(plotDataHigh, radarConfig.frame_rate_Hz, 'high-rate (25-100 BPM)', None)
    print('Heartrate : ', heartrate)

def FFTidx(freq:float, dataLen:int, fs:float):
    return int(freq * dataLen * (1/fs) ) +1

LOW_RATE_MAX    = 60
HIGH_RATE_MIN   = LOW_RATE_MAX
HIGH_RATE_MAX   = 150
MAX_PLOT_BPM    = 200
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


def plotDistData(dataPath:str, savePlot:bool = False):
    if USE_ALT_DIST:
        dataFilename    = dataPath+"/radar_distance_alt.npy" # os.path.join(path,"radar_distance.py")
        name_suffix = ' - alt'
    else:
        dataFilename    = dataPath+"/radar_distance.npy" # os.path.join(path,"radar_distance.py")
        name_suffix = ''

    print("Processing : ", dataFilename)
    distData        = np.load(dataFilename)
    distCount       = distData.shape[0]
    print("> Distance values : ", distCount)

    radarConfig     = importRadarConfig(dataPath)
    framesPerMinute    = radarConfig.frame_rate_Hz * 60
    print("> Frames-per-minute : ", framesPerMinute)

    plotLen = int(radarConfig.frame_rate_Hz * PLOT_RANGE)        # plot data for 2 min
    ofs =  (distCount - plotLen) //2    # for testing get data from the middle
    plotData = distData[ofs:ofs+plotLen,:]
    plotDistDataChunkFFT(radarConfig, plotData, savePlot, dataPath)




if __name__ == '__main__':
    # plotDistData(radarFilePathes[4], False)
    for path in radarFilePathes: plotDistData(path, True)



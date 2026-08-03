'''
    radarLib - Library of functions for Radar-Data from BGT60xx

'''

from dataclasses import dataclass
import json
import os

import numpy as np
import scipy as sp
import scipy.signal as signal
from scipy.signal import get_window, detrend


# fixed names for the radar data files
DATA_NAME       = "radar.npy"
CONFIG_NAME     = "config.json"
TIMESTAMP_NAME  = "radar_timestamp.csv"
ANGLES_NAME     = "radar_angles.npy"

SPEED_OF_LIGHT = 300e6  # 300.000 km/s = 300.000.000 m/s

radarLib_verbose    = False     # if set to True PreProcessRadarData will print some infos

@dataclass
class radarConfigData:
    '''
    dataclass for the data in the config.json with addition of some derived values
    
    defining it as a dataclass enables auto-completion
    '''
    num_samples_per_chirp   : int
    sample_rate_Hz          : int
    chirp_time_s            : float # duration of one chirp = num_samples/ sample_rate
    fft_step_Hz             : float # step size for the FFT of one chirp = 1/chirp_time_s
    start_frequency_Hz      : float
    end_frequency_Hz        : float
    freq_change_speed       : float # how fast changes the frequency in a chirp = (end-start)/time
    frame_repetition_time_s : float
    frame_rate_Hz           : float # simply 1/frame_repetition_time_s
    #
    num_chirps_per_frame    : int
    chirp_repetition_time_s : float
    aaf_cutoff_Hz           : int
    hp_cutoff_Hz            : int
    if_gain_dB              : int
    mimo_mode               : str
    rx_antennas             : list
    tx_antennas             : list
    tx_power_level          : int


def importRadarConfig(dataPath : str) -> radarConfigData:
    """
    read the JSOM-file with the used configuration and return dataclass-object
    :param dataPath:
    :return:
    """
    if radarLib_verbose:
        print("importRadarConfig("+dataPath+")")
    config_filename = os.path.join(dataPath, CONFIG_NAME)
    # read JSON-File with configuration info dictionary
    with open(config_filename) as configJson:
        configData = json.load(configJson)['device_config'] ['fmcw_single_shape']
    
    # calculate derivative values often used
    freqStart       = configData['start_frequency_Hz']
    freqStop        = configData['end_frequency_Hz']
    smplPerChirp    = configData['num_samples_per_chirp']
    smplRate        = configData['sample_rate_Hz']
    frameRepTime    = configData['frame_repetition_time_s']

    frameRate       = 1/frameRepTime
    chirpTime       = smplPerChirp / smplRate       # length of one chirp in s
    freqChgSpeed    = (freqStop-freqStart) / chirpTime  # change of frequency in Hz/s
    fftStepHz       = 1 / chirpTime

    # extend dictionary with derivative values
    configData['frame_rate_Hz']     = frameRate
    configData['chirp_time_s']      = chirpTime
    configData['freq_change_speed'] = freqChgSpeed
    configData['fft_step_Hz']       = fftStepHz
    
    # convert dictionary to dataclass and return
    return radarConfigData(**configData)
    
# functions for processing of raw radar-data
# ------------------------------------------------------


def importRadarData(dataPath : str) -> tuple: #changed to tuple
    '''
    import radar-data and configuration
    :param dataPath:
    :return: dictionary with configuration for this recording and numpy-array with raw radar data
    '''
    dataFile = os.path.join(dataPath, DATA_NAME)
    timestampFile = os.path.join(dataPath, TIMESTAMP_NAME)

    # read JSON-File with configuration and convert to dataclass
    radarConfig = importRadarConfig(dataPath)
    # read radar data
    radarData = np.load(dataFile, mmap_mode= "r")  #memory mapping
    # read timestamps
    try:
        timestamps = np.loadtxt(timestampFile)
    except:
        # create dummy timestamps if there are no real timestamps
        timestamps = np.array(range(radarData.shape[0]), dtype=np.float64) * radarConfig.frame_repetition_time_s

    return radarConfig, radarData, timestamps


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
        chirp = chirp -  chirpMean
    # calculate real-FFT from chirp
    fftData = np.fft.rfft(chirp)

    return fftData


def processOneFrame(frame:np.ndarray, mean:bool=False, window:np.ndarray=None) -> np.ndarray:
    '''
    process data from one radar-frame by calculating the mean from the FFT of the chirps in frame
    
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

    if (window is not None) and (len(window) !=chirpSmpls):
        print("Error: Length of window has to be equal to numerber of samples in chirp!")
        return None


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

def processOneFrameAlt(frame:np.ndarray, mean:bool=False, window:np.ndarray=None) -> np.ndarray:
    '''
    process data from one radar-frame by calculating the FFT of the mean from chirps in frame

    :param frame:   2D-ndarray with raw radar data of one frame
    :param mean:    should the offset removed from the raw-data
    :param window:  window to use before fft
    :return:    FFT-Results of mean of chirps in frame
    '''

    frameShape = frame.shape
    if len(frameShape) != 2:
        print("Error: Framedata have to be 2D!")
        return None

    chirpCnt   = frameShape[0]  # number of chirps in one frame
    chirpSmpls = frameShape[1]  # number of samples in one chirp

    if (window is not None) and (len(window) !=chirpSmpls):
        print("Error: Length of window has to be equal to numerber of samples in chirp!")
        return None

    # calc mean of all chirps in one frame
    chirpSum = np.mean(frame, axis=0)
    # chirpSum = np.zeros(chirpSmpls, dtype=float)
    # for i in range(chirpCnt): chirpSum = chirpSum + frame[i]
    # chirpSum = chirpSum / chirpCnt

    # remove DC-offset
    if mean:
        chirpMean = chirpSum.mean()
        chirpSum = chirpSum -  chirpMean
    # apply window
    if window is not None:
        chirpSum = chirpSum * window
    # calculate real-FFT from chirp
    fftData = np.fft.rfft(chirpSum)
    return fftData

def calcFrequencyFromDistance(radarConfig:radarConfigData, distance:float) -> float:
    rtTime = (2*distance) / SPEED_OF_LIGHT  # ToF for&back in s
    freq   = radarConfig.freq_change_speed * rtTime          # change of frequency in one round trip
    return freq

def calcDistanceFromFrequency(radarConfig:radarConfigData, frequency:float) -> float:
    rtTime = frequency / radarConfig.freq_change_speed       # how long neede the signal
    distance = (rtTime /2) * SPEED_OF_LIGHT
    return distance

def calcFFTbinFromFrequency(radarConfig:radarConfigData, frequency:float) -> int:
    fftBin = frequency / radarConfig.fft_step_Hz
    return int(fftBin)



def preProcessRadarData(dataPath:str, distance:float, window:np.ndarray|str=None, frameMax:int=None, altCalculation:bool = False) -> np.ndarray:
    '''
    import raw radar-data, process frames and return numpy-array with timestamps and angles from FFTbins

    the array returned will have 1 line per frame with 4 columns:
    - timestamps : timestamp of the frame
    - FFTbin calculated from distance
    - FFTbin before the FFTbin calculated from distance
    - FFTbin after the FFTbin calculated from distance

    calculating and returning the values for the FFTbins before and after is at nearly no cost
    but give the options to test the neighbor bins without rerunning the pre-processin

    normally the result should be saved as a npy-file [ np.save(dataPath+"\\"+ANGLES_NAME, results) ]
    to avoid time-consuming recalculation at later processing steps

    :param dataPath:    where are the data?
    :param distance:    float|    distance between sensor and target/animal in m
    :param window:      optional, window or name of window-function for FFT of chirps
    :param frameMax:    optional, max number of frames to process
    :param altCalculation: True -> use alternate calculation
    '''
    TX_ANTENNA = 0
    RX_ANTENNA = 0

    radarConfig, data, timestamps = importRadarData(dataPath)


    # create window-array by name
    if (window is not None) and (type(window) ==str):
        winSize = radarConfig.num_samples_per_chirp
        window = get_window( window, winSize)

    # limit number of frames to process (for testing)
    frameCount = data.shape[0]
    if frameMax is not None:
        if frameCount >frameMax: frameCount = frameMax

    # prepare result-array
    freq = calcFrequencyFromDistance(radarConfig, distance)
    FFTbin  = calcFFTbinFromFrequency(radarConfig, freq)
    if FFTbin <=0:
        print("Invalid distance!")
        return None
    
    print(f"   [Target Locked] FFT Bin: {FFTbin} for Distance: {distance}m")
    # create NP-array zero initialised to speed up
    results = np.zeros([frameCount,4], dtype=float)

    # process frames and fill results array
    for frameNum in range(frameCount):
        if radarLib_verbose and (frameNum % 1000 ==0):
            print("> Frame : ",frameNum)
        if len(data.shape) ==5:
            frameRaw = data[frameNum, TX_ANTENNA, RX_ANTENNA].copy()  # select frame i, txAnt 0, rxAnt 0
        else:
            frameRaw = data[frameNum, RX_ANTENNA].copy()  # select frame i, txAnt 0, rxAnt 0

        if altCalculation:
            frameFFT = processOneFrameAlt(frameRaw, mean=True, window=window)  # should use window to get better FFT results
        else:
            frameFFT = processOneFrame(frameRaw, mean=True, window=window)  # should use window to get better FFT results

        phi = np.angle( frameFFT[FFTbin-1:FFTbin+2] )

        #print("Timestamps shape/type:", type(timestamps), timestamps.shape)
        tmp =  np.insert(phi, 0, timestamps[frameNum])
        # Temporary debug snippet to put right before line 304 in radarLib.py:
        
        results[frameNum] = tmp
    
    # unwrap phase values
    results[:, 1:] = np.unwrap(results[:, 1:], axis=0) #alternative to exclude timestamps

    if radarLib_verbose:
        print(">Done! ",frameCount," frames preprocessed")
    return results

def saveRadarAngles(radarAngles:np.ndarray, dataPath:str):
    anglesFile = os.path.join(dataPath, ANGLES_NAME)
    np.save(anglesFile, radarAngles)


# ---------------------------------------------------------------------------------------------------------
# functions for preprocessed data
# ---------------------------------------------------------------------------------------------------------


def importRadarPhasedata(dataPath:str):
    '''
    import preprocessed radar-data and radar-configuration from datapath
    :param dataPath:    where to find files
    :return:            deviceConfig, preprocessed radar-data
    '''
    anglesFile = os.path.join(dataPath, ANGLES_NAME)    
    radarConfig = importRadarConfig(dataPath) # read JSON-File with configuration
    # read radar data
    phaseData = np.load(anglesFile)
    return radarConfig, phaseData


def calcDisplacement(radarConfig:radarConfigData, phaseData:np.ndarray) -> np.ndarray:
    '''
    calculate the distance from the phase shift between consecutive frames

    the calculated distance is relative to the distance equiv to the selected FFT-bin
    for normal use, only the changes of distance are used -> remove the mean of the data
    
    :param radarconfig:     radar-configuration
    :param phaseData:       Numpy-Array with phase data
    :return:    Numpy-Array with the relative distances (one value less than the phase data)
    '''
    # calculate the phase-change
    phaseDelta = phaseData[:-1].copy()
    phaseDelta = phaseDelta -  phaseData[1:]
    # from phase change calculate the freq difference (to the used FFT-bin=)
    freqDelta = (phaseDelta / (2 *np.pi)) * radarConfig.frame_rate_Hz
    # then calculate the RT-time difference form freq difference
    timeDelta = freqDelta / radarConfig.freq_change_speed
    # and convert to distance in m
    distance  = (timeDelta /2) *SPEED_OF_LIGHT
    return distance


def calcDisplacementAlt(radarConfig: radarConfigData, phaseData: np.ndarray) -> np.ndarray:
    '''
    calculate the distance direct from phase

    :param radarconfig:     radar-configuration
    :param phaseData:       Numpy-Array with phase data
    :return:    Numpy-Array with the relative distances (one value less than the phase data)
    '''
    phaseFixed = detrend(phaseData)
    meanFreq   = (radarConfig.start_frequency_Hz+radarConfig.end_frequency_Hz) /2
    meanWavelength  = SPEED_OF_LIGHT / meanFreq
    distPhase  = (phaseFixed / (4 * np.pi)) * meanWavelength  # calculate distance from phase

    return distPhase - np.mean(distPhase)



def filterData(data:np.ndarray, lowFreq:float, highFreq:float, fs:float, order : int =4):
    '''
    filter data with a butterworth bandpass filter
    :param data:
    :param lowFreq:
    :param highFreq:
    :param fs:
    :param order:
    :return:
    '''
    nyq = fs  /2
    if lowFreq is None:
        sos = signal.butter(order, highFreq / nyq, btype="lowpass", output='sos')
    else:
        sos = signal.butter(order, [lowFreq/nyq, highFreq/nyq], btype="bandpass", output='sos')
    return signal.sosfilt(sos, data)

def notchFilter(data:np.ndarray, freq:float, fs:float, Q : float =20):
    '''
    filter data with a 2nd-prder notch filter
    :param data:
    :param freq:
    :param fs:
    :param Q:
    :return:
    '''
    b,a = signal.iirnotch(freq, Q, fs)
    return signal.lfilter( b, a, data.copy())


def normalize(data:np.ndarray) -> np.ndarray:
    return (data - np.mean(data)) / np.std(data)

def calcRate(distData:np.ndarray, framerate_Hz:float) -> float:
    distNorm    = normalize(distData )
    distWindowed= distNorm * signal.get_window(window='blackman', Nx=len(distData))
    spectrum    = sp.fft.rfft( distWindowed )
    specAbs     = np.abs( spectrum )
    specAbs[0]  = 0     # suppress DC / near-DC leakage so it can never win the argmax
    specFreq    = sp.fft.rfftfreq(len(distData), 1/(framerate_Hz*60))  # display frequencies in 1/min

    peakIdx = np.argmax(specAbs)
    peakFreq = specFreq[peakIdx]
    
    return peakFreq



##peak to floor, signal to noise ratio

def calcRateWithSNR(distData, framerate_Hz, exclude_bins=3):
    distNorm = normalize(distData)
    distWindowed = distNorm * signal.get_window(window='blackman', Nx=len(distData))
    spectrum = sp.fft.rfft(distWindowed)
    specAbs = np.abs(spectrum)
    specAbs[0] = 0  # suppress DC
    specFreq = sp.fft.rfftfreq(len(distData), 1/(framerate_Hz*60))

    peakIdx = np.argmax(specAbs)
    peakFreq = specFreq[peakIdx]
    peakPower = specAbs[peakIdx] ** 2

    # noise floor = median power of everything EXCEPT a small window around the peak
    # (excluding the peak's own bins keeps the floor estimate from being biased upward by the peak itself)
    mask = np.ones(len(specAbs), dtype=bool)
    lo, hi = max(0, peakIdx-exclude_bins), min(len(specAbs), peakIdx+exclude_bins+1)
    mask[lo:hi] = False
    noiseFloor = np.median(specAbs[mask] ** 2)

    snr_dB = 10 * np.log10(peakPower / max(noiseFloor, 1e-30))

    return {"rate": peakFreq, "snr_dB": snr_dB}



from scipy.signal import find_peaks
peaks, props = find_peaks(specAbs, prominence=0)
best = peaks[np.argmax(props["prominences"])]
prominence = props["prominences"][np.argmax(props["prominences"])]
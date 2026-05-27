'''
    radarLib - Library to process Radar-Data from BGT60xx

'''

import numpy as np
import json
from scipy.signal import get_window

# functions for processing of raw radar-data
# ------------------------------------------------------

DATA_NAME       = "radar.npy"
CONFIG_NAME     = "config.json"
TIMESTAMP_NAME  = "radar_timestamp.csv"
ANGLES_NAME     = "radar_angles.npy"

SPEED_OF_LIGHT = 300e6  # 300.000 km/s = 300.000.000 m/s

def importRadarData(dataPath : str) -> (dict, np.ndarray, np.ndarray):
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
    data = np.load(dataFile)
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

def calcDistanceFromFrequency(radarconfig:dict, frequency:float) -> float:
    freqStart   = radarconfig['start_frequency_Hz']
    freqStop    = radarconfig['end_frequency_Hz']
    smplPerChirp= radarconfig['num_samples_per_chirp']
    smplRate    = radarconfig['sample_rate_Hz']
    chirpTime   = smplPerChirp / smplRate       # length of one chirp in s
    freqChgSpeed= (freqStop-freqStart) / chirpTime  # change of frequency in Hz/s
    rtTime = frequency / freqChgSpeed       # how long neede the signal
    distance = (rtTime /2) * SPEED_OF_LIGHT
    return distance

def calcFFTbinFromFrequency(radarconfig:dict, frequency:float) -> int:
    smplPerChirp = radarconfig['num_samples_per_chirp']
    smplRate = radarconfig['sample_rate_Hz']
    chirpTime = smplPerChirp / smplRate  # length of one chirp in s
    fftStepHz = 1 / chirpTime  # freq-diff between 2 FFT-bins
    
    fftBin = frequency / fftStepHz
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
    :param distance:    float|list of float     distance between sensor and target/animal in m
    :param window:      optional, window or name of window-function for FFT of chirps
    :param frameMax:    optional, max number of frames to process
    '''
    TX_ANTENNA = 0
    RX_ANTENNA = 0

    config, data, timestamp = importRadarData(dataPath)
    # create window-array by name
    if (window is not None) and (type(window) ==str):
        winSize = config['num_samples_per_chirp']
        window = get_window( window, winSize)

    # limit number of frames to process (for testing)
    frameCount = data.shape[0]
    if frameMax is not None:
        if frameCount >frameMax: frameCount = frameMax

    # prepare result-array
    if type(distance) == list:
        binCnt = len(distance)
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

# ---------------------------------------------------------------------------------------------------------
# functions for preprocessed data
# ---------------------------------------------------------------------------------------------------------
def saveRadarAngels( radarAngels:np.ndarray, dataPath:str):
    np.save(dataPath + "\\" + ANGLES_NAME, radarAngels)

def importRadarAngels(dataPath:str):
    '''
    import preprocessed radar-data and radar-configuration from datapath
    :param dataPath:    where to find files
    :return:            deviceConfig, preprocessed radar-data
    '''
    configFile  = dataPath + "\\" + CONFIG_NAME
    angelsFile  = dataPath + "\\" + ANGLES_NAME
    # read JSON-File with configuration
    with open(configFile) as configJson:
        deviceConfig = json.load(configJson)['device_config'] ['fmcw_single_shape']
    # read radar data
    angels = np.load(angelsFile)
    return deviceConfig, angels


def calcDisplacement(phaseData:np.ndarray, radarconfig:dict) -> np.ndarray:
    '''
    calculate the distance from the phase shift between consecutive frames
    
    the calculated distance is relative to the distance from the selected FFT-bin
    
    :param phaseData:       Numpy-Array with phase data
    :param radarconfig:     radarconfiguration
    :return:    Numpy-Array with the relative distances (one value less than the phase data)
    '''
    # calculate some values from the radarconfig
    frameRepTime = radarconfig["frame_repetition_time_s"]
    frameRate    = 1/frameRepTime
    freqStart   = radarconfig['start_frequency_Hz']
    freqStop    = radarconfig['end_frequency_Hz']
    smplPerChirp= radarconfig['num_samples_per_chirp']
    smplRate    = radarconfig['sample_rate_Hz']
    chirpTime   = smplPerChirp / smplRate       # length of one chirp in s
    freqChgSpeed= (freqStop-freqStart) / chirpTime  # change of frequency in Hz/s
    # calculate the phase-change
    phaseDelta = phaseData[:-1].copy - phaseData[1:]
    # from phase change calculate the freq difference (to the used FFT-bin=)
    freqDelta = (phaseDelta / (2 *np.pi)) * frameRate
    # then calculate the RT-time difference form freq difference
    timeDelta = freqDelta / freqChgSpeed
    # and convert to distance in m
    distance  = (timeDelta /2) /SPEED_OF_LIGHT
    return distance
    
    

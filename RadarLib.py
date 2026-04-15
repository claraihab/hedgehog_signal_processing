'''
    radarLib - Library to process Radar-Data from BGT60xx

'''

import numpy as np
import json

DATA_NAME   = "radar.npy"
CONFIG_NAME = "config.json"

def importRadarData(dataPath : str) -> (dict, np.ndarray):
    '''
    import radar-data and configuration
    :param dataPath:
    :return: dictionary with configuration for this recording and numpy-array with raw radar data
    '''
    configFile = dataPath + "\\" + CONFIG_NAME
    dataFile    = dataPath + "\\" + DATA_NAME
    # read JSON-File with configuration
    with open(configFile) as configJson:
        deviceConfig = json.load(configJson)['device_config'] ['fmcw_single_shape']
    # read radar data
    data = np.load(dataFile)

    return deviceConfig, data

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



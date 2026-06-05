'''
    radarLib - Library of functions for Radar-Data from BGT60xx

'''

from dataclasses import make_dataclass, dataclass
import json
import numpy as np
from scipy.signal import get_window

# fixed names for the radar data files
DATA_NAME       = "radar.npy"
CONFIG_NAME     = "config.json"
TIMESTAMP_NAME  = "radar_timestamp.csv"
ANGLES_NAME     = "radar_angles.npy"

SPEED_OF_LIGHT = 300e6  # 300.000 km/s = 300.000.000 m/s

@dataclass
class radarConfigData:
    '''
    dataclass for the data in the config.json with addition of some derivated values
    
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


def importRadarConfig(dataPath : str) -> radarConfigData:
    # read JSON-File with configuration info dictionary
    with open(dataPath + "\\" + CONFIG_NAME) as configJson:
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
    configData['frameRate']     = frameRate
    configData['chirpTime']     = chirpTime
    configData['freqChgSpeed']  = freqChgSpeed
    configData['fftStepHz']     = fftStepHz
    
    # convert dictionary to dataclass and return
    return radarConfigData(**configData)
    
# functions for processing of raw radar-data
# ------------------------------------------------------


def importRadarData(dataPath : str) -> (dict, np.ndarray, np.ndarray):
    '''
    import radar-data and configuration
    :param dataPath:
    :return: dictionary with configuration for this recording and numpy-array with raw radar data
    '''
    dataFile    = dataPath + "\\" + DATA_NAME
    timestampFile = dataPath + "\\" + TIMESTAMP_NAME
    # read JSON-File with configuration and convert to dataclass
    radarConfig = importRadarConfig(dataPath)
    # read radar data
    data = np.load(dataFile)
    # read timestamps
    timestamps = np.loadtxt(timestampFile)

    return radarConfig, data, timestamps

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
    process data from one radar-frame by calculating the mean from chirps in frame
    
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



def preProcessRadarData(dataPath:str, distance:float, window:np.ndarray|str=None, frameMax:int=None) -> np.ndarray:
    '''
    import raw radar-data, process frames and return numpy-array with timestamps and angles from FFTbins

    the array returned will have 1 lien per frame with 4 columns:
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
    
    # create NP-array zero initialised to speed up
    results = np.zeros([frameCount,4], dtype=float)

    # process frames and fill results array
    for frameNum in range(frameCount):
        frameRaw = data[frameNum, TX_ANTENNA, RX_ANTENNA].copy()  # select frame i, txAnt 0, rxAnt 0
        frameFFT = processOneFrame(frameRaw, mean=True, window=window)  # should use window to get better FFT results

        phi = np.angle( frameFFT[FFTbin-1:FFTbin+1] )
        results[frameNum] = np.insert(phi, 0, timestamps[frameNum])
        
    return results

# ---------------------------------------------------------------------------------------------------------
# functions for preprocessed data
# ---------------------------------------------------------------------------------------------------------
def saveRadarAngels( radarAngels:np.ndarray, dataPath:str):
    np.save(dataPath + "\\" + ANGLES_NAME, radarAngels)

def importRadarPhasedata(dataPath:str):
    '''
    import preprocessed radar-data and radar-configuration from datapath
    :param dataPath:    where to find files
    :return:            deviceConfig, preprocessed radar-data
    '''
    angelsFile  = dataPath + "\\" + ANGLES_NAME
    # read JSON-File with configuration
    radarConfig = importRadarConfig(dataPath)
    # read radar data
    phaseData = np.load(angelsFile)
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
    phaseDelta = phaseData[:-1].copy - phaseData[1:]
    # from phase change calculate the freq difference (to the used FFT-bin=)
    freqDelta = (phaseDelta / (2 *np.pi)) * radarConfig.frame_rate_Hz
    # then calculate the RT-time difference form freq difference
    timeDelta = freqDelta / radarConfig.freq_change_speed
    # and convert to distance in m
    distance  = (timeDelta /2) /SPEED_OF_LIGHT
    return distance
    
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
    min_idx = min_Hz * timebase
    max_idx = max_Hz * timebase
    data = FFT_data.copy()  # reserve original data
    if min_idx >0:
        data[0:min_idx] = 0
    if max_idx >0 and max_idx <=len(data):
        data[max_idx:]  = 0
    return data
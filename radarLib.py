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



def findTargetBin(dataPath:str, scanSeconds:float=120.0, minDistance:float=0.10, maxDistance:float=1.0,
                   searchBand:list=None, altCalculation:bool=False, window:np.ndarray|str=None) -> dict:
    '''
    automatically find the range bin with the strongest periodic vital-sign activity, instead
    of relying on a manually measured distance. A live target's bin shows periodic phase
    modulation from breathing/heartbeat; static clutter (walls, cage bars) reflects strongly
    but without that periodic modulation - so raw reflection strength alone is not enough.

    Strategy: take a short window of frames, unwrap the phase of every candidate bin over
    that window, bandpass-filter it to searchBand, and pick the bin with the highest energy
    (variance) in that band.

    :param dataPath:        where to find the recording
    :param scanSeconds:     how much data (in seconds) to use for the scan. Needs to be long
                             enough to contain several cycles of the slowest signal in
                             searchBand - heartbeat (a few bpm even in deep torpor) is a more
                             reliable pick for this than breathing, which can be as slow as
                             0.2 cycles/min and would need a much longer scan window to show
                             up reliably.
    :param minDistance:     skip candidate bins closer than this (m) - avoids TX/RX leakage
                             near the sensor, which otherwise tends to dominate raw energy.
    :param maxDistance:     skip candidate bins farther than this (m)
    :param searchBand:      [lowFreq, highFreq] in cycles/min to bandpass each candidate bin
                             before scoring it. Defaults to a wide band covering both torpid
                             and awake heart rate (2-280 bpm) so it works for either dataset.
    :param altCalculation:  True -> use processOneFrameAlt instead of processOneFrame
    :param window:          optional window function for the chirp FFT (same as preProcessRadarData)
    :return:    dict with keys: bin (int), distance (float, m), energy_per_bin (np.ndarray),
                bins (np.ndarray of candidate bin indices, for diagnostic plotting)
    '''
    if searchBand is None:
        searchBand = [2.0, 280.0]   # covers HIB_HEART_BAND through AWAKE_HEART_BAND

    TX_ANTENNA = 0
    RX_ANTENNA = 0

    radarConfig, data, timestamps = importRadarData(dataPath)

    if (window is not None) and (type(window) == str):
        window = get_window(window, radarConfig.num_samples_per_chirp)

    scanFrames = min(int(scanSeconds * radarConfig.frame_rate_Hz), data.shape[0])
    # use a chunk from the middle of the recording rather than the very start, to avoid any
    # startup transient in the sensor/recording
    startFrame = max(0, (data.shape[0] - scanFrames) // 2)

    minBin = max(1, calcFFTbinFromFrequency(radarConfig, calcFrequencyFromDistance(radarConfig, minDistance)))
    maxBin = calcFFTbinFromFrequency(radarConfig, calcFrequencyFromDistance(radarConfig, maxDistance))

    print(f"   [Bin Scan] scanning bins {minBin}-{maxBin} over {scanFrames} frames "
          f"({scanFrames/radarConfig.frame_rate_Hz:.1f}s) starting at frame {startFrame}")

    # collect the complex range-FFT value at each candidate bin, for each scanned frame
    candidateBins = np.arange(minBin, maxBin+1)
    complexByBin = np.zeros((scanFrames, len(candidateBins)), dtype=complex)

    for i, frameNum in enumerate(range(startFrame, startFrame+scanFrames)):
        if len(data.shape) == 5:
            frameRaw = data[frameNum, TX_ANTENNA, RX_ANTENNA].copy()
        else:
            frameRaw = data[frameNum, RX_ANTENNA].copy()

        if altCalculation:
            frameFFT = processOneFrameAlt(frameRaw, mean=True, window=window)
        else:
            frameFFT = processOneFrame(frameRaw, mean=True, window=window)

        complexByBin[i, :] = frameFFT[candidateBins]

    frames_per_min = int(radarConfig.frame_rate_Hz * 60)
    energyPerBin = np.zeros(len(candidateBins))

    for col in range(len(candidateBins)):
        phase = np.unwrap(np.angle(complexByBin[:, col]))
        phaseFiltered = filterData(phase, searchBand[0], searchBand[1], frames_per_min, order=4)
        energyPerBin[col] = np.var(phaseFiltered)

    bestIdx  = np.argmax(energyPerBin)
    bestBin  = int(candidateBins[bestIdx])
    bestFreq = bestBin * radarConfig.fft_step_Hz
    bestDist = calcDistanceFromFrequency(radarConfig, bestFreq)

    print(f"   [Bin Scan] best bin: {bestBin}  (~{bestDist:.3f} m)")

    return {
        "bin": bestBin,
        "distance": bestDist,
        "energy_per_bin": energyPerBin,
        "bins": candidateBins,
    }


def preProcessRadarData(dataPath:str, distance:float=None, targetBin:int=None, window:np.ndarray|str=None, frameMax:int=None, altCalculation:bool = False) -> np.ndarray:
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
    :param distance:    float|    distance between sensor and target/animal in m. Ignored if targetBin is given.
    :param targetBin:   int|      explicit FFT bin to lock onto (e.g. from findTargetBin()) - bypasses the distance lookup
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
    if targetBin is not None:
        FFTbin = targetBin
    else:
        if distance is None:
            raise ValueError("preProcessRadarData: need either distance or targetBin")
        freq = calcFrequencyFromDistance(radarConfig, distance)
        FFTbin  = calcFFTbinFromFrequency(radarConfig, freq)
    if FFTbin <=0:
        print("Invalid distance!")
        return None
    
    if targetBin is not None:
        print(f"   [Target Locked] FFT Bin: {FFTbin} (explicit targetBin)")
    else:
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


def calcRate(distData: np.ndarray, framerate_Hz: float, zero_pad_factor: float = 2.0) -> float:
    distNorm = normalize(distData)
    distWindowed = distNorm * signal.get_window(window='blackman', Nx=len(distData))
    nfft = int(len(distData) * zero_pad_factor)
    spectrum = sp.fft.rfft(distWindowed, n=nfft)
    specAbs = np.abs(spectrum)
    specAbs[0] = 0
    specFreq = sp.fft.rfftfreq(nfft, 1 / (framerate_Hz * 60))

    peakIdx = np.argmax(specAbs)
    return specFreq[peakIdx]


def calcRateSNR(distData: np.ndarray, framerate_Hz: float, exclude_bins: int = 3, n_harmonics: int = 4,
                 min_freq: float = None, max_freq: float = None, prominence_pad_frac: float = 0.3) -> dict:
    '''
    Peak-picking plus SNR and prominence estimation.

    Harmonics are matched by TARGET FREQUENCY (k * peakFreq), not by multiplying the array
    index - multiplying the index only gives the right harmonic bin when index 0 corresponds
    to frequency 0, which is false once min_freq truncation is active.

    Prominence is computed on a window padded moderately beyond [min_freq, max_freq] (not
    the tight truncated window, not the full spectrum): tight truncation forces prominence
    to 0 whenever the peak sits at the window's edge (no neighbor to compare against); the
    full spectrum re-admits the bandpass filter's own crushed-to-near-zero stopband,
    inflating prominence toward ~1.0 for nearly everything. Padding by a fraction of the
    band's own width avoids both failure modes.
    '''
    distNorm     = normalize(distData)
    distWindowed = distNorm * signal.get_window(window='blackman', Nx=len(distData))
    spectrum     = sp.fft.rfft(distWindowed)
    specAbs      = np.abs(spectrum)
    specAbs[0]   = 0
    specFreq     = sp.fft.rfftfreq(len(distData), 1 / (framerate_Hz * 60))

    if (min_freq is not None) or (max_freq is not None):
        keep = np.ones(len(specFreq), dtype=bool)
        if min_freq is not None:
            keep &= (specFreq >= min_freq)
        if max_freq is not None:
            keep &= (specFreq <= max_freq)
        specAbs_restricted  = specAbs[keep]
        specFreq_restricted = specFreq[keep]
    else:
        specAbs_restricted, specFreq_restricted = specAbs, specFreq

    if len(specAbs_restricted) == 0:
        return {"rate": np.nan, "snr_dB": np.nan, "prominence": np.nan, "prominence_ratio": np.nan}

    peakIdx  = np.argmax(specAbs_restricted)
    peakFreq = specFreq_restricted[peakIdx]

    # harmonics by target frequency - correct regardless of truncation/offset
    harmonicIdx = []
    for k in range(1, n_harmonics + 1):
        target_freq = k * peakFreq
        if specFreq_restricted[0] <= target_freq <= specFreq_restricted[-1]:
            closest_bin = int(np.argmin(np.abs(specFreq_restricted - target_freq)))
            if closest_bin not in harmonicIdx:
                harmonicIdx.append(closest_bin)

    signalPower = np.mean([specAbs_restricted[idx] ** 2 for idx in harmonicIdx]) if harmonicIdx else specAbs_restricted[peakIdx] ** 2

    mask = np.ones(len(specAbs_restricted), dtype=bool)
    for idx in harmonicIdx:
        lo, hi = max(0, idx - exclude_bins), min(len(specAbs_restricted), idx + exclude_bins + 1)
        mask[lo:hi] = False

    noiseFloor = np.median(specAbs_restricted[mask] ** 2) if mask.any() else 1e-30
    snr_dB = 10 * np.log10(signalPower / max(noiseFloor, 1e-30))

    # prominence: modestly padded window 
    if (min_freq is not None) and (max_freq is not None):
        band_width = max_freq - min_freq
        pad = band_width * prominence_pad_frac
        prom_min = max(0, min_freq - pad)
        prom_max = max_freq + pad
        keep_prom = (specFreq >= prom_min) & (specFreq <= prom_max)
        specAbs_prom, specFreq_prom = specAbs[keep_prom], specFreq[keep_prom]
    else:
        specAbs_prom, specFreq_prom = specAbs_restricted, specFreq_restricted

    peakIdx_prom = int(np.argmin(np.abs(specFreq_prom - peakFreq)))
    prominence = signal.peak_prominences(specAbs_prom, [peakIdx_prom])[0][0]
    prominence_ratio = prominence / specAbs_prom[peakIdx_prom] if specAbs_prom[peakIdx_prom] > 0 else 0.0

    return {"rate": peakFreq, "snr_dB": snr_dB, "prominence": prominence, "prominence_ratio": prominence_ratio}






def calcRateSNR(distData: np.ndarray, framerate_Hz: float, exclude_bins: int = 3, n_harmonics: int = 4,
                min_freq: float = None, max_freq: float = None, prominence_pad_frac: float = 0.3,
                zero_pad_factor: float = 2.0) -> dict:
    '''
    Peak-picking plus SNR and prominence estimation with zero-padding.
    
    `zero_pad_factor=2.0` zero-pads the signal to 2x its original length prior to the FFT,
    interpolating spectral bins to increase digital frequency resolution.
    '''
    distNorm = normalize(distData)
    distWindowed = distNorm * signal.get_window(window='blackman', Nx=len(distData))
    
    # Apply zero-padding by setting FFT length nfft = len(distData) * zero_pad_factor
    nfft = int(len(distData) * zero_pad_factor)
    spectrum = sp.fft.rfft(distWindowed, n=nfft)
    specAbs = np.abs(spectrum)
    specAbs[0] = 0
    specFreq = sp.fft.rfftfreq(nfft, 1 / (framerate_Hz * 60))

    if (min_freq is not None) or (max_freq is not None):
        keep = np.ones(len(specFreq), dtype=bool)
        if min_freq is not None:
            keep &= (specFreq >= min_freq)
        if max_freq is not None:
            keep &= (specFreq <= max_freq)
        specAbs_restricted = specAbs[keep]
        specFreq_restricted = specFreq[keep]
    else:
        specAbs_restricted, specFreq_restricted = specAbs, specFreq

    if len(specAbs_restricted) == 0:
        return {"rate": np.nan, "snr_dB": np.nan, "prominence": np.nan, "prominence_ratio": np.nan}

    peakIdx = np.argmax(specAbs_restricted)
    peakFreq = specFreq_restricted[peakIdx]

    harmonicIdx = []
    for k in range(1, n_harmonics + 1):
        target_freq = k * peakFreq
        if specFreq_restricted[0] <= target_freq <= specFreq_restricted[-1]:
            closest_bin = int(np.argmin(np.abs(specFreq_restricted - target_freq)))
            if closest_bin not in harmonicIdx:
                harmonicIdx.append(closest_bin)

    signalPower = np.mean([specAbs_restricted[idx] ** 2 for idx in harmonicIdx]) if harmonicIdx else specAbs_restricted[peakIdx] ** 2

    mask = np.ones(len(specAbs_restricted), dtype=bool)
    for idx in harmonicIdx:
        lo, hi = max(0, idx - exclude_bins), min(len(specAbs_restricted), idx + exclude_bins + 1)
        mask[lo:hi] = False

    noiseFloor = np.median(specAbs_restricted[mask] ** 2) if mask.any() else 1e-30
    snr_dB = 10 * np.log10(signalPower / max(noiseFloor, 1e-30))

    if (min_freq is not None) and (max_freq is not None):
        band_width = max_freq - min_freq
        pad = band_width * prominence_pad_frac
        prom_min = max(0, min_freq - pad)
        prom_max = max_freq + pad
        keep_prom = (specFreq >= prom_min) & (specFreq <= prom_max)
        specAbs_prom, specFreq_prom = specAbs[keep_prom], specFreq[keep_prom]
    else:
        specAbs_prom, specFreq_prom = specAbs_restricted, specFreq_restricted

    peakIdx_prom = int(np.argmin(np.abs(specFreq_prom - peakFreq)))
    prominence = signal.peak_prominences(specAbs_prom, [peakIdx_prom])[0][0]
    prominence_ratio = prominence / specAbs_prom[peakIdx_prom] if specAbs_prom[peakIdx_prom] > 0 else 0.0

    return {"rate": peakFreq, "snr_dB": snr_dB, "prominence": prominence, "prominence_ratio": prominence_ratio}




def calcRateSNR2(distData: np.ndarray, framerate_Hz: float, exclude_bins: int = 3, n_harmonics: int = 4,
                min_freq: float = None, max_freq: float = None, prominence_pad_frac: float = 0.3) -> dict:
    '''
    Peak-picking plus SNR and prominence estimation.
    '''
    distNorm = normalize(distData)
    distWindowed = distNorm * signal.get_window(window='blackman', Nx=len(distData))
    spectrum = sp.fft.rfft(distWindowed)
    specAbs = np.abs(spectrum)
    specAbs[0] = 0
    specFreq = sp.fft.rfftfreq(len(distData), 1 / (framerate_Hz * 60))

    # Restrict search band only for finding the fundamental peak
    if (min_freq is not None) or (max_freq is not None):
        keep = np.ones(len(specFreq), dtype=bool)
        if min_freq is not None:
            keep &= (specFreq >= min_freq)
        if max_freq is not None:
            keep &= (specFreq <= max_freq)
        specAbs_restricted = specAbs[keep]
        specFreq_restricted = specFreq[keep]
    else:
        keep = np.ones(len(specFreq), dtype=bool)
        specAbs_restricted, specFreq_restricted = specAbs, specFreq

    if len(specAbs_restricted) == 0:
        return {"rate": np.nan, "snr_dB": np.nan, "prominence": np.nan, "prominence_ratio": np.nan}

    # 1. Fundamental peak selection
    peakIdx_restricted = np.argmax(specAbs_restricted)
    peakFreq = specFreq_restricted[peakIdx_restricted]

    # 2. Harmonics search in FULL spectrum (Fixes Issue 3) & window leakage masking (Fixes Issue 2)
    signal_mask = np.zeros(len(specFreq), dtype=bool)
    
    for k in range(1, n_harmonics + 1):
        target_freq = k * peakFreq
        if specFreq[0] <= target_freq <= specFreq[-1]:
            closest_bin = int(np.argmin(np.abs(specFreq - target_freq)))
            # Capture the full main lobe of the window around each harmonic bin
            lo = max(0, closest_bin - exclude_bins)
            hi = min(len(specFreq), closest_bin + exclude_bins + 1)
            signal_mask[lo:hi] = True

    # 3. Sum signal power across all signal bins (Fixes Issue 1)
    signalPower = np.sum(specAbs[signal_mask] ** 2)
    n_signal_bins = np.sum(signal_mask)

    # 4. Noise floor calculation with bandwidth normalization (Fixes Issue 4)
    noise_mask = keep & (~signal_mask)
    noise_bins_power = specAbs[noise_mask] ** 2

    if len(noise_bins_power) > 0:
        median_noise_per_bin = np.median(noise_bins_power)
    else:
        median_noise_per_bin = 1e-30

    # Scale per-bin noise power by total number of signal bins
    total_noise_power = median_noise_per_bin * n_signal_bins
    snr_dB = 10 * np.log10(signalPower / max(total_noise_power, 1e-30))

    # 5. Prominence calculation with local peak validation (Fixes Issue 5)
    if (min_freq is not None) and (max_freq is not None):
        band_width = max_freq - min_freq
        pad = band_width * prominence_pad_frac
        prom_min = max(0, min_freq - pad)
        prom_max = max_freq + pad
        keep_prom = (specFreq >= prom_min) & (specFreq <= prom_max)
        specAbs_prom, specFreq_prom = specAbs[keep_prom], specFreq[keep_prom]
    else:
        specAbs_prom, specFreq_prom = specAbs, specFreq

    peakIdx_prom = int(np.argmin(np.abs(specFreq_prom - peakFreq)))

    # Ensure peakIdx_prom is a strict local maximum before running peak_prominences
    is_local_max = (
        0 < peakIdx_prom < len(specAbs_prom) - 1
        and specAbs_prom[peakIdx_prom] >= specAbs_prom[peakIdx_prom - 1]
        and specAbs_prom[peakIdx_prom] >= specAbs_prom[peakIdx_prom + 1]
    )

    if is_local_max:
        prominence = float(signal.peak_prominences(specAbs_prom, [peakIdx_prom])[0][0])
    else:
        prominence = float(specAbs_prom[peakIdx_prom] - np.min(specAbs_prom))

    prominence_ratio = prominence / specAbs_prom[peakIdx_prom] if specAbs_prom[peakIdx_prom] > 0 else 0.0

    return {
        "rate": peakFreq,
        "snr_dB": snr_dB,
        "prominence": prominence,
        "prominence_ratio": prominence_ratio
    }




def calcRateSNR3(distData: np.ndarray, framerate_Hz: float, exclude_bins: int = 3, n_harmonics: int = 4,
                min_freq: float = None, max_freq: float = None, prominence_pad_frac: float = 0.3,
                zero_pad_factor: float = 2.0) -> dict:
    '''
    Peak-picking plus SNR and prominence estimation with zero-padding.
    '''
    distNorm = normalize(distData)
    distWindowed = distNorm * signal.get_window(window='blackman', Nx=len(distData))
    
    # 1. Zero-padding setup
    nfft = int(len(distData) * zero_pad_factor)
    spectrum = sp.fft.rfft(distWindowed, n=nfft)
    specAbs = np.abs(spectrum)
    specAbs[0] = 0
    specFreq = sp.fft.rfftfreq(nfft, 1 / (framerate_Hz * 60))

    # Scale exclude_bins to match the zero-padded frequency bin spacing
    effective_exclude = int(np.ceil(exclude_bins * zero_pad_factor))

    # 2. Fundamental peak selection within restricted search band
    if (min_freq is not None) or (max_freq is not None):
        keep = np.ones(len(specFreq), dtype=bool)
        if min_freq is not None:
            keep &= (specFreq >= min_freq)
        if max_freq is not None:
            keep &= (specFreq <= max_freq)
        specAbs_restricted = specAbs[keep]
        specFreq_restricted = specFreq[keep]
    else:
        keep = np.ones(len(specFreq), dtype=bool)
        specAbs_restricted, specFreq_restricted = specAbs, specFreq

    if len(specAbs_restricted) == 0:
        return {"rate": np.nan, "snr_dB": np.nan, "prominence": np.nan, "prominence_ratio": np.nan}

    peakIdx_restricted = np.argmax(specAbs_restricted)
    peakFreq = specFreq_restricted[peakIdx_restricted]

    # 3. Harmonic Masking & Integration over full spectrum
    signal_mask = np.zeros(len(specFreq), dtype=bool)
    
    for k in range(1, n_harmonics + 1):
        target_idx = peakIdx_restricted *k
        if target_idx >= len(specFreq):
            break   # abort loop because this and all higher harmonics out of FFT range
        # create mask around harmonic of peak
        lo = max(0, target_idx - effective_exclude)
        hi = min(len(specFreq), target_idx + effective_exclude + 1)
        signal_mask[lo:hi] = True

    # Total signal power integrated over main lobes
    signalPower = np.sum(specAbs[signal_mask] ** 2)
    n_signal_bins = np.sum(signal_mask)

    # 4. Noise power normalized to signal bandwidth
    noise_mask = keep & (~signal_mask)
    noise_bins_power = specAbs[noise_mask] ** 2

    median_noise_per_bin = np.median(noise_bins_power) if len(noise_bins_power) > 0 else 1e-30
    
    # Scale per-bin noise by the number of bins used for signal integration
    total_noise_power = median_noise_per_bin * n_signal_bins
    snr_dB = 10 * np.log10(signalPower / max(total_noise_power, 1e-30))

    # 5. Prominence calculation with boundary guard
    if (min_freq is not None) and (max_freq is not None):
        band_width = max_freq - min_freq
        pad = band_width * prominence_pad_frac
        prom_min = max(0, min_freq - pad)
        prom_max = max_freq + pad
        keep_prom = (specFreq >= prom_min) & (specFreq <= prom_max)
        specAbs_prom, specFreq_prom = specAbs[keep_prom], specFreq[keep_prom]
    else:
        specAbs_prom, specFreq_prom = specAbs, specFreq

    peakIdx_prom = int(np.argmin(np.abs(specFreq_prom - peakFreq)))     # das funktioniert so leider nicht :(


    # Ensure index is a local maximum before calling peak_prominences
    is_local_max = (
        0 < peakIdx_prom < len(specAbs_prom) - 1
        and specAbs_prom[peakIdx_prom] >= specAbs_prom[peakIdx_prom - 1]
        and specAbs_prom[peakIdx_prom] >= specAbs_prom[peakIdx_prom + 1]
    )

    if is_local_max:
        prominence = float(signal.peak_prominences(specAbs_prom, [peakIdx_prom])[0][0])
    else:
        prominence = float(specAbs_prom[peakIdx_prom] - np.min(specAbs_prom))

    prominence_ratio = prominence / specAbs_prom[peakIdx_prom] if specAbs_prom[peakIdx_prom] > 0 else 0.0

    return {
        "rate": peakFreq,
        "snr_dB": snr_dB,
        "prominence": prominence,
        "prominence_ratio": prominence_ratio
    }
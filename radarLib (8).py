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

def calcFFTbinFromFrequency(radarConfig:radarConfigData, frequency:float) -> int:
    fftBin = frequency / radarConfig.fft_step_Hz
    return int(fftBin)


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

# ---------------------------------------------------------------------------------------------------------
# functions for preprocessed data
# ---------------------------------------------------------------------------------------------------------


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


def notchFilter(data:np.ndarray, freq:float, fs:float, Q : float =5):
    '''
    filter data with a 2nd-order notch filter
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

    # Fundamental peak selection within restricted search band
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

    # Map restricted peak back to full spectrum index
    full_peakIdx = np.flatnonzero(keep)[peakIdx_restricted]

    # Harmonic Masking & Integration over full spectrum
    signal_mask = np.zeros(len(specFreq), dtype=bool)

    for k in range(1, n_harmonics + 1):
        # used full spectrum index for harmonics (new)
        target_idx = full_peakIdx * k
        if target_idx >= len(specFreq):
            break   # abort loop because this and all higher harmonics out of FFT range
        lo = max(0, target_idx - effective_exclude)
        hi = min(len(specFreq), target_idx + effective_exclude + 1)
        signal_mask[lo:hi] = True

    # Total signal power integrated over main lobes
    signalPower = np.sum(specAbs[signal_mask] ** 2)
    n_signal_bins = np.sum(signal_mask)

    # Noise power normalized to signal bandwidth
    noise_mask = keep & (~signal_mask)
    noise_bins_power = specAbs[noise_mask] ** 2

    median_noise_per_bin = np.median(noise_bins_power) if len(noise_bins_power) > 0 else 1e-30
    total_noise_power = median_noise_per_bin * n_signal_bins
    snr_dB = 10 * np.log10(signalPower / max(total_noise_power, 1e-30))

    # Prominence calculation with genuine local peak snapping
    if (min_freq is not None) and (max_freq is not None):
        band_width = max_freq - min_freq
        pad = band_width * prominence_pad_frac
        prom_min = max(0, min_freq - pad)
        prom_max = max_freq + pad
        keep_prom = (specFreq >= prom_min) & (specFreq <= prom_max)
        specAbs_prom, specFreq_prom = specAbs[keep_prom], specFreq[keep_prom]
    else:
        specAbs_prom, specFreq_prom = specAbs, specFreq

    # Find all local maxima in the prominence slice
    local_peaks, _ = signal.find_peaks(specAbs_prom)

    # Target bin index in specFreq_prom closest to selected peak frequency
    target_prom_idx = int(np.argmin(np.abs(specFreq_prom - peakFreq)))

    if len(local_peaks) > 0:
        # Snap to the nearest genuine local peak
        peakIdx_prom = int(local_peaks[np.argmin(np.abs(local_peaks - target_prom_idx))])
        # Compute exact scipy peak prominence
        prominence = float(signal.peak_prominences(specAbs_prom, [peakIdx_prom])[0][0])
    else:
        # no genuine local maximum in the slice (e.g. peak sits right at a boundary) -
        # fall back to a simple height-above-minimum estimate at the closest bin
        peakIdx_prom = target_prom_idx
        prominence = float(specAbs_prom[peakIdx_prom] - np.min(specAbs_prom))

    prominence_ratio = prominence / specAbs_prom[peakIdx_prom] if specAbs_prom[peakIdx_prom] > 0 else 0.0

    return {
        "rate": peakFreq,
        "snr_dB": snr_dB,
        "prominence": prominence,
        "prominence_ratio": prominence_ratio,
        "signal_power": signalPower,       
        "noise_power": total_noise_power,   
    }


#Build a list of the first n harmonics: peakList = [peakIdx, peakIdx*2,peakIdx*3,peakIdx*4]
#Calculate the window len: I would use a window a little bit smaller then the distance between harmonics: winLen = peakIdx-2
#call scipy.peak_prominence with the list and window len : prominences,_,_ = scipy.peak_prominence(spcAbs, peakList, winLen)
#scale each prominence to the height of the corresponding peak : prominence = prominences[0] / specAbs[peakList[0]]
#what does the prominence 

"""
Revised displacement / rate / SNR / prominence logic.

Assumes the following exist in your module (as in your current code):
    normalize(), SPEED_OF_LIGHT, and a radarConfig with start_frequency_Hz / end_frequency_Hz.
"""
import numpy as np
import scipy as sp
from scipy import signal
from scipy.signal import detrend


# ----------------------------------------------------------------------------------------------
# Displacement
# ----------------------------------------------------------------------------------------------
def calcDisplacement(radarConfig, phaseData: np.ndarray, detrend_type: str = 'linear') -> np.ndarray:
    '''
    Displacement (m) of the target relative to its mean position, from the slow-time phase of the
    selected range bin:  d = phi * lambda / (4*pi)   (round trip -> 4*pi).

    * np.unwrap makes the phase continuous (a no-op if it is already unwrapped and steps are < pi).
      Cumulating wrapped frame-to-frame phase differences is mathematically the same thing, so the
      old difference-based function and the "Alt" function collapse into this one.
    * detrend removes the mean (and, for 'linear', slow drift). Use detrend_type='constant' to
      remove the mean only.

    :return: array with the SAME length as phaseData (the old calcDisplacement returned n-1 values)
    '''
    phase = np.unwrap(np.asarray(phaseData, dtype=float))
    phase = detrend(phase, type=detrend_type)
    meanFreq = (radarConfig.start_frequency_Hz + radarConfig.end_frequency_Hz) / 2
    wavelength = SPEED_OF_LIGHT / meanFreq
    return phase * wavelength / (4 * np.pi)


calcDisplacementAlt = calcDisplacement   # backwards-compatible alias


# ----------------------------------------------------------------------------------------------
# Rate, SNR, prominence
# ----------------------------------------------------------------------------------------------
def _logparabolic_offset(mag: np.ndarray, i: int) -> float:
    '''Sub-bin peak offset (in bins, within +-0.5) by parabolic interpolation of the log-magnitude.'''
    if i <= 0 or i >= len(mag) - 1:
        return 0.0
    a, b, c = np.log(np.maximum(mag[i - 1:i + 2], 1e-30))
    denom = a - 2 * b + c
    if denom >= 0:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))


def calcRateSNR4(distData: np.ndarray, framerate_Hz: float, exclude_bins: int = 3, n_harmonics: int = 4,
                 min_freq: float = None, max_freq: float = None, prominence_pad_frac: float = 0.3,
                 zero_pad_factor: float = 2.0, include_harmonics: bool = False,
                 harmonic_min_snr_dB: float = 6.0, min_noise_bins: int = 10) -> dict:
    '''
    Peak picking (with sub-bin interpolation), SNR, prominence and peak dominance.

    snr_dB            noise-corrected SNR of the FUNDAMENTAL lobe (default). Harmonic lobes are only
                      added if include_harmonics=True and the lobe peak is > harmonic_min_snr_dB
                      above the noise floor.
    prominence(_ratio) scipy prominence of the selected peak, 0 if the selected peak is not a genuine
                      local maximum (it sits on the slope of a bigger peak outside the search band).
    peak_to_second_dB selected peak vs. the strongest competing local maximum in the search band
                      (outside fundamental/harmonic lobes). Much more discriminating than
                      prominence_ratio, which is ~1 for any global maximum.
    '''
    result = {"rate": np.nan, "snr_dB": np.nan, "prominence": np.nan, "prominence_ratio": np.nan,
              "peak_to_second_dB": np.nan, "peak_is_local_max": False, "peak_at_band_edge": False,
              "signal_power": np.nan, "noise_power": np.nan}

    n = len(distData)
    # detrend BEFORE windowing: the window turns any leftover mean/drift into low-frequency leakage
    x = normalize(detrend(np.asarray(distData, dtype=float)))
    x = x * signal.get_window('blackman', n)

    nfft = int(round(n * zero_pad_factor))
    specAbs = np.abs(sp.fft.rfft(x, n=nfft))
    specAbs[0] = 0
    if not np.all(np.isfinite(specAbs)) or specAbs.max() <= 0:
        return result
    specFreq = sp.fft.rfftfreq(nfft, 1 / (framerate_Hz * 60))      # cycles per minute
    power = specAbs ** 2
    nbins = len(specFreq)
    df = specFreq[1] - specFreq[0]
    half = int(np.ceil(exclude_bins * zero_pad_factor))           # Blackman main lobe: +-3 un-padded bins

    # --- search band (DC bin is zeroed, so it is never part of the band) ---------------------------
    keep = np.ones(nbins, dtype=bool)
    keep[0] = False
    if min_freq is not None:
        keep &= specFreq >= min_freq
    if max_freq is not None:
        keep &= specFreq <= max_freq
    band = np.flatnonzero(keep)
    if band.size == 0:
        return result

    peak = int(band[np.argmax(specAbs[band])])
    # Interpolate only if the peak is strictly inside the band, so the reported rate can never leave it.
    # On an edge bin the neighbour outside the band lies on a skirt (e.g. DC leakage) and would pull the
    # estimate out of the band by up to half a bin.
    at_edge = bool(peak <= band[0] or peak >= band[-1])
    peak_pos = peak + (0.0 if at_edge else _logparabolic_offset(specAbs, peak))
    result["rate"] = float(peak_pos * df)
    result["peak_at_band_edge"] = at_edge

    # --- lobes: harmonics are centred on k * (fractional) fundamental position ----------------------
    def lobe(center):
        m = np.zeros(nbins, dtype=bool)
        m[max(0, center - half):min(nbins, center + half + 1)] = True
        return m

    fund = lobe(peak)
    harm = []                                                      # (centre, mask)
    for k in range(2, n_harmonics + 1):
        c = int(round(k * peak_pos))
        if c >= nbins:
            break
        harm.append((c, lobe(c)))
    sig_all = fund.copy()
    for _, m in harm:
        sig_all |= m

    # --- noise floor ---------------------------------------------------------------------------------
    noise_mask = keep & ~sig_all
    if noise_mask.sum() < min_noise_bins:      # narrow band: keep harmonic lobes in, the median is robust
        noise_mask = keep & ~fund
    if noise_mask.sum() >= min_noise_bins:
        # power of Gaussian noise is exponentially distributed: median = ln(2) * mean
        noise_per_bin = max(float(np.median(power[noise_mask]) / np.log(2)), 1e-30)

        used = fund.copy()
        if include_harmonics:
            thr = noise_per_bin * 10 ** (harmonic_min_snr_dB / 10)
            for c, m in harm:
                if power[max(0, c - 2):c + 3].max() >= thr:
                    used |= m
        n_used = int(used.sum())
        noise_total = noise_per_bin * n_used
        signal_net = max(float(power[used].sum()) - noise_total, 0.0)   # lobe bins contain noise too
        result["snr_dB"] = float(10 * np.log10(max(signal_net / noise_total, 1e-3)))  # floor: -30 dB
        result["signal_power"] = signal_net
        result["noise_power"] = noise_total

    # --- prominence (on a slice padded around the search band) --------------------------------------
    if (min_freq is not None) and (max_freq is not None):
        pad = (max_freq - min_freq) * prominence_pad_frac
        keep_prom = (specFreq >= max(0, min_freq - pad)) & (specFreq <= max_freq + pad)
    else:
        keep_prom = np.ones(nbins, dtype=bool)
    prom_idx = np.flatnonzero(keep_prom)
    mag = specAbs[prom_idx]
    local, _ = signal.find_peaks(mag)
    t = peak - prom_idx[0]
    is_local = bool(np.any(local == t))
    # no snapping to a *different* peak: that would report the prominence of something other than "rate"
    prominence = float(signal.peak_prominences(mag, [t])[0][0]) if is_local else 0.0
    result["prominence"] = prominence
    result["prominence_ratio"] = prominence / specAbs[peak]
    result["peak_is_local_max"] = is_local

    # --- dominance over the strongest competing peak in the search band ------------------------------
    cand = band[signal.find_peaks(specAbs[band])[0]]
    cand = cand[~sig_all[cand]]
    if cand.size:
        result["peak_to_second_dB"] = float(20 * np.log10(specAbs[peak] / max(specAbs[cand].max(), 1e-30)))

    return result


# stack all windows of one temperature tercile (same window length -> same grid), then:
# mean_spec = np.mean(stack, axis=0); plot 10*np.log10(mean_spec) vs f
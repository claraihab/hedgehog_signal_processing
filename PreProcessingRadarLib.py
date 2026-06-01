'''
First library including directories and functions for preprocessing of raw radar data from BGT60xx

'''

#packages needed 
import numpy as np
import os
import json
from scipy.signal import get_window, butter, filtfilt, detrend, iirnotch, lfilter
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# functions for processing of raw radar-data

DATA_NAME       = "radar.npy"
CONFIG_NAME     = "config.json"
TIMESTAMP_NAME  = "radar_timestamp.csv"
ANGLES_NAME     = "radar_angles.npy"

SPEED_OF_LIGHT = 300e6  # 300.000 km/s = 300.000.000 m/s
TARGET_DISTANCE = 0.4 
MAX_FRAME= 6000



def importRadarData(dataPath : str) -> tuple : 
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
    data = np.load(dataFile, mmap_mode = "r")
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

def calcFFTbinFromFrequency(radarconfig:dict, frequency:float) -> int:
    smplPerChirp = radarconfig['num_samples_per_chirp']
    smplRate = radarconfig['sample_rate_Hz']
    chirpTime = smplPerChirp / smplRate  # length of one chirp in s
    fftStepHz = 1 / chirpTime           # freq-diff between 2 FFT-bins

    fftBin      = frequency / fftStepHz
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
    :param FFTbin:      int|list of int     which bins of the FFT to get
    :param window:      optional, window or name of window-function for FFT of chirps
    :param frameMax:    optional, max number of frames to process
    '''
    TX_ANTENNA = 0
    RX_ANTENNA = 0

    config, data, timestamp = importRadarData(dataPath)
    # create window-array by name
    if window is not None:
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
            freq = calcFrequencyFromDistance(config, d)
            bin =  calcFFTbinFromFrequency(config, freq)
            FFTbin.append(bin)
    else:
        binCnt  = 1
        freq = calcFrequencyFromDistance(config, distance)
        FFTbin  = calcFFTbinFromFrequency(config, freq)
        
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
            tmp = np.array([timestamp[frameNum], phi])
            results[frameNum] = tmp

    return results

# functions for preprocessed data

def saveRadarAngles(radarAngles:np.ndarray, dataPath:str):
    np.save(dataPath + "\\" + ANGLES_NAME, radarAngles)




'''
Second library includes functions for the filters used on the preprocessed data
'''

# all in beats-per-minute
HEART_RATE_HIBERNATION_MIN      = 5   # beats per minute
HEART_RATE_HIBERNATION_MAX      = 20
BREATH_RATE_HIBERNATION_MAX     = 5
BREATH_RATE_HIBERNATION_MIN     = 0.5

HEART_RATE_NORMAL_MIN       = 100
HEART_RATE_NORMAL_MAX       = 360
BREATH_RATE_NORMAL_MIN       = 10
BREATH_RATE_NORMAL_MAX       = 50

# Sliding Window Settings
WINDOW_SEC_NORMAL       = 60
WINDOW_STEP_NORMAL      = 2
WINDOW_SEC_HIBERNATION  = 180
WINDOW_STEP_HIBERNATION = 5


def importRadarAngles(dataPath:str):
    '''
    import preprocessed radar-data and radar-configuration from datapath
    :param dataPath:    where to find files
    :return:            deviceConfig, preprocessed radar-data
    '''
    configFile  = dataPath + "\\" + CONFIG_NAME
    anglesFile  = dataPath + "\\" + ANGLES_NAME
    # read JSON-File with configuration
    with open(configFile) as configJson:
        deviceConfig = json.load(configJson)['device_config'] ['fmcw_single_shape']
    # read radar data
    angles = np.load(anglesFile, mmap_mode = "r")
    return deviceConfig, angles





def apply_notch_filter(data, target_freq, fs, quality_factor=30.0):
    """
    Removes a specific frequency from the signal.
    :param target_freq: The frequency to remove (e.g., 0.33 Hz for 20 BPM)
    :param fs: Sampling frequency (your 227 Hz)
    :param quality_factor: How narrow the notch is (higher = narrower)
    """
    # Design the notch filter
    b_notch, a_notch = iirnotch(target_freq, quality_factor, fs)
    
    # Apply it to the signal
    filtered_data = filtfilt(b_notch, a_notch, data)
    
    return filtered_data


#need to change this according to my path...
def doPreprocess(dataPath: str):
    print('Importing data from :', dataPath)
    #  preprocess_data
    results = preProcessRadarData(dataPath, distance = TARGET_DISTANCE, frameMax=MAX_FRAME)
    print('> shape :', results.shape)
    saveRadarAngles(results, dataPath)


def postProcessRadarData(dataPath: str):
    '''
    postProcess radar data

    bla, bla, bla

    :param data_path: path to data
    :return:    hibernation_data, arousal_data  : results for each window
    '''
    global diagnostic_fft_saved_norm, diagnostic_fft_saved_hiber
    configData, results = importRadarAngles(dataPath)

    # test for format of preprocessing results
    if len(results.shape) > 1:
        results = results[:, 1]

    print(results.shape)
    print(f"Successfully loaded config and {len(results)} precalculated frames.")

    fs = 1 / (configData['frame_repetition_time_s'])  # important

    smplPerMinute = fs * 60  # convert from Hz to samples-per-minute

    phase_unwrapped = np.unwrap(results)
    phi_detrended = detrend(phase_unwrapped)

    # SIGNAL FILTERING
    # Deep Hibernation Heart: 2-15 BPM
    heart_signal_raw = apply_filter(phi_detrended, [HEART_RATE_HIBERNATION_MIN, HEART_RATE_HIBERNATION_MAX],
                                    smplPerMinute, btype='bandpass')
    heart_signal_final = np.zeros(len(heart_signal_raw))  # will be filled while processing windows, additional notch filter applied

    # Deep Hibernation Resp: Lowered ceiling to under 5 BrPM
    resp_signal = apply_filter(phi_detrended, BREATH_RATE_HIBERNATION_MAX, smplPerMinute, btype='low')

    # calculate window size and step in samples
    window_samples, step_samples = int(fs * WINDOW_SEC_HIBERNATION), int(fs * WINDOW_STEP_HIBERNATION)

    hibernation_data = []

    # WINDOW PROCESSING LOOP for hibernation
    fft_window = np.blackman(window_samples)
    for start in range(0, len(heart_signal_raw) - window_samples, step_samples):
        # Calculate the midpoint sample of current window, then convert to seconds from start of file
        time_sec = (start + (window_samples / 2)) / fs

        # Respiration rate Calculation
        brpm, r_fft_freq, r_fft, r_mask = calculate_peak_rate(resp_signal[start: start + window_samples],
                                                              smplPerMinute=smplPerMinute,
                                                              beats_min=BREATH_RATE_HIBERNATION_MIN,
                                                              beats_max=BREATH_RATE_HIBERNATION_MAX,
                                                              window=fft_window)

        # Dynamic Notch for Deep Hibernation
        harmonic_bpm = (brpm * 2) #filter expects it in Hz 
        h_chunk_raw = heart_signal_raw[start: start + window_samples]
        h_chunk_clean = apply_notch_filter(h_chunk_raw, harmonic_bpm, smplPerMinute)
        # save notch-filtered heart-rate-data into heart_signal_final
        heart_signal_final[start: start + window_samples] = h_chunk_clean

        # Hibernation Heart BPM Calculation
        hrpm, h_fft_freq, h_fft, h_mask = calculate_peak_rate(h_chunk_clean,
                                                              smplPerMinute=smplPerMinute,
                                                              beats_min=HEART_RATE_HIBERNATION_MIN,
                                                              beats_max=HEART_RATE_HIBERNATION_MAX,
                                                              window=fft_window)

        hibernation_data.append({
            "Time_Sec": round(time_sec, 2),
            "Heartrate": round(hrpm, 2),
            "Breathrate": round(brpm, 2),
            "Movement_Index": round(0, 4)

        })

        if not diagnostic_fft_saved_hiber:
            save_diagnostic_plot(r_fft, r_fft_freq, r_mask,
                                 h_fft, h_fft_freq, h_mask,
                                 os.path.join(dataPath, "diagnostic_fft_window_hiber.png"))


            diagnostic_fft_saved_hiber = True

    # WINDOW PROCESSING LOOP for arousel
    arousal_data = []
    window_samples_arou, step_samples_arou = int(fs * WINDOW_SEC_NORMAL), int(fs * WINDOW_STEP_NORMAL)


    # Arousal Ranges
    arousal_heart_raw = apply_filter(phi_detrended, [HEART_RATE_NORMAL_MIN, HEART_RATE_NORMAL_MAX], smplPerMinute,
                                     btype='bandpass')  # 100-360 BPM
    arousal_resp_raw = apply_filter(phi_detrended, BREATH_RATE_NORMAL_MAX, smplPerMinute,
                                    btype='lowpass')  # until 50 BrPM
    fft_window_arou = np.blackman(window_samples_arou)
    
    for start in range(0, len(heart_signal_raw) - window_samples_arou, step_samples_arou):
        # Calculate the midpoint sample of current window, then convert to seconds from start of file
        time_sec = (start + (window_samples_arou / 2)) / fs

        # Respiration rate Calculation
        arousal_brpm, ar_fft_freq, ar_fft, ar_mask = calculate_peak_rate(
            arousal_resp_raw[start: start + window_samples_arou],
            smplPerMinute=smplPerMinute, beats_min=BREATH_RATE_NORMAL_MIN, beats_max=BREATH_RATE_NORMAL_MAX,
            window=fft_window_arou)

        # heart rate Calculation
        arousal_bpm, ah_fft_freq, ah_fft, ah_mask = calculate_peak_rate(
            arousal_heart_raw[start: start + window_samples_arou],
            smplPerMinute=smplPerMinute, beats_min=HEART_RATE_NORMAL_MIN, beats_max=HEART_RATE_NORMAL_MAX,
            window=fft_window_arou)

        # Activity / Shivering metric
        movement_intensity = np.std(phi_detrended[
                                    start: start + window_samples_arou])  # compute standard deviation, detecting bigger motions

        arousal_data.append({
            "Time_Sec": time_sec,
            "Heartrate": round(arousal_bpm, 2),
            "Breathrate": round(arousal_brpm, 2),
            "Movement_Index": round(movement_intensity, 4)
        })

        # write plots of last window for diagnostics
        if not diagnostic_fft_saved_norm:

            save_diagnostic_plot(ar_fft, ar_fft_freq, ar_mask,
                                 ah_fft, ah_fft_freq, ah_mask,
                                 os.path.join(dataPath, "diagnostic_fft_window_arousel.png"))

            diagnostic_fft_saved_norm = True


    return hibernation_data, arousal_data, configData, resp_signal, heart_signal_final, arousal_resp_raw, arousal_heart_raw



def apply_filter(data, cutoff, fs, btype='bandpass', order=4):
    nyq = 0.5 * fs
    if btype == 'bandpass':
        low, high = cutoff[0] / nyq, cutoff[1] / nyq
        b, a = butter(order, [low, high], btype=btype)
    else:
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype=btype)
    return filtfilt(b, a, data)



def calculate_peak_rate(signal, start_idx, win_samples, smplPerMinute, f_min, f_max, n_fft=32768):
    """Slices a signal chunk, applies a Blackman window, and finds the peak frequency in BPM."""
    chunk = signal[start_idx : start_idx + win_samples]
    windowed = chunk * np.blackman(win_samples)
    
    yf = np.abs(fft(windowed, n=n_fft))
    xf = fftfreq(n_fft, 1/fs)
    mask = (xf >= f_min) & (xf <= f_max)
    
    rate = xf[mask][np.argmax(yf[mask])] * 60
    return rate, xf, yf, mask

##this is wronggggggg


def save_diagnostic_plot(fft_result_breath, fft_freq_breath, breath_mask,
                         fft_result_heart, fft_freq_heart, heart_mask,
                         file_name):
    '''
    create a plot of fft-result for heart-rate and breath-rate and save as PNG
    :param fft_result_breath:
    :param fft_freq_breath:
    :param breath_mask:
    :param fft_result_heart:
    :param fft_freq_heart:
    :param heart_mask:
    :param file_name:
    :return:    None
    '''
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    x_axis = fft_freq_breath[breath_mask]
    y_axis = fft_result_breath[breath_mask]
    ax1.plot(x_axis , y_axis, color='blue')
    ax1.set_title("Respiration Spectrum (Magnitude vs. Frequency)")
    ax1.set_xlabel("Breaths Per Minute (BrPM)")
    
    x_axis = fft_freq_heart[heart_mask]
    y_axis = fft_result_heart[heart_mask]
    ax2.plot(x_axis, y_axis, color='red')
    ax2.set_title("Heart Spectrum (Magnitude vs. Frequency)")
    ax2.set_xlabel("Beats Per Minute (BPM)")
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=150)
    plt.close(fig)
    print(">>> Saved diagnostic spectrum profile to :",file_name)






def plotMovement(resp_signal, heart_signal, radar_wavelength, time_axis, data_path, name):
    resp_mm = (resp_signal * radar_wavelength * 1000 /2) / (2 * np.pi)
    heart_mm = (heart_signal * radar_wavelength * 1000 /2)  / (2 * np.pi)

    # Plot 1: Deep Hibernation Baseline Vitals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    ax1.plot(time_axis, resp_mm, color='blue', linewidth=0.5)
    ax1.set_title(f"Respiration Profile")

    ax2.plot(time_axis, heart_mm, color='red', linewidth=0.3)
    ax2.set_title("Heart Profile")
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"plot_{name}.png"), dpi=150)
    plt.close(fig)

def plotRates(rate_data, data_path, name):
    # get series from data
    v_time = [v['Time_Sec'] for v in rate_data]
    v_h_bpm = [v['Heartrate'] for v in rate_data]
    v_b_bpm = [v['Breathrate'] for v in rate_data]
    # v_move = [v['Movement_Index'] for v in arousal_data]

    fig, (ax_ah, ax_ab) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    ax_ah.plot(v_time, v_h_bpm, color='purple', marker='o', markersize=2, linestyle='-')
    ax_ah.set_title(f"Heart-Rate in bpm")
    ax_ab.plot(v_time, v_b_bpm, color='orange', linewidth=0.8)
    ax_ab.set_title("Breath-Rate in bpm")

    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"rate_plot_{name}.png"), dpi=150)
    plt.close(fig)


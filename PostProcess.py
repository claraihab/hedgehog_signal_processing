'''
Second loop for filtering data
Starts from the angles.npy
'''

import sys
import os
import numpy as np
from scipy.signal import detrend

# sys.path.append(r"C:\Users\Clara Guindy\Documents\GitHub\hedgehog_signal_processing")
from PreProcessingRadarLib import *

data_path = "D:\\Projekte\\Igel-Box\\Radar\\mit_Igel\\recording_2026_02_26_11_42_11"


MAX_FRAME = 60000

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


diagnostic_fft_saved_norm = False
diagnostic_fft_saved_hiber = False

def postProcessRadarData(data_path):
    '''
    postProcess radar data

    bla, bla, bla

    :param data_path: path to data
    :return:    hibernation_data, arousal_data  : results for each window
    '''
    global diagnostic_fft_saved_norm, diagnostic_fft_saved_hiber
    configData, results = importRadarAngles(data_path)

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
        harmonic_bpm = (brpm * 2)
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
                                 os.path.join(data_path, "diagnostic_fft_window_hiber.png"))


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
                                 os.path.join(data_path, "diagnostic_fft_window_arousel.png"))

            diagnostic_fft_saved_norm = True


    return hibernation_data, arousal_data, configData, resp_signal, heart_signal_final, arousal_resp_raw, arousal_heart_raw


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


#
SPEED_OF_LIGHT = 3e8  # 300.000.000 m/s
TARGET_DISTANCE = 0.4   # distance to target/animal in m
TARGET_RT_TIME  = (TARGET_DISTANCE*2) / SPEED_OF_LIGHT  # time for one roundtrip

def doPreprocess():
    print('Importing data from :', data_path)
    #  preprocess_data
    results = preProcessRadarData(data_path, distance = TARGET_DISTANCE, frameMax=MAX_FRAME)
    print('> shape :', results.shape)
    saveRadarAngles(results, data_path)

# doPreprocess()
hibernation_data, arousal_data, config_data, resp_signal, heart_signal_final, arousel_resp_signal, arousel_heart_signal = postProcessRadarData(data_path)

mean_radar_frequency = (config_data["start_frequency_Hz"] + config_data["end_frequency_Hz"]) / 2
radar_wavelength = SPEED_OF_LIGHT / mean_radar_frequency
time_axis = np.linspace(0, len(resp_signal) * config_data["frame_repetition_time_s"], len(resp_signal))

# plotMovement(arousel_resp_signal, arousel_heart_signal, radar_wavelength, time_axis, data_path, "arousel")
# plotMovement(resp_signal, heart_signal_final, radar_wavelength, time_axis, data_path, "hiber")

plotRates(hibernation_data, data_path, 'hiber')
plotRates(arousal_data, data_path, 'arousal')
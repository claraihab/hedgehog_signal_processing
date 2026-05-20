'''
Second loop for filtering data
Starts from the angles.npy 
'''

SPEED_OF_LIGHT = 3e8
TARGET_DISTANCE = 0.4
TARGET_RT_TIME = (TARGET_DISTANCE * 2) / SPEED_OF_LIGHT

import sys
sys.path.append(r"C:\Users\Clara Guindy\Documents\GitHub\hedgehog_signal_processing")
import sys
import os
import numpy as np
from scipy.signal import detrend

# sys.path.append(r"C:\Users\Clara Guindy\Documents\GitHub\hedgehog_signal_processing")
from PreProcessingRadarLib import *


ROOT_DIR = r"D:\igel\Radar-Test"
HARDWARE_SUBFOLDER = "RadarIfxBGT60TR13C_00"

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

recording_folders = [f for f in os.listdir(ROOT_DIR) if f.startswith("recording_")]

diagnostic_fft_saved = False

def postProcessRadarData(data_path):
    '''
    postProcess radar data

    bla, bla, bla

    :param data_path: path to data
    :return:    hibernation_data, arousal_data  : results for each window
    '''
    global diagnostic_fft_saved
    configData, results = importRadarAngles(data_path)

    # test for format of preprocessing results
    if len(results.shape) >1:
        results = results[:,1]

    print(results.shape)
    print(f"Successfully loaded config and {len(results)} precalculated frames.")

    try:
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
                "Time_Sec": round((start + window_samples/2)/fs, 2),
                "Hibernation_BPM": round(bpm, 2),
                "Hibernation_BrPM": round(brpm, 2),
            })

        for start in range(0, len(heart_signal_raw) - window_samples_arou, step_samples_arou):
            
            time_sec = (start + (window_samples_arou / 2)) / fs #calculate midpoint sample

            # Arousal Analysis 
            ah_chunk = arousal_heart_raw[start : start + window_samples_arou] * np.blackman(window_samples_arou)
            ah_yf = np.abs(fft(ah_chunk, n=n_fft))
            ah_mask = (h_xf >= 1.6) & (h_xf <= 6.0)
            arousal_bpm = h_xf[ah_mask][np.argmax(ah_yf[ah_mask])] * 60

            ar_chunk = arousal_resp_raw[start : start + window_samples] * np.blackman(window_samples_arou)
            ar_yf = np.abs(fft(ar_chunk, n=n_fft))
            ar_mask = (r_xf >= 0.5) & (r_xf <= 2.0)
            arousal_brpm = r_xf[ar_mask][np.argmax(ar_yf[ar_mask])] * 60

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
                "Arousal_BPM": round(arousal_bpm, 2),
                "Arousal_BrPM": round(arousal_brpm, 2),
                "Movement_Index": round(movement_intensity, 4)
            })

        # write plots of last window for diagnostics
        if not diagnostic_fft_saved:
            save_diagnostic_plot(r_fft, r_fft_freq, r_mask, h_fft, h_fft_freq, h_mask,
                                 os.path.join(ROOT_DIR, "diagnostic_fft_window.png"))

            save_diagnostic_plot(ar_fft, ar_fft_freq, ar_mask, ah_fft, ah_fft_freq, ah_mask,
                                 os.path.join(ROOT_DIR, "diagnostic_fft_window_arousel.png"))

            diagnostic_fft_saved = True

    except Exception as e:
        print(f"Error in {data_path}: {e}")

    return hibernation_data, arousal_data, configData, resp_signal, heart_signal_final, arousal_heart_raw, arousal_resp_raw
# ---------------------------------------------------------------------------------------

for rec_folder in recording_folders:
    data_path = os.path.join(ROOT_DIR, rec_folder, HARDWARE_SUBFOLDER)
    if not os.path.exists(os.path.join(data_path, "radar_angles.npy")):
        continue

    hibernation_data, arousal_data, config_data, resp_signal, heart_signal_final = postProcessRadarData(data_path)
    mean_radar_frequency = (config_data["start_frequency_Hz"] + config_data["stop_frequency_Hz"]) / 2
    radar_wavelength = SPEED_OF_LIGHT / mean_radar_frequency

    # PLOTTING & EXPORT
    time_axis = np.linspace(0, len(resp_signal)*config_data["frame_repetition_time_s"], len(resp_signal))

    resp_mm = (resp_signal * radar_wavelength * 1000 /2) / (2 * np.pi)
    heart_mm = (heart_signal_final * radar_wavelength * 1000 /2)  / (2 * np.pi)

    # Plot 1: Deep Hibernation Baseline Vitals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    ax1.plot(time_axis, resp_mm, color='blue', linewidth=0.5)
    ax1.set_title(f"Respiration Profile : {rec_folder}")

    ax2.plot(time_axis, heart_mm, color='red', linewidth=0.3)
    ax2.set_title("Heart Profile (Cleaned via Dynamic Notch)")
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"plot_{rec_folder}.png"), dpi=150)
    plt.close(fig)

    # Plot 2: Arousal Profiles
    fig_arousal, (ax_ah, ax_am) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    v_time = [v['Time_Sec'] for v in arousal_data]
    v_a_bpm = [v['Arousal_BPM'] for v in arousal_data]
    v_move = [v['Movement_Index'] for v in arousal_data]

    ax_ah.plot(v_time, v_a_bpm, color='purple', marker='o', markersize=2, linestyle='-')
    ax_ah.set_title(f"Arousal Tracking - High Frequency Heart Spectrum (100-360 BPM)")
    ax_am.plot(v_time, v_move, color='orange', linewidth=0.8)
    ax_am.set_title("Movement / Shivering Index (Phase Variance)")

    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"arousal_plot_{rec_folder}.png"), dpi=150)
    plt.close(fig_arousal)

    # Save CSV Data
    df_final = pd.DataFrame(arousal_data)
    df_final.to_csv(os.path.join(data_path, f"vitals_{rec_folder}.csv"), index=False)
    print(f"Done! Saved Baseline Plot, Arousal Plot, and CSV for {rec_folder}")


print("\n--- BATCH COMPLETE ---")


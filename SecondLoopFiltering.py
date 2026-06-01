'''
Second loop for filtering data
Starts from the angles.npy 
'''

SPEED_OF_LIGHT = 3e8
TARGET_DISTANCE = 0.4
TARGET_RT_TIME = (TARGET_DISTANCE * 2) / SPEED_OF_LIGHT
MAX_FRAME = 60000

import sys
import os
sys.path.append(r"C:\Users\Clara Guindy\Documents\GitHub\hedgehog_signal_processing")
from PreProcessingRadarLib import *

import numpy as np
import pandas as pd

ROOT_DIR = r"D:\igel\Radar-Test"
HARDWARE_SUBFOLDER = "RadarIfxBGT60TR13C_00"
HARDWARE_SUBFOLDER2 = "RadarIfxBGT60TR13C_01"

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


diagnostic_fft_saved_norm = False
diagnostic_fft_saved_hiber = False



for rec_folder in recording_folders:
    data_path = os.path.join(ROOT_DIR, rec_folder, HARDWARE_SUBFOLDER)
    if not os.path.exists(os.path.join(data_path, "radar_angles.npy")):
        continue
    print(f"\n>>> Processing: {rec_folder}")
    
    try:
        results = preProcessRadarData(data_path, distance = TARGET_DISTANCE, frameMax=MAX_FRAME)
        
        hibernation_data, arousal_data, config_data, resp_signal, heart_signal_final, arousal_resp_signal, arousal_heart_signal, r_fft, r_fft_freq, r_mask, h_fft, h_fft_freq, h_mask, ar_fft, ar_fft_freq, ar_mask, ah_fft, ah_fft_freq, ah_mask = postProcessRadarData(data_path) 


        # write plots of last window for diagnostics
        if not diagnostic_fft_saved:
            save_diagnostic_plot(r_fft, r_fft_freq, r_mask, h_fft, h_fft_freq, h_mask, os.path.join(ROOT_DIR, "diagnostic_fft_window.png"))
            save_diagnostic_plot(ar_fft, ar_fft_freq, ar_mask, ah_fft, ah_fft_freq, ah_mask, os.path.join(ROOT_DIR, "diagnostic_fft_window_arousal.png"))

        diagnostic_fft_saved = True

        mean_radar_frequency = (config_data["start_frequency_Hz"] + config_data["end_frequency_Hz"]) / 2
        radar_wavelength = SPEED_OF_LIGHT / mean_radar_frequency
        time_axis = np.linspace(0, len(resp_signal) * config_data["frame_repetition_time_s"], len(resp_signal))

        # plotMovement(arousal_resp_signal, arousal_heart_signal, radar_wavelength, time_axis, data_path, "arousal")
        # plotMovement(resp_signal, heart_signal_final, radar_wavelength, time_axis, data_path, "hiber")

        plotRates(hibernation_data, data_path, 'hiber')
        plotRates(arousal_data, data_path, 'arousal')

        # Save CSV Data
        hib_final = pd.DataFrame(hibernation_data)
        arou_final = pd.DataFrame(arousal_data)
        hib_final.to_csv(os.path.join(data_path, f"hibernation_vitals_{rec_folder}.csv"), index=False)
        arou_final.to_csv(os.path.join(data_path, f"arousal_vitals_{rec_folder}.csv"), index=False)
        print(f"Done! Saved PLOTS and CSVs for {rec_folder}")

    except Exception as e:
        print(f"Error in {rec_folder}: {e}")

print("\n--- BATCH COMPLETE ---")


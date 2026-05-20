'''
Loop for radar signal processing - Hibernating Hedgehog Edition
RadarIfxBGT60TR13C_00
Includes automated diagnostic plotting, hibernation, and arousal tracking.
'''

import sys
sys.path.append(r"C:\Users\Clara Guindy\Documents\GitHub\hedgehog_signal_processing")
from PreProcessingRadarLib import *


# CONFIGURATION 

ROOT_DIR = r"D:\igel\Radar-Test"
HARDWARE_SUBFOLDER = "RadarIfxBGT60TR13C_00"
MAX_FRAME = 20000 

# calculate roundtrip time in s for radar 
SPEED_OF_LIGHT = 3e8  # 300.000.000 m/s
TARGET_DISTANCE = 0.4   # distance to target/animal in m
TARGET_RT_TIME  = (TARGET_DISTANCE*2) / SPEED_OF_LIGHT  # time for one roundtrip


recording_folders = [f for f in os.listdir(ROOT_DIR) if f.startswith("recording_")]
print(f"Found {len(recording_folders)} recordings. Starting batch process...")

valid_recordings_count = 0

for rec_folder in recording_folders:
    data_path = os.path.join(ROOT_DIR, rec_folder, HARDWARE_SUBFOLDER)
    if not os.path.exists(os.path.join(data_path, "radar.npy")):
        continue

    print(f"\n>>> Processing: {rec_folder}")
    
    try:
        print("Import Data from: ", data_path)
        configData,  radarData, timestamp = importRadarData(data_path)

        MIN_FRAMES = 7000  # Minimum frames required (~10 minutes)
        total_frames = radarData.shape[0]

        if total_frames < MIN_FRAMES:
            print(f"Skipping {rec_folder}: Only has {total_frames} frames (under the 7000 threshold).")
            continue  # Instantly skips to next file

        valid_recordings_count += 1

        fs = 1 / (configData['frame_repetition_time_s']) #important
        
        freqStart       = configData['start_frequency_Hz']
        freqStop        = configData['end_frequency_Hz']
        smplPerChirp    = configData['num_samples_per_chirp']
        smplRate        = configData['sample_rate_Hz']
        chirpRepTime    = configData['chirp_repetition_time_s']
        chirpPerFrame  = configData['num_chirps_per_frame']

        # calculate more values from config
        freqCenter = (freqStart + freqStop)/2  #where is this used?
        wavelength = SPEED_OF_LIGHT / configData['start_frequency_Hz'] 
        targetFreq = calcFrequencyFromDistance(configData, TARGET_DISTANCE)
        
        chirpTime       = smplPerChirp / smplRate   # length of one chirp in s
        freqRange       = freqStop - freqStart
        freqChgSpd      = freqRange / chirpTime     # change-rate of frequency in Hz/s
        fftStepHz       = 1 / chirpTime             # step size of FFT result in Hz, the freq of bin[n] is fft_step_size * n
        fftStepM        = (SPEED_OF_LIGHT / smplRate) * smplPerChirp

        freqChgTarget = TARGET_RT_TIME * freqChgSpd
        fftTargetBin  = freqChgTarget / fftStepHz
        ffTargetBinIdx  = int(fftTargetBin)         # what FFT-bin has wanted data

        print("Data-Shape: ", radarData.shape)

        #starting preprocessing
        frame_count = min(radarData.shape[0], MAX_FRAME)
        results = np.zeros(frame_count)
        
        # Phase Extraction
        for f in range(frame_count):
            frameRaw = radarData[f,0,0].copy()
            frameFFT = processOneFrame(frameRaw, mean=True, window=None)
            frameRangeBin = frameFFT[ffTargetBinIdx] 
            results[f] = np.angle(frameRangeBin)
        
        saveRadarAngles(results, data_path)

    except Exception as e:
        print(f"Error in {rec_folder}: {e}")

print("\n--- BATCH COMPLETE ---")
print(f"Total number of recordings that passed the threshold: {valid_recordings_count}")

#45 changes

#this works

#process both
#
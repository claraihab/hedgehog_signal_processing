'''
    radarDataNew - new version of processing radar-data
'''

import numpy as np
import matplotlib.pyplot as plt

from RadarLib import *

DATA_PATH = "D:\\Projekte\\Igel-Box\\Radar\\mit_hund\\RadarIfxBGT60TR13C_00"

# calculate roundtrip time in s for radar
SPEED_OF_LIGHT = 300e6  # 300.000 km/s = 300.000.000 m/s
TARGET_DISTANCE = 0.4   # distance to target/animal in m
TARGET_RT_TIME  = (TARGET_DISTANCE*2) / SPEED_OF_LIGHT  # time for one roundtrip

print("Import Data from: ", DATA_PATH)
configData, radarData = importRadarData(DATA_PATH)

freqStart       = configData['start_frequency_Hz']
freqStop        = configData['end_frequency_Hz']
smplPerChirp    = configData['num_samples_per_chirp']
smplRate        = configData['sample_rate_Hz']
chirpRepTime    = configData['chirp_repetition_time_s']
chirpPerFramne  = configData['num_chirps_per_frame']

# calculate more values from config
chirpTime       = smplPerChirp / smplRate   # length of one chirp in s
freqRange       = freqStop - freqStart
freqChgSpd      = freqRange / chirpTime     # change-rate of frequency in Hz/s
fftStepHz       = 1 / chirpTime             # step size of FFT result in Hz, the freq of bin[n] is fft_step_size * n
fftStepM        = (SPEED_OF_LIGHT / smplRate) * smplPerChirp

freqChgTarget = TARGET_RT_TIME * freqChgSpd
fftTargetBin  = freqChgTarget / fftStepHz
ffTargetBinIdx  = int(fftTargetBin)         # what FFT-bin has wanted data

print("Data-Shape: ", radarData.shape)

frame_count = radarData.shape[0]                 # number of frames in data

MAX_FRAME = 1000

if frame_count >MAX_FRAME:
    frame_count  = MAX_FRAME

results = np.zeros(frame_count)

# process each frame
for f in range(frame_count):
    if (f % 100) ==0: print('Frame :', f)

    frameRaw = radarData[f,0,0].copy()      # select frame i, txAnt 0, rxAnt 0
    frameFFT = processOneFrame( frameRaw, mean = True, window = None)  # should use window to get better FFT results

    # select the bin for distance
    frameRangeBin   = frameFFT[ffTargetBinIdx]

    # calculate phi from complex value in FFT-bin
    phi = np.angle(frameRangeBin)

    # save calculated phi in result-array
    results[f] = phi

# Plot intermediate raw phase values
plt.figure()
plt.plot(results)
plt.show()

# write result-array to file
# np.save bla bla bla

# calculate changes of phi: phi[i]-phi[i-1]

# filter data to separate breath-rate, heart-rate and noise
# for breath-rate use a lowpass filter
# for heart-rate use a bandpass-filter
# possible use an additional notch filter at frequencies that are multiple of breath-rate

# 2 possible methodes:
# A) use filter, then FFT to get frequency
# B) FFT for full data, select parts of spectrum for breath- & heart-rate





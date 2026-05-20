'''
    radarPreprocess - pre-Process raw radar-data and save results of preprocessing in files
'''

import os
import datetime

from RadarLib import *

ROOT_DIR = r"D:\igel\Radar-Test"
HARDWARE_SUBFOLDER_1 = "RadarIfxBGT60TR13C_00"      # first sensor
HARDWARE_SUBFOLDER_2 = "RadarIfxBGT60TR13C_01"      # second sensor
ANIMAL_DISTANCE   = 0.4     # distance between sensor and animal

recording_folders = [f for f in os.listdir(ROOT_DIR) if f.startswith("recording_")]
print(f"Found {len(recording_folders)} recordings. Starting batch process...")

subfolders = [HARDWARE_SUBFOLDER_1,HARDWARE_SUBFOLDER_2]

for rec_folder in recording_folders:
    for subfolder in subfolders:
        rec_folder = os.path.join(ROOT_DIR, rec_folder, subfolder)
        time_string = rec_folder.replace("recording_", "")
        start_anchor = datetime.strptime(time_string, "%Y_%m_%d_%H_%M_%S")

        if not os.path.exists(os.path.join(rec_folder, "radar.npy")) or not os.path.exists(rec_folder):
            continue

        print(f"\n>>> Processing: {rec_folder}")
        try:
            radarAngels = preProcessRadarData(rec_folder, ANIMAL_DISTANCE)
            saveRadarAngels(radarAngels, rec_folder)
        except Exception as e:
            print(f"Error in processing {rec_folder}: {e}")

        
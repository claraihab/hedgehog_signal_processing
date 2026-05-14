'''
Loop for radar signal processing
RadarIfxBGT60TR13C_00
Includes fft, filtering for breathing and heart rate, saving plots as csv
added timestamps
'''
import matplotlib
matplotlib.use('Agg') # Stop windows from popping up
from HedgehogLibrary import *


# CONFIGURATION 
ROOT_DIR = r"D:\igel\Radar-Test"
HARDWARE_SUBFOLDER = "RadarIfxBGT60TR13C_00"
MAX_FRAME = 7000 
if frame_count >MAX_FRAME:
    frame_count  = MAX_FRAME



recording_folders = [f for f in os.listdir(ROOT_DIR) if f.startswith("recording_")]
print(f"Found {len(recording_folders)} recordings. Starting batch process...")


for rec_folder in recording_folders:

    combined_vitals = []
    data_path = os.path.join(ROOT_DIR, rec_folder, HARDWARE_SUBFOLDER)
    time_string = rec_folder.replace("recording_", "")
    start_anchor = datetime.strptime(time_string, "%Y_%m_%d_%H_%M_%S")


    if not os.path.exists(os.path.join(data_path, "radar.npy")) or not os.path.exists(ts_path):
        continue

    print(f"\n>>> Processing: {rec_folder}")

        
    try:
        configData, timeStamps, radarData = importRadarData(data_path)
        radar_start_val = timeStamps[0]
        fs = 1 / (configData['frame_repetition_time_s'])
        wavelength = 3e8 / configData['start_frequency_Hz'] # defined wavelength 

        frame_count = min(radarData.shape[0], MAX_FRAME)
        results = np.zeros(frame_count)

        # Phase Extraction
        for f in range(frame_count):
            frameRaw = radarData[f,0,0].copy()
            frameFFT = processOneFrame(frameRaw, mean=True, window=None)
            frameRangeBin = frameFFT[ffTargetBinIdx] 
            results[f] = np.angle(frameRangeBin)

        phase_unwrapped = np.unwrap(results)
        # phase_diff = np.diff(phase_unwrapped)
        phi_detrended = detrend(phase_unwrapped)

        # 1. PRE-FILTERING 
        low_h, high_h = 0.16 / (fs / 2), 0.66 / (fs / 2)
        b_h, a_h = butter(2, [low_h, high_h], btype='bandpass')
        heart_signal_raw = filtfilt(b_h, a_h, phi_detrended)

        low_r = 0.16 / (fs / 2)
        b_r, a_r = butter(2, low_r, btype='low')
        resp_signal = filtfilt(b_r, a_r, phi_detrended)

        # SLIDING WINDOW WITH DYNAMIC NOTCH
        window_sec, step_sec, n_fft = 10, 1, 16384 
        window_samples = int(fs * window_sec)
        step_samples = int(fs * step_sec)
        
        #combined_vitals = []

        # We'll store cleaned signals for plotting
        heart_signal_final = np.zeros_like(heart_signal_raw)

        for start in range(0, len(heart_signal_raw) - window_samples, step_samples):
            # Resp First 
            r_chunk = resp_signal[start : start + window_samples] * np.blackman(window_samples)
            r_yf = np.abs(fft(r_chunk, n=n_fft))
            r_xf = fftfreq(n_fft, 1/fs)
            r_mask = (r_xf >= 0.016) & (r_xf <= 0.16)
            brpm = r_xf[r_mask][np.argmax(r_yf[r_mask])] * 60

            # Get the timestamp for the middle of the current window
            mid_idx = int(start + window_samples / 2)
            raw_val = timeStamps[mid_idx]
        
            # Calculate the real-world time for THIS window
            radar_start_val = timeStamps[0]
            elapsed_mu = raw_val - radar_start_val
            actual_dt = start_anchor + timedelta(microseconds=elapsed_mu)
        

            #  Dynamic Notch for Heart 
            harmonic_hz = (brpm * 2) / 60
            h_chunk_raw = heart_signal_raw[start : start + window_samples]
            h_chunk_clean = apply_notch_filter(h_chunk_raw, harmonic_hz, fs)
            
            # Save cleaned chunk to the full array for the final plot
            heart_signal_final[start : start + window_samples] = h_chunk_clean

            # Heart BPM 
            h_chunk_windowed = h_chunk_clean * np.blackman(window_samples)
            h_yf = np.abs(fft(h_chunk_windowed, n=n_fft))
            h_xf = fftfreq(n_fft, 1/fs)
            h_mask = (h_xf >= 0.16) & (h_xf <= 0.66)
            bpm = h_xf[h_mask][np.argmax(h_yf[h_mask])] * 60

            combined_vitals.append({
                "Timestamp": actual_dt.strftime("%Y-%m-%d %H:%M:%S"),  
                "Time_Sec": round((start + window_samples/2)/fs, 2),
                "Heart_Rate_BPM": round(bpm, 2),
                "Resp_Rate_BrPM": round(brpm, 2)
            })
        

        # PLOTTING 
        time_axis = np.linspace(0, len(results)/fs, len(results))
        resp_mm = (resp_signal * wavelength * 1000) / (4 * np.pi)
        heart_mm = (heart_signal_final * wavelength * 1000) / (4 * np.pi)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        ax1.plot(time_axis, resp_mm, color='blue', linewidth=0.5)
        ax1.set_title(f"Respiration: {rec_folder}")
        ax2.plot(time_axis, heart_mm, color='red', linewidth=0.3)
        ax2.set_title("Heart Rate (Cleaned via Dynamic Notch)")
        
        plt.tight_layout()
        plot_filename = os.path.join(data_path, f"plot_{rec_folder}.png")
        plt.savefig(plot_filename, dpi=150) # Lower DPI to save time on huge files
        plt.close(fig) # Now this will work correctly with 'Agg'

        # SAVE CSV

        df_final = pd.DataFrame(combined_vitals)
        output_name = os.path.join(data_path, f"vitals_{rec_folder}.csv")
        df_final.to_csv(output_name, index=False) 
        #df_final.to_csv(os.path.join(data_path, f"vitals_{rec_folder}.csv"), index=False)

        print(f"Done! Saved CSV and Plot for {rec_folder}")


    except Exception as e:
        print(f"Error in {rec_folder}: {e}")

print("\n--- BATCH COMPLETE ---")


# Add this after loop to smooth the "staircase" effect
df_final['Heart_Rate_Smooth'] = df_final['Heart_Rate_BPM'].rolling(window=5, center=True).mean()


##ideas
## adaptive pass filter  # change for arousals, to 1.5?
# remove recordings below 10 minutes
## try vmd logic..?
#discard the first 2–5 seconds of the filtered signal
# calculare snr




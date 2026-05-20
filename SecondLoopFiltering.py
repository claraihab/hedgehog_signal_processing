'''
Second loop for filtering data
Starts from the angles.npy 
'''


sys.path.append(r"C:\Users\Clara Guindy\Documents\GitHub\hedgehog_signal_processing")
from PreProcessingRadarLib import *


ROOT_DIR = r"D:\igel\Radar-Test"
HARDWARE_SUBFOLDER = "RadarIfxBGT60TR13C_00"
MAX_FRAME = 20000 

# all in beats-per-minute
HEART_RATE_HIBERNATION_MIN  =   5   # beats per minute
HEART_RATE_HIBERNATION_MAX  = 20
BREATH_RATE_HIBERNATION_MAX     = 5

HEART_RATE_NORMAL_MIN       = 100
HEART_RATE_NORMAL_MAX       = 360
BREATH_RATE_NORMAL_MAX       = 50

recording_folders = [f for f in os.listdir(ROOT_DIR) if f.startswith("recording_")]


for rec_folder in recording_folders:
    data_path = os.path.join(ROOT_DIR, rec_folder, HARDWARE_SUBFOLDER)
    if not os.path.exists(os.path.join(data_path, "radar_angles.npy")):
        continue

    configData, results = importRadarAngles(data_path)

    print(results.shape)
    print(f"Successfully loaded config and {len(results)} precalculated frames.")
    
    
    try: 
        fs = 1 / (configData['frame_repetition_time_s']) #important

        smplPerMinute = fs * 60 # convert from Hz to samples-per-minute


        phase_unwrapped = np.unwrap(results)
        phi_detrended = detrend(phase_unwrapped)

        # SIGNAL FILTERING 
        # Deep Hibernation Heart: 2-15 BPM 
        heart_signal_raw = apply_filter(phi_detrended, [HEART_RATE_HIBERNATION_MIN, HEART_RATE_HIBERNATION_MAX], smplPerMinute, btype='bandpass')
        # Deep Hibernation Resp: Lowered ceiling to under 5 BrPM 
        resp_signal = apply_filter(phi_detrended, BREATH_RATE_HIBERNATION_MAX, smplPerMinute, btype='low')

        # Arousal Ranges 
        arousal_heart_raw = apply_filter(phi_detrended, [HEART_RATE_NORMAL_MIN, HEART_RATE_NORMAL_MAX], smplPerMinute, btype='bandpass') # 100-360 BPM
        arousal_resp_raw = apply_filter(phi_detrended, BREATH_RATE_NORMAL_MAX , smplPerMinute, btype='lowpass')  # until 50 BrPM

        # Sliding Window Settings
        window_sec, step_sec = 180, 5
        window_arou, step_arou = 60, 2
        window_samples, step_samples = int(fs * window_sec), int(fs * step_sec)
        window_samples_arou, step_samples_arou = int(fs * window_arou), int(fs *step_arou)

        fft_window = np.blackman(window_samples)

        hibernation_data = []
        arousal_data = []

    
        heart_signal_final = np.zeros_like(heart_signal_raw)

        # WINDOW PROCESSING LOOP for hibernation
        fft_window = np.blackman(window_samples)

        for start in range(0, len(heart_signal_raw) - window_samples, step_samples):
            # Calculate the midpoint sample of current window, then convert to seconds from start of file
            time_sec = (start + (window_samples / 2) ) / fs
            time_min = (start + (window_samples / 2) ) / smplPerMinute

            brpm, r_fft_freq, r_fft, r_mask =  calculate_peak_rate(resp_signal[start : start + window_samples],
                                smplPerMinute=smplPerMinute, beats_min=0.5, beats_max=BREATH_RATE_HIBERNATION_MAX, window=fft_window)

            """
            "# Respiration Frequency Calculation
            r_chunk = resp_signal[start : start + window_samples] * fft_window
            r_fft = np.abs(fft(r_chunk))
            r_fft_freq= fftfreq(len(r_chunk), 1/smplPerMinute)
            # Mask updated for stricter breathing parameters (0.5 to 5 BrPM)
            r_mask = (r_fft_freq >= 0.5) & (r_fft_freq <= BREATH_RATE_HIBERNATION_MAX)
            brpm = r_fft_freq[r_mask][np.argmax(r_fft[r_mask])]
            """

            # Dynamic Notch for Deep Hibernation
            harmonic_bpm = (brpm * 2)
            h_chunk_raw = heart_signal_raw[start : start + window_samples]
            h_chunk_clean = apply_notch_filter(h_chunk_raw, harmonic_bpm, smplPerMinute)
            heart_signal_final[start : start + window_samples] = h_chunk_clean

            hrpm, h_fft_freq, h_fft, h_mask =  calculate_peak_rate(h_chunk_clean,
                                smplPerMinute=smplPerMinute, beats_min=HEART_RATE_HIBERNATION_MIN, beats_max=HEART_RATE_HIBERNATION_MAX, window=fft_window)

            """
            # Hibernation Heart BPM Calculation
            h_chunk_windowed = h_chunk_clean * fft_window
            h_fft = np.abs(fft(h_chunk_windowed))
            h_fft_freq = fftfreq(len(h_chunk_windowed), 1/smplPerMinute)
            h_mask = (h_fft_freq >= HEART_RATE_HIBERNATION_MIN) & (h_fft_freq <= HEART_RATE_HIBERNATION_MAX)
            hrpm = h_fft_freq[h_mask][np.argmax(h_fft[h_mask])]
            """
            
            hibernation_data.append({
                "Time_Sec": round(time_sec, 2),
                "Hibernation_BPM": round(bpm, 2),
                "Hibernation_BrPM": round(brpm, 2),
            })


        # WINDOW PROCESSING LOOP for arousel

        fft_window_arou = np.blackman(window_samples_arou)

        for start in range(0, len(heart_signal_raw) - window_samples_arou, step_samples_arou):
            
            # Arousal Analysis 
            ah_chunk = arousal_heart_raw[start : start + window_samples_arou] * fft_window_arou
            ah_yf = np.abs(fft(ah_chunk))
            ah_mask = (h_xf >= 1.6) & (h_xf <= 6.0)
            arousal_bpm = h_xf[ah_mask][np.argmax(ah_yf[ah_mask])] * 60

            ar_chunk = arousal_resp_raw[start : start + window_samples] * fft_window_arou
            ar_yf = np.abs(fft(ar_chunk))
            ar_mask = (r_xf >= 0.5) & (r_xf <= 2.0)
            arousal_brpm = r_xf[ar_mask][np.argmax(ar_yf[ar_mask])] * 60

            # Activity / Shivering metric
            movement_intensity = np.std(phi_detrended[start : start + window_samples_arou]) #compute standard deviation, detecting bigger motions

            arousal_data.append({
                "Time_Sec": time_sec, 
                "Arousal_BPM": round(arousal_bpm, 2), 
                "Arousal_BrPM": round(arousal_brpm, 2), 
                "Movement_Index": round(movement_intensity, 4)
            })

            # DIAGNOSTIC PLOT: MAGNITUDE VS FREQ 
            if not diagnostic_fft_saved:
                diag_fig, (d_ax1, d_ax2) = plt.subplots(2, 1, figsize=(10, 6))
                d_ax1.plot(r_xf[r_mask] * 60, r_yf[r_mask], color='blue')
                d_ax1.set_title("Respiration Spectrum (Magnitude vs. Frequency)")
                d_ax1.set_xlabel("Breaths Per Minute (BrPM)")
                
                d_ax2.plot(h_xf[h_mask] * 60, h_yf[h_mask], color='red')
                d_ax2.set_title("Heart Spectrum (Magnitude vs. Frequency)")
                d_ax2.set_xlabel("Beats Per Minute (BPM)")
                
                plt.tight_layout()
                plt.savefig(os.path.join(ROOT_DIR, "diagnostic_fft_window.png"), dpi=150)
                plt.close(diag_fig)
                print(">>> Saved diagnostic spectrum profile to ROOT_DIR.")
                diagnostic_fft_saved = True

        # PLOTTING & EXPORT
        time_axis = np.linspace(0, len(results)/fs, len(results))
        resp_mm = (resp_signal * wavelength * 1000) / (4 * np.pi)
        heart_mm = (heart_signal_final * wavelength * 1000) / (4 * np.pi)

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

    except Exception as e:
        print(f"Error in {rec_folder}: {e}")

print("\n--- BATCH COMPLETE ---")


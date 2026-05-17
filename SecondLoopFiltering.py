'''
Second loop for filtering data
Starts from the angles.npy 
'''

from PreProcessingRadarLib import *
from FilteringLib import *



recording_folders = [f for f in os.listdir(ROOT_DIR) if f.startswith("recording_")]

for rec_folder in recording_folders:
    configData, results = importRadarAngles(data_path)
    print(f"Successfully loaded config and {len(results)} precalculated frames.")
    
    try:
        phase_unwrapped = np.unwrap(results)
        phi_detrended = detrend(phase_unwrapped)

        # SIGNAL FILTERING 
        # Deep Hibernation Heart: 2-15 BPM (0.033 Hz - 0.25 Hz)
        heart_signal_raw = apply_filter(phi_detrended, [0.033, 0.25], fs, btype='bandpass')
        # Deep Hibernation Resp: Lowered ceiling to under 5 BrPM 
        resp_signal = apply_filter(phi_detrended, 0.083, fs, btype='low')

        # Arousal Ranges 
        arousal_heart_raw = apply_filter(phi_detrended, [1.6, 6.0], fs, btype='bandpass') # 100-360 BPM
        arousal_resp_raw = apply_filter(phi_detrended, 0.83 , fs, btype='lowpass')  # until 50 BrPM

        # Sliding Window Settings
        window_sec, step_sec = 180, 5
        window_arou, step_arou = 60, 2
        window_samples, step_samples = int(fs * window_sec), int(fs * step_sec)
        window_samples_arou, step_samples_arou = int(fs * window_arou), int(fs *step_arou)
        n_fft = 32768
        
        hibernation_data = []
        arousal_data = []

        
        heart_signal_final = np.zeros_like(heart_signal_raw)

        # WINDOW PROCESSING LOOP

        for start in range(0, len(heart_signal_raw) - window_samples, step_samples):
            
            # Respiration Frequency Calculation
            r_chunk = resp_signal[start : start + window_samples] * np.blackman(window_samples)
            r_yf = np.abs(fft(r_chunk, n=n_fft))
            r_xf = fftfreq(n_fft, 1/fs)
            # Mask updated for stricter breathing parameters (0.5 to 5 BrPM)
            r_mask = (r_xf >= 0.0083) & (r_xf <= 0.083)
            brpm = r_xf[r_mask][np.argmax(r_yf[r_mask])] * 60

            # Dynamic Notch for Deep Hibernation 
            harmonic_hz = (brpm * 2) / 60
            h_chunk_raw = heart_signal_raw[start : start + window_samples]
            h_chunk_clean = apply_notch_filter(h_chunk_raw, harmonic_hz, fs)
            heart_signal_final[start : start + window_samples] = h_chunk_clean

            # Hibernation Heart BPM Calculation
            h_chunk_windowed = h_chunk_clean * np.blackman(window_samples)
            h_yf = np.abs(fft(h_chunk_windowed, n=n_fft))
            h_xf = fftfreq(n_fft, 1/fs)
            h_mask = (h_xf >= 0.033) & (h_xf <= 0.3)
            bpm = h_xf[h_mask][np.argmax(h_yf[h_mask])] * 60

            hibernation_data.append({
                "Time_Sec": round((start + window_samples/2)/fs, 2),
                "Hibernation_BPM": round(bpm, 2),
                "Hibernation_BrPM": round(brpm, 2),
            })

        for start in range(0, len(heart_signal_raw) - window_samples_arou, step_samples_arou):
            
            # Arousal Analysis 
            ah_chunk = arousal_heart_raw[start : start + window_samples_arou] * np.blackman(window_samples_arou)
            ah_yf = np.abs(fft(ah_chunk, n=n_fft_arou))
            ah_mask = (h_xf >= 1.6) & (h_xf <= 6.0)
            arousal_bpm = h_xf[ah_mask][np.argmax(ah_yf[ah_mask])] * 60

            ar_chunk = arousal_resp_raw[start : start + window_samples] * np.blackman(window_samples_arou)
            ar_yf = np.abs(fft(ar_chunk, n=n_fft))
            ar_mask = (r_xf >= 0.5) & (r_xf <= 2.0)
            arousal_brpm = r_xf[ar_mask][np.argmax(ar_yf[ar_mask])] * 60

            # Activity / Shivering metric
            movement_intensity = np.std(phi_detrended[start : start + window_samples_arou]) #compute standard deviation, detecting bigger motions

            arousal_data.append({
                "Time_Sec": time_sec, 
                "Arousal_BPM": round(a_bpm, 2), 
                "Arousal_BrPM": round(a_brpm, 2), 
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
        v_time = [v['Time_Sec'] for v in combined_vitals]
        v_a_bpm = [v['Arousal_BPM'] for v in combined_vitals]
        v_move = [v['Movement_Index'] for v in combined_vitals]
        
        ax_ah.plot(v_time, v_a_bpm, color='purple', marker='o', markersize=2, linestyle='-')
        ax_ah.set_title(f"Arousal Tracking - High Frequency Heart Spectrum (100-360 BPM)")
        ax_am.plot(v_time, v_move, color='orange', linewidth=0.8)
        ax_am.set_title("Movement / Shivering Index (Phase Variance)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(data_path, f"arousal_plot_{rec_folder}.png"), dpi=150)
        plt.close(fig_arousal)

        # Save CSV Data
        df_final = pd.DataFrame(combined_vitals)
        df_final.to_csv(os.path.join(data_path, f"vitals_{rec_folder}.csv"), index=False)
        print(f"Done! Saved Baseline Plot, Arousal Plot, and CSV for {rec_folder}")

    except Exception as e:
        print(f"Error in {rec_folder}: {e}")

print("\n--- BATCH COMPLETE ---")



'''
separate processing - heart rate on a short window, breathing rate on a long window
'''

import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from pathlib import Path
from radarTestWaves import *




 
def calcRespirationRate(radarConfig: radarConfigData, distData: np.ndarray, 
                        breathRange: list, zero_pad_factor: float = 2.0) -> dict:
    frames_per_min = int(radarConfig.frame_rate_Hz * 60)
    plotDataLow = distData.copy()

    plotDataLow[:, 1] = filterData(plotDataLow[:, 1], breathRange[0], breathRange[1], frames_per_min, order=4)
    breathResult = calcRateSNR3(
        plotDataLow[:, 1], radarConfig.frame_rate_Hz, n_harmonics=4,
        min_freq=max(0.01, breathRange[0] * 0.5), max_freq=breathRange[1] * 1.5,
        zero_pad_factor=zero_pad_factor
    )

    return {
        'Breathrate': breathResult["rate"],
        'BreathStd': np.std(plotDataLow[:, 1]),
        'Breath_SNR_dB': breathResult["snr_dB"],
        'Breath_Prominence': breathResult["prominence_ratio"]
    }


def calcHeartRate(radarConfig: radarConfigData, distData: np.ndarray, 
                  heartRange: list, breathRate: float = None, zero_pad_factor: float = 2.0) -> dict:
    frames_per_min = int(radarConfig.frame_rate_Hz * 60)
    plotDataHigh = distData.copy()

    # Notch out breathing harmonics from the short heart window
    if breathRate is not None and breathRate > 0:
        for harm in [2, 3, 4]:
            plotDataHigh[:, 1] = notchFilter(plotDataHigh[:, 1], freq=breathRate * harm, Q=5, fs=frames_per_min)

    plotDataHigh[:, 1] = filterData(plotDataHigh[:, 1], heartRange[0], heartRange[1], frames_per_min, order=4)

    heartResult = calcRateSNR3(
        plotDataHigh[:, 1], radarConfig.frame_rate_Hz, n_harmonics=4,
        min_freq=max(0.01, heartRange[0] * 0.5), max_freq=heartRange[1] * 1.5,
        zero_pad_factor=zero_pad_factor
    )

    return {
        'Heartrate': heartResult["rate"],
        'HeartStd': np.std(plotDataHigh[:, 1]),
        'Heart_SNR_dB': heartResult["snr_dB"],
        'Heart_Prominence': heartResult["prominence_ratio"]
    }


# =====================================================================
# 2. Time-Series Windowing Routines
# =====================================================================

def calcRespRateOverTime(dataPath: str, plotRangeResp: int = 300, zero_pad_factor: float = 2.0) -> list[dict]:
    dataFilename = getDistFilename(dataPath, USE_ALT_PHASE, USE_ALT_DIST)
    distData = np.load(dataFilename)
    distCount = distData.shape[0]
    radarConfig = importRadarConfig(dataPath)
    breath_rates = []

    rateLenResp = int(radarConfig.frame_rate_Hz * plotRangeResp)
    if distCount < rateLenResp:
        print(f"  [WARN] {dataPath}: Length ({distCount} frames) < window size ({rateLenResp} frames). Skipping.")
        return breath_rates

    for ofs in range(0, distCount - rateLenResp, rateLenResp // 2):
        rateData = distData[ofs:ofs + rateLenResp, :]
        window_timestamp = distData[ofs, 0]
        window_std = np.std(rateData[:, 1])

        breath_rate = calcRespirationRate(radarConfig, rateData, HIB_BREATH_BAND, zero_pad_factor=zero_pad_factor)
        breath_rates.append({
            "Timestamp_s": window_timestamp,
            "Window_Start_s": ofs / radarConfig.frame_rate_Hz,
            "Window_End_s": (ofs + rateLenResp) / radarConfig.frame_rate_Hz,
            "Breathing_Rate_BPM": breath_rate['Breathrate'],
            "Breathing_Std": breath_rate['BreathStd'],
            "Breath_SNR_dB": breath_rate['Breath_SNR_dB'],
            "Breath_Prominence": breath_rate['Breath_Prominence'],
            "Arousal_Flag": "hibernating",
            "Displacement_Std": window_std
        })

    return breath_rates


def calcHeartRateOverTime(dataPath: str, plotRangeHeart: int = 60, 
                          breath_rates: list = None, zero_pad_factor: float = 2.0) -> list[dict]:
    dataFilename = getDistFilename(dataPath, USE_ALT_PHASE, USE_ALT_DIST)
    distData = np.load(dataFilename)
    distCount = distData.shape[0]
    radarConfig = importRadarConfig(dataPath)
    heart_rates = []

    breath_df = pd.DataFrame(breath_rates) if breath_rates else None
    rateLenHeart = int(radarConfig.frame_rate_Hz * plotRangeHeart)

    if distCount < rateLenHeart:
        return heart_rates

    for ofs in range(0, distCount - rateLenHeart, rateLenHeart // 2):
        rateData = distData[ofs:ofs + rateLenHeart, :]
        window_timestamp = distData[ofs, 0]
        window_std = np.std(rateData[:, 1])
        window_start_s = ofs / radarConfig.frame_rate_Hz

        # Find matching respiration rate to notch out harmonics
        breathRate = None
        if breath_df is not None and not breath_df.empty:
            match = breath_df[(breath_df["Window_Start_s"] <= window_start_s) &
                              (breath_df["Window_End_s"] >= window_start_s)]
            if not match.empty:
                breathRate = match["Breathing_Rate_BPM"].iloc[0]
            else:
                breathRate = breath_df["Breathing_Rate_BPM"].median()

        heart_rate = calcHeartRate(radarConfig, rateData, HIB_HEART_BAND, breathRate=breathRate, zero_pad_factor=zero_pad_factor)
        heart_rates.append({
            "Timestamp_s": window_timestamp,
            "Heart_Rate_BPM": heart_rate['Heartrate'],
            "Heart_Std": heart_rate['HeartStd'],
            "Heart_SNR_dB": heart_rate['Heart_SNR_dB'],
            "Heart_Prominence": heart_rate['Heart_Prominence'],
            "Arousal_Flag": "hibernating",
            "Displacement_Std": window_std
        })

    return heart_rates



def _spectrum(sig: np.ndarray, framerate_Hz: float, max_bpm: float, zero_pad_factor: float = 2.0):
    '''Normalized spectrum for plotting with 2x zero-padding matching calcRateSNR resolution.'''
    norm = (sig - np.mean(sig)) / np.std(sig)
    windowed = norm * sp.signal.get_window(window='blackman', Nx=len(sig))
    
    nfft = int(len(sig) * zero_pad_factor)
    spec = np.abs(sp.fft.rfft(windowed, n=nfft))
    spec[0] = 0
    
    freq = sp.fft.rfftfreq(nfft, 1 / (framerate_Hz * 60))
    idx = np.searchsorted(freq, max_bpm)
    
    max_val = max(spec[:idx].max(), 1e-20) if idx > 0 else 1.0
    return freq[:idx], spec[:idx] / max_val


def plotDualWindowChunk(dataPath: str, breathBand: list, heartBand: list,
                        plotRangeResp: float = 300, plotRangeHeart: float = 60,
                        savePlot: bool = False, max_plot_bpm: float = None,
                        zero_pad_factor: float = 2.0):
    if max_plot_bpm is None:
        max_plot_bpm = MAX_PLOT_BPM
    name_suffix = ' - alt' if USE_ALT_DIST else ''

    dataFilename = getDistFilename(dataPath, USE_ALT_PHASE, USE_ALT_DIST)
    distData = np.load(dataFilename)
    radarConfig = importRadarConfig(dataPath)

    frames_to_skip = int(10.0 * radarConfig.frame_rate_Hz)
    if distData.shape[0] > frames_to_skip:
        distData = distData[frames_to_skip:, :]

    breathLen = int(radarConfig.frame_rate_Hz * plotRangeResp)
    heartLen  = int(radarConfig.frame_rate_Hz * plotRangeHeart)
    if distData.shape[0] < breathLen:
        print(f"Skipping {dataPath} - shorter than the {plotRangeResp}s breathing window.")
        return None

    breath_ofs = (distData.shape[0] - breathLen) // 2
    heart_ofs  = breath_ofs + (breathLen - heartLen) // 2

    breathWindowData = distData[breath_ofs: breath_ofs + breathLen, :]
    heartWindowData  = distData[heart_ofs: heart_ofs + heartLen, :]

    breath_result = calcRespirationRate(radarConfig, breathWindowData, breathBand, zero_pad_factor=zero_pad_factor)
    heart_result  = calcHeartRate(radarConfig, heartWindowData, heartBand,
                                  breathRate=breath_result['Breathrate'], zero_pad_factor=zero_pad_factor)

    frames_per_min = int(radarConfig.frame_rate_Hz * 60)
    breathFiltered = filterData(breathWindowData[:, 1].copy(), breathBand[0], breathBand[1], frames_per_min, order=4)
    heartFiltered = heartWindowData[:, 1].copy()
    for h in [2, 3, 4]:
        heartFiltered = notchFilter(heartFiltered, freq=breath_result['Breathrate'] * h, Q=5, fs=frames_per_min)
    heartFiltered = filterData(heartFiltered, heartBand[0], heartBand[1], frames_per_min, order=4)

    breath_t = np.arange(len(breathFiltered)) / radarConfig.frame_rate_Hz
    heart_t  = np.arange(len(heartFiltered)) / radarConfig.frame_rate_Hz

    fig, (ax_breath, ax_heart, ax_spec) = plt.subplots(3, 1, figsize=(15, 11))

    ax_breath.plot(breath_t, breathFiltered, color='blue')
    ax_breath.grid(True)
    ax_breath.set_title(f"Breathing Displacement ({plotRangeResp:.0f}s window) - "
                        f"{breath_result['Breathrate']:.2f} bpm, SNR={breath_result['Breath_SNR_dB']:.1f}dB, "
                        f"prom={breath_result['Breath_Prominence']:.2f}")
    ax_breath.set_ylabel("Displacement (m)")
    ax_breath.set_xlabel("Time (s)")

    ax_heart.plot(heart_t, heartFiltered, color='red')
    ax_heart.grid(True)
    ax_heart.set_title(f"Heart Displacement ({plotRangeHeart:.0f}s window) - "
                       f"{heart_result['Heartrate']:.2f} bpm, SNR={heart_result['Heart_SNR_dB']:.1f}dB, "
                       f"prom={heart_result['Heart_Prominence']:.2f}")
    ax_heart.set_ylabel("Displacement (m)")
    ax_heart.set_xlabel("Time (s)")

    breathFreq, breathSpec = _spectrum(breathFiltered, radarConfig.frame_rate_Hz, max_plot_bpm, zero_pad_factor=zero_pad_factor)
    heartFreq, heartSpec   = _spectrum(heartFiltered, radarConfig.frame_rate_Hz, max_plot_bpm, zero_pad_factor=zero_pad_factor)

    ax_spec.plot(breathFreq, breathSpec, color='blue', label='Respiration')
    ax_spec.plot(heartFreq, heartSpec, color='red', label='Heart')
    ax_spec.axvline(breath_result['Breathrate'], color='blue', linestyle=":")
    ax_spec.annotate(f"{breath_result['Breathrate']:.2f}", (breath_result['Breathrate'], 0),
                      textcoords="offset points", xytext=(5, 5), color='blue', fontsize=9)
    ax_spec.axvline(heart_result['Heartrate'], color='red', linestyle=":")
    ax_spec.annotate(f"{heart_result['Heartrate']:.2f}", (heart_result['Heartrate'], 0),
                      textcoords="offset points", xytext=(5, 5), color='red', fontsize=9)
    ax_spec.grid(True)
    ax_spec.set_title("Combined Frequency Spectrum (2x Zero-Padded Resolution)")
    ax_spec.set_xlabel("Frequency (BPM)")
    ax_spec.set_ylabel("Normalized Magnitude")
    ax_spec.legend(loc="upper right")

    fig.suptitle(dataPath)
    plt.tight_layout()

    if savePlot:
        plt.savefig(os.path.join(dataPath, f"dualwindow_plot{name_suffix}.png"), dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return {**breath_result, **heart_result}



if __name__ == "__main__":
    from radarTestDefs import radarFilePaths1

    for data_path in radarFilePaths1:
        try:
            # Plot generation
            plotDualWindowChunk(data_path, HIB_BREATH_BAND, HIB_HEART_BAND, savePlot=True)
            print(f"Plot saved for: {data_path}")

            # Rate calculations (returns Python lists)
            breath_rates = calcRespRateOverTime(data_path, plotRangeResp=300)
            heart_rates  = calcHeartRateOverTime(data_path, plotRangeHeart=60, breath_rates=breath_rates) #should try making it higher?

            # Convert Respiration to DataFrame & add UTC column
            df_breath = pd.DataFrame(breath_rates)
            if not df_breath.empty:
                df_breath["Timestamp_local"] = pd.to_datetime(df_breath["Timestamp_s"], unit="s", utc=True).dt.tz_convert("Europe/Berlin")
                df_breath["Timestamp_UTC"] = pd.to_datetime(df_breath["Timestamp_s"], unit="s", utc=True)
                df_breath.to_csv(os.path.join(data_path, "breathing_rates.csv"), index=False)

            # Convert Heart Rate to DataFrame & add UTC column
            df_heart = pd.DataFrame(heart_rates)
            if not df_heart.empty:
                df_heart["Timestamp_UTC"] = pd.to_datetime(df_heart["Timestamp_s"], unit="s", utc=True)
                df_heart["Timestamp_local"] = pd.to_datetime(df_heart["Timestamp_s"], unit="s", utc=True).dt.tz_convert("Europe/Berlin")
                df_heart.to_csv(os.path.join(data_path, "heart_rates.csv"), index=False)

            print(f"Saved CSVs for: {data_path}")

        except Exception:
            print(f"Failed: {data_path}")
            import traceback
            traceback.print_exc()


# =====================================================================
# 3. Merging & Batch Summary Helpers
# =====================================================================

def combineRates(dataPath: str, breath_rates: list, heart_rates: list, save: bool = True) -> pd.DataFrame:
    """Combines respiration and heart rate time series on timestamp."""
    df_breath = pd.DataFrame(breath_rates)
    df_heart = pd.DataFrame(heart_rates)

    if df_breath.empty or df_heart.empty:
        return pd.DataFrame()

    # Time conversion
    for df in (df_breath, df_heart):
        df["Timestamp_UTC"] = pd.to_datetime(df["Timestamp_s"], unit="s", utc=True)
        df["Timestamp_local"] = df["Timestamp_UTC"].dt.tz_convert("Europe/Berlin")

    df_breath.sort_values("Timestamp_s", inplace=True)
    df_heart.sort_values("Timestamp_s", inplace=True)

    # Save individual CSVs
    df_breath.to_csv(os.path.join(dataPath, "breathing_rates.csv"), index=False)
    df_heart.to_csv(os.path.join(dataPath, "heart_rates.csv"), index=False)

    # Merge nearest timestamps
    overlap_cols = [c for c in ["Arousal_Flag", "Displacement_Std", "Timestamp_UTC", "Timestamp_local"]
                    if c in df_breath.columns and c in df_heart.columns]
    df_breath_clean = df_breath.drop(columns=overlap_cols)

    combined = pd.merge_asof(df_heart, df_breath_clean, on="Timestamp_s", direction="nearest")

    if save:
        combined.to_csv(os.path.join(dataPath, "rates.csv"), index=False)

    return combined



def summarize_batch_dataset(all_combined_dfs: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Generates per-recording metrics and appends an overall grand total summary row."""
    summary_rows = []
    
    for folder_path, df in all_combined_dfs:
        if df.empty:
            continue
            
        # Fix label collision: include parent directory (e.g., "recording_01/RadarIfxBGT60TR13C_00")
        p = Path(folder_path)
        rec_id = f"{p.parent.name}/{p.name}" if p.parent.name else p.name
            
        summary_rows.append({
            "Recording": rec_id,
            "Heart_Mean_BPM": df["Heart_Rate_BPM"].mean(),
            "Heart_SNR_Mean_dB": df["Heart_SNR_dB"].mean(),
            "Heart_Prom_Mean": df["Heart_Prominence"].mean(),
            "Breath_Mean_BPM": df["Breathing_Rate_BPM"].mean(),
            "Breath_SNR_Mean_dB": df["Breath_SNR_dB"].mean(),
            "Breath_Prom_Mean": df["Breath_Prominence"].mean(),
            "Valid_Windows": len(df)
        })

    summary_df = pd.DataFrame(summary_rows)

    # Append overall grand mean across all recordings
    if not summary_df.empty:
        overall_row = {
            "Recording": "=== OVERALL MEAN ===",
            "Heart_Mean_BPM": summary_df["Heart_Mean_BPM"].mean(),
            "Heart_SNR_Mean_dB": summary_df["Heart_SNR_Mean_dB"].mean(),
            "Heart_Prom_Mean": summary_df["Heart_Prom_Mean"].mean(),
            "Breath_Mean_BPM": summary_df["Breath_Mean_BPM"].mean(),
            "Breath_SNR_Mean_dB": summary_df["Breath_SNR_Mean_dB"].mean(),
            "Breath_Prom_Mean": summary_df["Breath_Prom_Mean"].mean(),
            "Valid_Windows": summary_df["Valid_Windows"].sum()
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)

    return summary_df

# =====================================================================
# 4. Main Batch Runner
# =====================================================================

def run_batch_processing(paths: list[str], plotRangeResp: int = 300, plotRangeHeart: int = 60):
    all_dfs = []

    for data_path in paths:
        try:
            print(f"\nProcessing: {data_path}")
            
            # 1. Compute Respiration & Heart Rate over time
            breath_rates = calcRespRateOverTime(data_path, plotRangeResp=plotRangeResp)
            heart_rates = calcHeartRateOverTime(data_path, plotRangeHeart=plotRangeHeart, breath_rates=breath_rates)

            # 2. Combine and save individual CSVs
            combined_df = combineRates(data_path, breath_rates, heart_rates, save=True)
            if not combined_df.empty:
                all_dfs.append((data_path, combined_df))
                print(f"  -> Successfully generated rates.csv ({len(combined_df)} rows)")

        except Exception as e:
            print(f"  [ERROR] Failed processing {data_path}: {e}")
            traceback.print_exc()

    # 3. Aggregate Batch Summary
    if all_dfs:
        summary_df = summarize_batch_dataset(all_dfs)
        summary_path = "batch_vitals_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print("\n" + "="*50)
        print("=== BATCH PROCESSING SUMMARY ===")
        print("="*50)
        print(summary_df.to_string(index=False))
        print(f"\nBatch summary written to {summary_path}")


if __name__ == "__main__":
    from radarTestDefs import radarFilePaths1
    
    # Run batch process
    run_batch_processing(radarFilePaths1, plotRangeResp=300, plotRangeHeart=60)



'''
separate processing - heart rate on a short window, breathing rate on a long window
'''

import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from radarTestWaves import *



def calcRespirationRate(radarConfig: radarConfigData, distData: np.ndarray, breathRange: list):
    frames_per_min = int(radarConfig.frame_rate_Hz * 60)
    plotDataLow = distData.copy()

    plotDataLow[:, 1] = filterData(plotDataLow[:, 1], breathRange[0], breathRange[1], frames_per_min, order=4)
    breathResult = calcRateSNR(
        plotDataLow[:, 1], radarConfig.frame_rate_Hz, n_harmonics=4,
        min_freq=max(0.01, breathRange[0] * 0.5), max_freq=breathRange[1] * 1.5
    )

    return {
        'Breathrate': breathResult["rate"],
        'BreathStd': np.std(plotDataLow[:, 1]),
        'Breath_SNR_dB': breathResult["snr_dB"],
        'Breath_Prominence': breathResult["prominence_ratio"]
    }


def calcHeartRate(radarConfig: radarConfigData, distData: np.ndarray, heartRange: list, breathRate: float = None):
    '''
    :param breathRate: breathing rate (1/min) to notch out of the heart-band signal before
                        filtering. Must be passed in explicitly - this function has no way
                        to know it on its own, since it only ever sees the short heart-window
                        slice, which is too short to reliably resolve breathing itself.
    '''
    frames_per_min = int(radarConfig.frame_rate_Hz * 60)
    plotDataHigh = distData.copy()

    if breathRate is not None and breathRate > 0:
        for harm in [2, 3, 4]:
            plotDataHigh[:, 1] = notchFilter(plotDataHigh[:, 1], freq=breathRate * harm, Q=5, fs=frames_per_min)

    plotDataHigh[:, 1] = filterData(plotDataHigh[:, 1], heartRange[0], heartRange[1], frames_per_min, order=4)

    heartResult = calcRateSNR(
        plotDataHigh[:, 1], radarConfig.frame_rate_Hz, n_harmonics=4,
        min_freq=max(0.01, heartRange[0] * 0.5), max_freq=heartRange[1] * 1.5
    )

    return {
        'Heartrate': heartResult["rate"],
        'HeartStd': np.std(plotDataHigh[:, 1]),
        'Heart_SNR_dB': heartResult["snr_dB"],
        'Heart_Prominence': heartResult["prominence_ratio"]
    }


def calcRespRateOverTime(dataPath: str, plotRangeResp: int):
    dataFilename = getDistFilename(dataPath, USE_ALT_PHASE, USE_ALT_DIST)
    distData     = np.load(dataFilename)
    distCount    = distData.shape[0]
    radarConfig  = importRadarConfig(dataPath)
    breath_rates = []

    rateLenResp = int(radarConfig.frame_rate_Hz * plotRangeResp)
    if distCount < rateLenResp:
        print(f"   WARNING: recording has {distCount} frames, shorter than one "
              f"{plotRangeResp}s window ({rateLenResp} frames needed) - skipping, 0 rate windows produced.")
        return breath_rates

    for ofs in range(0, distCount - rateLenResp, rateLenResp // 2):
        rateData = distData[ofs:ofs + rateLenResp, :]
        window_timestamp = distData[ofs, 0]
        displacement = rateData[:, 1]
        window_std = np.std(displacement)
        status = "hibernating"

        breath_rate = calcRespirationRate(radarConfig, rateData, HIB_BREATH_BAND)
        breath_rates.append({
            "Timestamp_s": window_timestamp,
            "Window_Start_s": ofs / radarConfig.frame_rate_Hz,
            "Window_End_s": (ofs + rateLenResp) / radarConfig.frame_rate_Hz,
            "Breathing_Rate_BPM": breath_rate['Breathrate'],
            "Breathing_Std": breath_rate['BreathStd'],
            "Breath_SNR_dB": breath_rate['Breath_SNR_dB'],
            "Breath_Prominence": breath_rate['Breath_Prominence'],
            "Arousal_Flag": status,
            "Displacement_Std": window_std
        })

    return breath_rates


def calcHeartRateOverTime(dataPath: str, plotRangeHeart: int, breath_rates: list = None):
    '''
    :param breath_rates: output of calcRespRateOverTime() for this SAME recording - required
                          so each heart window can look up the breathing rate of whichever
                          (longer) breathing window contains it, and notch it out correctly.
                          If not provided, notch filtering is skipped entirely for every
                          window (heart rate is still computed, just without breathing-
                          harmonic removal - expect more contamination, same as the
                          undefined-breathRate bug this replaces, but without crashing).
    '''
    dataFilename = getDistFilename(dataPath, USE_ALT_PHASE, USE_ALT_DIST)
    distData     = np.load(dataFilename)
    distCount    = distData.shape[0]
    radarConfig  = importRadarConfig(dataPath)
    heart_rates  = []

    breath_df = pd.DataFrame(breath_rates) if breath_rates else None

    rateLenHeart = int(radarConfig.frame_rate_Hz * plotRangeHeart)
    if distCount < rateLenHeart:
        print(f"   WARNING: recording has {distCount} frames, shorter than one "
              f"{plotRangeHeart}s window ({rateLenHeart} frames needed) - skipping, 0 rate windows produced.")
        return heart_rates

    for ofs in range(0, distCount - rateLenHeart, rateLenHeart // 2):
        rateData = distData[ofs:ofs + rateLenHeart, :]
        window_timestamp = distData[ofs, 0]
        displacement = rateData[:, 1]
        window_std = np.std(displacement)
        status = "hibernating"
        window_start_s = ofs / radarConfig.frame_rate_Hz

        breathRate = None
        if breath_df is not None and len(breath_df):
            match = breath_df[(breath_df["Window_Start_s"] <= window_start_s) &
                               (breath_df["Window_End_s"] >= window_start_s)]
            if len(match):
                breathRate = match["Breathing_Rate_BPM"].iloc[0]
            else:
                breathRate = breath_df["Breathing_Rate_BPM"].median()

        heart_rate = calcHeartRate(radarConfig, rateData, HIB_HEART_BAND, breathRate=breathRate)
        heart_rates.append({
            "Timestamp_s": window_timestamp,
            "Heart_Rate_BPM": heart_rate['Heartrate'],
            "Heart_Std": heart_rate['HeartStd'],
            "Heart_SNR_dB": heart_rate['Heart_SNR_dB'],
            "Heart_Prominence": heart_rate['Heart_Prominence'],
            "Arousal_Flag": status,
            "Displacement_Std": window_std
        })

    return heart_rates



def _spectrum(sig: np.ndarray, framerate_Hz: float, max_bpm: float):
    '''normalized spectrum for plotting - zero-mean BEFORE windowing (avoids DC leakage),
    DC bin suppressed, magnitude scaled 0-1 so breath/heart curves are visually comparable'''
    norm = (sig - np.mean(sig)) / np.std(sig)
    windowed = norm * sp.signal.get_window(window='blackman', Nx=len(sig))
    spec = np.abs(sp.fft.rfft(windowed))
    spec[0] = 0
    freq = sp.fft.rfftfreq(len(sig), 1 / (framerate_Hz * 60))
    idx = FFTidx(max_bpm, len(sig), framerate_Hz * 60)
    return freq[:idx], spec[:idx] / max(spec[:idx].max(), 1e-20)


def plotDualWindowChunk(dataPath: str, breathBand: list, heartBand: list,
                         plotRangeResp: float = 300, plotRangeHeart: float = 60,
                         savePlot: bool = False, max_plot_bpm: float = None):
    if max_plot_bpm is None:
        max_plot_bpm = MAX_PLOT_BPM
    name_suffix = ' - alt' if USE_ALT_DIST else ''

    dataFilename = getDistFilename(dataPath, USE_ALT_PHASE, USE_ALT_DIST)
    distData = np.load(dataFilename)
    radarConfig = importRadarConfig(dataPath)

    # skip startup transient, same convention as plotDistData()
    frames_to_skip = int(10.0 * radarConfig.frame_rate_Hz)
    if distData.shape[0] > frames_to_skip:
        distData = distData[frames_to_skip:, :]

    breathLen = int(radarConfig.frame_rate_Hz * plotRangeResp)
    heartLen  = int(radarConfig.frame_rate_Hz * plotRangeHeart)
    if distData.shape[0] < breathLen:
        print(f"Skipping {dataPath} - shorter than the {plotRangeResp}s breathing window.")
        return None

    breath_ofs = (distData.shape[0] - breathLen) // 2                  # middle of recording
    heart_ofs  = breath_ofs + (breathLen - heartLen) // 2              # centered inside that same span

    breathWindowData = distData[breath_ofs: breath_ofs + breathLen, :]
    heartWindowData  = distData[heart_ofs: heart_ofs + heartLen, :]

    breath_result = calcRespirationRate(radarConfig, breathWindowData, breathBand)
    heart_result  = calcHeartRate(radarConfig, heartWindowData, heartBand,
                                   breathRate=breath_result['Breathrate'])

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

    breathFreq, breathSpec = _spectrum(breathFiltered, radarConfig.frame_rate_Hz, max_plot_bpm)
    heartFreq, heartSpec = _spectrum(heartFiltered, radarConfig.frame_rate_Hz, max_plot_bpm)
    ax_spec.plot(breathFreq, breathSpec, color='blue', label='Respiration')
    ax_spec.plot(heartFreq, heartSpec, color='red', label='Heart')
    ax_spec.axvline(breath_result['Breathrate'], color='blue', linestyle=":")
    ax_spec.annotate(f"{breath_result['Breathrate']:.2f}", (breath_result['Breathrate'], 0),
                      textcoords="offset points", xytext=(5, 5), color='blue', fontsize=9)
    ax_spec.axvline(heart_result['Heartrate'], color='red', linestyle=":")
    ax_spec.annotate(f"{heart_result['Heartrate']:.2f}", (heart_result['Heartrate'], 0),
                      textcoords="offset points", xytext=(5, 5), color='red', fontsize=9)
    ax_spec.grid(True)
    ax_spec.set_title("Combined Frequency Spectrum")
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
            heart_rates  = calcHeartRateOverTime(data_path, plotRangeHeart=60, breath_rates=breath_rates)

            # Convert Respiration to DataFrame & add UTC column
            df_breath = pd.DataFrame(breath_rates)
            if not df_breath.empty:
                df_breath["Timestamp_UTC"] = pd.to_datetime(df_breath["Timestamp_s"], unit="s", utc=True)
                df_breath.to_csv(os.path.join(data_path, "breathing_rates.csv"), index=False)

            # Convert Heart Rate to DataFrame & add UTC column
            df_heart = pd.DataFrame(heart_rates)
            if not df_heart.empty:
                df_heart["Timestamp_UTC"] = pd.to_datetime(df_heart["Timestamp_s"], unit="s", utc=True)
                df_heart.to_csv(os.path.join(data_path, "heart_rates.csv"), index=False)

            print(f"Saved CSVs for: {data_path}")

        except Exception:
            print(f"Failed: {data_path}")
            import traceback
            traceback.print_exc()


##combine both heart and breaths in one for hibernating:




'''
separate processing for hibernation - heart rate on a short window, breathing rate on a long window 
'''
import os
import traceback
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt


from pathlib import Path
from radarTestWaves import *


SKIP_START_S    = 15.0
BREATH_WINDOW_S = 300
HEART_WINDOW_S  = 60


EDGE_CELLS = {
    "Breath": dict(low=0.75, high=0.25),   # band 0.4-3.0 -> flags < 0.55 (0.4, 0.5 BPM) and >= 2.95 (3.0 BPM)
    "Heart":  dict(low=1.75, high=2.0),    # band 2-30   -> flags < 3.75 and >= 28 (same as before)
}


def edgeFlags(rate, band, n_frames, frame_rate_Hz, cells):
    if rate is None or not np.isfinite(rate):
        return False, False
    cell = 60.0 * frame_rate_Hz / n_frames                  # BPM per Fourier cell
    low  = rate <  band[0] + cells["low"]  * cell
    high = rate >= band[1] - cells["high"] * cell
    return bool(low), bool(high)


def loadDistance(dataPath):
    distData = np.load(getDistFilename(dataPath, USE_ALT_PHASE, USE_ALT_DIST))
    radarConfig = importRadarConfig(dataPath)
    skip = int(SKIP_START_S * radarConfig.frame_rate_Hz)
    if distData.shape[0] > skip:
        distData = distData[skip:, :]
    return distData, radarConfig


# calc functions
def calcRespirationRate(radarConfig: radarConfigData, distData: np.ndarray,
                        breathRange: list, zero_pad_factor: float = 2.0,
                        cells: dict = None) -> dict:
    
    cells = EDGE_CELLS["Breath"] if cells is None else cells
    frames_per_min = int(radarConfig.frame_rate_Hz * 60)
    plotDataLow = distData.copy()
 
    plotDataLow[:, 1] = filterData(plotDataLow[:, 1], breathRange[0], breathRange[1], frames_per_min, order=4)
    breathResult = calcRateSNR4(
        plotDataLow[:, 1], radarConfig.frame_rate_Hz, n_harmonics=4,
        min_freq=breathRange[0], max_freq=breathRange[1],
        zero_pad_factor=zero_pad_factor
    )
    low, high = edgeFlags(breathResult["rate"], breathRange, len(distData), radarConfig.frame_rate_Hz, cells)
    
    return {
        'Breathrate': breathResult["rate"],
        'BreathStd': np.std(plotDataLow[:, 1]),
        'Breath_SNR_dB': breathResult["snr_dB"],
        'Breath_Prominence': breathResult["prominence_ratio"],
        'Breath_EdgeLow': low, 'Breath_EdgeHigh': high, 'Breath_Edge': low or high,      # NEW
        'Breath_Filtered': plotDataLow[:, 1],      # signal the rate was estimated from (used by the plot)
    }
 

def calcHeartRate(radarConfig: radarConfigData, distData: np.ndarray,
                  heartRange: list, breathRate: float = None, zero_pad_factor: float = 2.0,
                  cells: dict = None) -> dict:
    cells = EDGE_CELLS["Heart"] if cells is None else cells
    
    frames_per_min = int(radarConfig.frame_rate_Hz * 60)
    plotDataHigh = distData.copy()
 
    # Notch out breathing harmonics from the short heart window
    if breathRate is not None and breathRate > 0:
        for harm in [2, 3, 4]:
            plotDataHigh[:, 1] = notchFilter(plotDataHigh[:, 1], freq=breathRate * harm, Q=5, fs=frames_per_min)
 
    plotDataHigh[:, 1] = filterData(plotDataHigh[:, 1], heartRange[0], heartRange[1], frames_per_min, order=4)
 
    heartResult = calcRateSNR4(
        plotDataHigh[:, 1], radarConfig.frame_rate_Hz, n_harmonics=4,
        min_freq=heartRange[0], max_freq=heartRange[1],
        zero_pad_factor=zero_pad_factor
    )
    low, high = edgeFlags(heartResult["rate"], heartRange, len(distData), radarConfig.frame_rate_Hz, cells)
    return {
        'Heartrate': heartResult["rate"],
        'HeartStd': np.std(plotDataHigh[:, 1]),
        'Heart_SNR_dB': heartResult["snr_dB"],
        'Heart_Prominence': heartResult["prominence_ratio"],
        'Heart_EdgeLow': low, 'Heart_EdgeHigh': high, 'Heart_Edge': low or high,
        'Heart_Filtered': plotDataHigh[:, 1],      # signal the rate was estimated from (used by the plot)
    }


# 2. Time-Series Windowing Routines

def windowStarts(n_frames: int, win_len: int):
    """Start frames of the sliding windows (50 % overlap). Used by the table AND by the plot,
    so both always work on the same window grid."""
    return range(0, n_frames - win_len, win_len // 2)


def nearestMidIndex(win_mids_s, heart_mid_s: float) -> int:
    """Index of the breathing window whose midpoint is nearest to the heart window midpoint.
    Same rule as merge_asof(direction='nearest') on Mid_s in combineRates(): the breathing rate that
    ends up in a table row is also the one used for the notch filter of that row's heart window."""
    return int(np.argmin(np.abs(np.asarray(win_mids_s, dtype=float) - heart_mid_s)))


def calcRespRateOverTime(dataPath: str, plotRangeResp: int = BREATH_WINDOW_S, zero_pad_factor: float = 2.0) -> list[dict]:
    
    distData, radarConfig = loadDistance(dataPath)
    distCount = distData.shape[0]
    breath_rates = []

    rateLenResp = int(radarConfig.frame_rate_Hz * plotRangeResp)
    if distCount < rateLenResp:
        print(f"  [WARN] {dataPath}: Length ({distCount} frames) < window size ({rateLenResp} frames). Skipping.")
        return breath_rates

    for ofs in windowStarts(distCount, rateLenResp):
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
            "Breath_Edge": breath_rate['Breath_Edge'],
            "Breath_EdgeLow": breath_rate['Breath_EdgeLow'],
            "Breath_EdgeHigh": breath_rate['Breath_EdgeHigh'],
            "Breath_Window_s": plotRangeResp,
            "Displacement_Std": window_std
        })

    return breath_rates


        
def calcHeartRateOverTime(dataPath: str, plotRangeHeart: int = HEART_WINDOW_S, 
                          breath_rates: list = None, zero_pad_factor: float = 2.0) -> list[dict]:
    
    distData, radarConfig = loadDistance(dataPath)
    distCount = distData.shape[0]
    heart_rates = []

    breath_df = pd.DataFrame(breath_rates) if breath_rates else None
    breath_mids = None
    if breath_df is not None and not breath_df.empty:
        breath_mids = ((breath_df["Window_Start_s"] + breath_df["Window_End_s"]) / 2).to_numpy()
    rateLenHeart = int(radarConfig.frame_rate_Hz * plotRangeHeart)

    if distCount < rateLenHeart:
        return heart_rates

    for ofs in windowStarts(distCount, rateLenHeart):
        rateData = distData[ofs:ofs + rateLenHeart, :]
        window_timestamp = distData[ofs, 0]
        window_std = np.std(rateData[:, 1])
        heart_mid_s = ofs / radarConfig.frame_rate_Hz + plotRangeHeart / 2

        # Breathing window paired with this heart window (nearest midpoint, same rule as combineRates);
        # its rate is used to notch out the breathing harmonics
        breathRate = None
        if breath_mids is not None:
            breathRate = breath_df["Breathing_Rate_BPM"].iloc[nearestMidIndex(breath_mids, heart_mid_s)]

        heart_rate = calcHeartRate(radarConfig, rateData, HIB_HEART_BAND, breathRate=breathRate, zero_pad_factor=zero_pad_factor)
        heart_rates.append({
            "Timestamp_s": window_timestamp,
            "Heart_Rate_BPM": heart_rate['Heartrate'],
            "Heart_Std": heart_rate['HeartStd'],
            "Heart_SNR_dB": heart_rate['Heart_SNR_dB'],
            "Heart_Prominence": heart_rate['Heart_Prominence'],
            "Arousal_Flag": "hibernating",
            "Heart_Edge": heart_rate['Heart_Edge'],
            "Heart_EdgeLow": heart_rate['Heart_EdgeLow'],
            "Heart_EdgeHigh": heart_rate['Heart_EdgeHigh'],
            "Heart_Window_s": plotRangeHeart,
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
                        plotRangeResp: float = BREATH_WINDOW_S, plotRangeHeart: float = HEART_WINDOW_S,
                        savePlot: bool = False, max_plot_bpm: float = None,
                        zero_pad_factor: float = 2.0, heart_timestamp_s: float = None):
    """
    Plots ONE row of the rates table: a heart window from the same sliding grid as calcHeartRateOverTime,
    and the breathing window the table pairs with it (nearest midpoint, which is also the window whose rate
    notches the heart signal). Both windows go through calcRespirationRate / calcHeartRate and the plot shows the
    filtered signals those functions return, so the rates in the titles are the values in the table.

    :param heart_timestamp_s: Timestamp_s of the heart window (column Timestamp_s in rates.csv).
                              None -> the heart window in the middle of the recording.
    """
    if max_plot_bpm is None:
        max_plot_bpm = MAX_PLOT_BPM
    name_suffix = ' - alt' if USE_ALT_DIST else ''

    distData, radarConfig = loadDistance(dataPath)          # SKIP_START_S already removed
    fs = radarConfig.frame_rate_Hz

    # Store it in a variable to use later
    range_val = np.ptp(distData[:, 1])
    print("Peak-to-peak range:", range_val)

    col = distData[:, 1]
    # Print in standard decimal format (fixed-point notation)
    print(f"Min:   {col.min():.8f}")
    print(f"Max:   {col.max():.8f}")
    print(f"Range: {np.ptp(col):.8f}")

    breathLen = int(fs * plotRangeResp)
    heartLen  = int(fs * plotRangeHeart)
    n = distData.shape[0]
    heart_starts  = list(windowStarts(n, heartLen))
    breath_starts = list(windowStarts(n, breathLen))
    if not heart_starts or not breath_starts:
        print(f"Skipping {dataPath} - shorter than one {plotRangeResp}s breathing window.")
        return None

    # heart window: the one in the middle, or the one that starts at the requested Timestamp_s
    if heart_timestamp_s is None:
        k = len(heart_starts) // 2
    else:
        ts = distData[heart_starts, 0]
        k = int(np.argmin(np.abs(ts - heart_timestamp_s)))
        if abs(ts[k] - heart_timestamp_s) > 1.0:
            print(f"[WARN] no heart window starts at Timestamp_s={heart_timestamp_s}; using {ts[k]:.3f}")
    heart_ofs = heart_starts[k]

    # paired breathing window: nearest midpoint (same rule as table)
    heart_mid_s = heart_ofs / fs + plotRangeHeart / 2
    breath_mids = [(o + breathLen / 2) / fs for o in breath_starts]
    breath_ofs  = breath_starts[nearestMidIndex(breath_mids, heart_mid_s)]

    heart_ts  = distData[heart_ofs, 0]
    breath_ts = distData[breath_ofs, 0]
    print(f"heart window : Timestamp_s        = {heart_ts:.3f}  ({heart_ofs / fs:.1f}-{(heart_ofs + heartLen) / fs:.1f} s)")
    print(f"breath window: Breath_Timestamp_s = {breath_ts:.3f}  ({breath_ofs / fs:.1f}-{(breath_ofs + breathLen) / fs:.1f} s)")

    breathWindowData = distData[breath_ofs: breath_ofs + breathLen, :]
    heartWindowData  = distData[heart_ofs: heart_ofs + heartLen, :]

    breath_result = calcRespirationRate(radarConfig, breathWindowData, breathBand, zero_pad_factor=zero_pad_factor)
    heart_result  = calcHeartRate(radarConfig, heartWindowData, heartBand,
                                  breathRate=breath_result['Breathrate'], zero_pad_factor=zero_pad_factor)

    # plot exactly the signals the rates were estimated from
    breathFiltered = breath_result['Breath_Filtered']
    heartFiltered  = heart_result['Heart_Filtered']

    breath_t = np.arange(len(breathFiltered)) / fs
    heart_t  = np.arange(len(heartFiltered)) / fs

    fig, (ax_breath, ax_heart, ax_spec) = plt.subplots(3, 1, figsize=(15, 11))

    ax_breath.plot(breath_t, breathFiltered, color='blue')
    ax_breath.grid(True)
    ax_breath.set_title(f"Breathing Displacement ({plotRangeResp:.0f}s window) - "
                        f"{breath_result['Breathrate']:.1f} bpm, SNR={breath_result['Breath_SNR_dB']:.1f}dB, "
                        f"prom={breath_result['Breath_Prominence']:.2f}")
    ax_breath.set_ylabel("Displacement (m)")
    ax_breath.set_xlabel("Time (s)")

    ax_heart.plot(heart_t, heartFiltered, color='red')
    ax_heart.grid(True)
    ax_heart.set_title(f"Heart Displacement ({plotRangeHeart:.0f}s window) - "
                       f"{heart_result['Heartrate']:.1f} bpm, SNR={heart_result['Heart_SNR_dB']:.1f}dB, "
                       f"prom={heart_result['Heart_Prominence']:.2f}")
    ax_heart.set_ylabel("Displacement (m)")
    ax_heart.set_xlabel("Time (s)")

    breathFreq, breathSpec = _spectrum(breathFiltered, fs, max_plot_bpm, zero_pad_factor=zero_pad_factor)
    heartFreq, heartSpec   = _spectrum(heartFiltered, fs, max_plot_bpm, zero_pad_factor=zero_pad_factor)

    ax_spec.plot(breathFreq, breathSpec, color='blue', label='Respiration')
    ax_spec.plot(heartFreq, heartSpec, color='red', label='Heart')

    br = breath_result['Breathrate']
    hr = heart_result['Heartrate']

    # fundamental frequencies
    ax_spec.axvline(br, color='blue', linestyle=":")
    ax_spec.annotate(f"{br:.1f}", (br, 0),
                     textcoords="offset points", xytext=(5, 5), color='blue', fontsize=9)
    ax_spec.axvline(hr, color='red', linestyle=":")
    ax_spec.annotate(f"{hr:.1f}", (hr, 0),
                     textcoords="offset points", xytext=(5, 5), color='red', fontsize=9)

    # 2x, 3x, 4x harmonics of the heart rate, dotted and fainter than the fundamental;
    # one shared legend entry
    if hr is not None and np.isfinite(hr):
        first = True
        for h in (2, 3, 4):
            hf = hr * h
            if hf < max_plot_bpm:                        # only if inside the plotted range
                ax_spec.axvline(hf, color='red', linestyle=":", alpha=0.5,
                                label="Heart harmonics (2x-4x)" if first else None)
                ax_spec.annotate(f"{h}x", xy=(hf, 1.0), xycoords=("data", "axes fraction"),
                                 ha="center", va="bottom", fontsize=8, color='red')
                first = False

    # no set_xlim: matplotlib's default margins leave space before 0 and after the last value
    ax_spec.grid(True)
    ax_spec.set_title("Combined Frequency Spectrum (2x Zero-Padded Resolution)")
    ax_spec.set_xlabel("Frequency (BPM)")
    ax_spec.set_ylabel("Normalized Magnitude")
    ax_spec.legend(loc="upper right")

    fig.suptitle(f"{dataPath}\nheart window Timestamp_s = {heart_ts:.0f}, "
                 f"breathing window Breath_Timestamp_s = {breath_ts:.0f}")
    plt.tight_layout()

    if savePlot:
        ts_suffix = "" if heart_timestamp_s is None else f"_{heart_ts:.0f}"
        plt.savefig(os.path.join(dataPath, f"dualwindow_plot_final{name_suffix}{ts_suffix}.png"), dpi=150)
        plt.close(fig)
    else:
        plt.show()

    result = {**breath_result, **heart_result}
    result.pop('Breath_Filtered', None)
    result.pop('Heart_Filtered', None)
    return result


# Merge breathing and heart windows to rates.csv 

def combineRates(dataPath: str, breath_rates: list, heart_rates: list, save: bool = True) -> pd.DataFrame:
    """
    Combines the breathing (long window) and heart (short window) time series.
    Windows are matched by their MIDPOINT, not by their start, so every heart window
    gets the breathing window that is centred closest to it.
    """
    df_breath = pd.DataFrame(breath_rates)
    df_heart  = pd.DataFrame(heart_rates)

    if df_breath.empty or df_heart.empty:
        return pd.DataFrame()

    for df in (df_breath, df_heart):
        df["Timestamp_UTC"] = pd.to_datetime(df["Timestamp_s"], unit="s", utc=True)

    df_breath["Mid_s"] = df_breath["Timestamp_s"] + df_breath.get("Breath_Window_s", BREATH_WINDOW_S) / 2
    df_heart["Mid_s"]  = df_heart["Timestamp_s"]  + df_heart.get("Heart_Window_s",  HEART_WINDOW_S) / 2

    df_breath = df_breath.sort_values("Mid_s").reset_index(drop=True)
    df_heart  = df_heart.sort_values("Mid_s").reset_index(drop=True)

    if save:
        df_breath.to_csv(os.path.join(dataPath, "breathing_rates.csv"), index=False)
        df_heart.to_csv(os.path.join(dataPath, "heart_rates.csv"), index=False)

    # columns that exist in both tables stay from the heart table
    overlap_cols = [c for c in ["Arousal_Flag", "Displacement_Std", "Timestamp_UTC"]
                    if c in df_breath.columns and c in df_heart.columns]
    df_breath_clean = (df_breath.drop(columns=overlap_cols)
                                .rename(columns={"Timestamp_s": "Breath_Timestamp_s"}))

    # same pairing rule as nearestMidIndex(): the breathing rate in a row is the one that notched its heart window
    combined = pd.merge_asof(df_heart, df_breath_clean, on="Mid_s", direction="nearest")

    if save:
        combined.to_csv(os.path.join(dataPath, "rates.csv"), index=False)

    return combined


# One recording: rates -> merge -> plot

def processRecording(data_path: str, plotRangeResp: int = BREATH_WINDOW_S,
                     plotRangeHeart: int = HEART_WINDOW_S, make_plot: bool = True) -> pd.DataFrame:
    """Everything that happens to one recording; returns the combined table (also saved as rates.csv)."""
    breath_rates = calcRespRateOverTime(data_path, plotRangeResp=plotRangeResp)
    heart_rates  = calcHeartRateOverTime(data_path, plotRangeHeart=plotRangeHeart, breath_rates=breath_rates)

    combined = combineRates(data_path, breath_rates, heart_rates, save=True)

    if make_plot:
        plotDualWindowChunk(data_path, HIB_BREATH_BAND, HIB_HEART_BAND,
                            plotRangeResp=plotRangeResp, plotRangeHeart=plotRangeHeart, savePlot=True)
    return combined


# 4. Batch summary (all windows vs. edge-free windows)
def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(False).astype(bool)
    return pd.Series(False, index=df.index)

def _snrStats(snr: pd.Series, tag: str, prefix: str) -> dict:
    snr = snr.dropna()
    q1, med, q3 = snr.quantile([.25, .5, .75]) if len(snr) else (np.nan,) * 3
    return {
        f"{prefix}_SNR_Median_dB_{tag}": med,
        f"{prefix}_SNR_Q1_dB_{tag}":     q1,
        f"{prefix}_SNR_Q3_dB_{tag}":     q3,
        f"{prefix}_SNR_Mean_dB_{tag}":   snr.mean(),
        f"{prefix}_N_{tag}":             int(snr.size),
    }

def _summarize(df, prefix, rate_col, snr_col, prom_col, edge_col):
    edge = _flag(df, edge_col)
    ok = ~edge & df[rate_col].notna()
    return {
        **_snrStats(df[snr_col],         "All",   prefix),   # every window
        **_snrStats(df.loc[ok, snr_col], "Valid", prefix),   # edge-free windows only
        f"{prefix}_Edge_Frac": float(edge.mean()),
    }


def _summaryRow(label: str, df: pd.DataFrame) -> dict:
    key = [c for c in ("Recording", "Breath_Timestamp_s") if c in df.columns]
    breath_df = df.drop_duplicates(key) if "Breath_Timestamp_s" in df.columns else df
    return {
        "Recording": label,
        "Heart_Windows": len(df),
        "Breath_Windows": len(breath_df),
        **_summarize(df,        "Heart",  "Heart_Rate_BPM",     "Heart_SNR_dB",  "Heart_Prominence",  "Heart_Edge"),
        **_summarize(breath_df, "Breath", "Breathing_Rate_BPM", "Breath_SNR_dB", "Breath_Prominence", "Breath_Edge"),
    }
def _overlaps(start, end, ivp):
    m = np.zeros(len(start), bool)
    for s, e in zip(ivp.start_s, ivp.end_s):
        m |= (start < e) & (end > s)
    return m

def _inside(start, end, ivp):
    m = np.zeros(len(start), bool)
    for s, e in zip(ivp.start_s, ivp.end_s):
        m |= (start >= s) & (end <= e)
    return m


def summarize_batch_dataset(all_combined_dfs: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Per-recording metrics plus one pooled row over all windows (not a mean of recording means)."""
    rows, frames = [], []
    for folder_path, df in all_combined_dfs:
        if df.empty:
            continue
        p = Path(folder_path)
        rec_id = f"{p.parent.name}/{p.name}" if p.parent.name else p.name
        rows.append(_summaryRow(rec_id, df))
        frames.append(df)

    if not rows:
        return pd.DataFrame()

    rows.append(_summaryRow("=== ALL WINDOWS (pooled) ===", pd.concat(frames, ignore_index=True)))
    return pd.DataFrame(rows)


# Batch runner and entry point

def run_batch_processing(paths: list[str], plotRangeResp: int = BREATH_WINDOW_S,
                         plotRangeHeart: int = HEART_WINDOW_S, make_plot: bool = True):
    all_dfs, skipped = [], []

    for data_path in paths:
        print(f"\nProcessing: {data_path}")

        dist_file = getDistFilename(data_path, USE_ALT_PHASE, USE_ALT_DIST)
        if not os.path.exists(dist_file):
            print(f"  [SKIP] missing distance file: {dist_file}")
            skipped.append(data_path)
            continue

        try:
            combined = processRecording(data_path, plotRangeResp, plotRangeHeart, make_plot)
            if combined.empty:
                print("  [SKIP] no windows produced (recording too short)")
                skipped.append(data_path)
                continue
            all_dfs.append((data_path, combined))
            print(f"  -> rates.csv written ({len(combined)} rows)")
        except Exception:
            print(f"  [ERROR] failed: {data_path}")
            traceback.print_exc()
            skipped.append(data_path)

    if all_dfs:
        summary_df = summarize_batch_dataset(all_dfs)
        summary_path = "batch_vitals_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 50)
        print("=== BATCH PROCESSING SUMMARY ===")
        print("=" * 50)
        print(summary_df.round(2).to_string(index=False))
        print(f"\nBatch summary written to {summary_path}")

    if skipped:
        print(f"\nSkipped / failed recordings ({len(skipped)}):")
        for p in skipped:
            print("  ", p)


def buildStateTable(paths, iv_csv="arousal_intervals.csv", aroused_win_s=30):
    """arousal_intervals.csv: path,start_utc,end_utc (CO2-confirmed arousals)"""
    iv = pd.read_csv(iv_csv)
    iv["start_s"] = pd.to_datetime(iv["start_utc"], utc=True).map(lambda t: t.timestamp())
    iv["end_s"]   = pd.to_datetime(iv["end_utc"],   utc=True).map(lambda t: t.timestamp())

    hib, aro = [], []
    for p in paths:
        ivp = iv[iv["path"] == p]

        # hibernating: neither the heart window nor its paired breathing window touches an arousal
        f = os.path.join(p, "rates.csv")
        if os.path.exists(f):
            h = pd.read_csv(f)
            hs, bs = h["Timestamp_s"].to_numpy(), h["Breath_Timestamp_s"].to_numpy()
            bad = (_overlaps(hs, hs + h["Heart_Window_s"].to_numpy(), ivp) |
                   _overlaps(bs, bs + h["Breath_Window_s"].to_numpy(), ivp))
            hib.append(h[~bad].assign(Recording=p))

        # aroused: window lies fully inside an arousal
        f = os.path.join(p, "awake_rates.csv")
        if os.path.exists(f) and os.path.getsize(f) > 0:
            a = pd.read_csv(f)
            s = a["Timestamp_s"].to_numpy()
            aro.append(a[_inside(s, s + aroused_win_s, ivp)].assign(Recording=p))

    hib_df, aro_df = pd.concat(hib, ignore_index=True), pd.concat(aro, ignore_index=True)
    return pd.DataFrame([_summaryRow("Hibernating", hib_df),
                         _summaryRow("Aroused",     aro_df)])


if __name__ == "__main__":
    from radarTestDefs import radarFilePaths1

    run_batch_processing(radarFilePaths1)            # Writes rates.csv per recording
    table = buildStateTable(radarFilePaths1)         # Extracts and groups by state
    table.to_csv("snr_by_state.csv", index=False)    # Saves the final summary
    print(table.round(2).T.to_string())
    
'''
    radarAttic - code that is currently not used by the active radar pipeline.
    Nothing in here is imported by any other file. To bring a function back, move it into the
    module named in its "from ..." header and add back any import it needs.

    Notes:
    - "(definition N of M)" marks a function that was defined several times in the same file;
      only the LAST definition was ever active, the earlier ones were shadowed.
    - Section order follows the dependency order of the source files.
'''

import os
import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal as signal
from scipy.signal import detrend
import matplotlib.pyplot as plt

# Names these snippets expect from the source modules (not imported here on purpose,
# to avoid circular imports - the attic is an archive, not a library):
#   radarLib:   SPEED_OF_LIGHT, radarConfigData, importRadarConfig, normalize, ANGLES_NAME ...
#   radarTools: PLOT_COLORS, normalize ...


# ====================================================================================================
# from radarLib.py - no caller in any of the six files (or shadowed by a later definition of the same name)
# ====================================================================================================

# --- ANGLES_NAME  [was radarLib.py:20-20]
ANGLES_NAME     = "radar_angles.npy"


# --- calcDistanceFromFrequency  [was radarLib.py:228-231]
def calcDistanceFromFrequency(radarConfig:radarConfigData, frequency:float) -> float:
    rtTime = frequency / radarConfig.freq_change_speed       # how long neede the signal
    distance = (rtTime /2) * SPEED_OF_LIGHT
    return distance


# --- findTargetBin  [was radarLib.py:239-328]
def findTargetBin(dataPath:str, scanSeconds:float=120.0, minDistance:float=0.10, maxDistance:float=1.0,
                   searchBand:list=None, altCalculation:bool=False, window:np.ndarray|str=None) -> dict:
    '''
    automatically find the range bin with the strongest periodic vital-sign activity, instead
    of relying on a manually measured distance. A live target's bin shows periodic phase
    modulation from breathing/heartbeat; static clutter (walls, cage bars) reflects strongly
    but without that periodic modulation - so raw reflection strength alone is not enough.

    Strategy: take a short window of frames, unwrap the phase of every candidate bin over
    that window, bandpass-filter it to searchBand, and pick the bin with the highest energy
    (variance) in that band.

    :param dataPath:        where to find the recording
    :param scanSeconds:     how much data (in seconds) to use for the scan. Needs to be long
                             enough to contain several cycles of the slowest signal in
                             searchBand - heartbeat (a few bpm even in deep torpor) is a more
                             reliable pick for this than breathing, which can be as slow as
                             0.2 cycles/min and would need a much longer scan window to show
                             up reliably.
    :param minDistance:     skip candidate bins closer than this (m) - avoids TX/RX leakage
                             near the sensor, which otherwise tends to dominate raw energy.
    :param maxDistance:     skip candidate bins farther than this (m)
    :param searchBand:      [lowFreq, highFreq] in cycles/min to bandpass each candidate bin
                             before scoring it. Defaults to a wide band covering both torpid
                             and awake heart rate (2-280 bpm) so it works for either dataset.
    :param altCalculation:  True -> use processOneFrameAlt instead of processOneFrame
    :param window:          optional window function for the chirp FFT (same as preProcessRadarData)
    :return:    dict with keys: bin (int), distance (float, m), energy_per_bin (np.ndarray),
                bins (np.ndarray of candidate bin indices, for diagnostic plotting)
    '''
    if searchBand is None:
        searchBand = [2.0, 280.0]   # covers HIB_HEART_BAND through AWAKE_HEART_BAND

    TX_ANTENNA = 0
    RX_ANTENNA = 0

    radarConfig, data, timestamps = importRadarData(dataPath)

    if (window is not None) and (type(window) == str):
        window = get_window(window, radarConfig.num_samples_per_chirp)

    scanFrames = min(int(scanSeconds * radarConfig.frame_rate_Hz), data.shape[0])
    # use a chunk from the middle of the recording rather than the very start, to avoid any
    # startup transient in the sensor/recording
    startFrame = max(0, (data.shape[0] - scanFrames) // 2)

    minBin = max(1, calcFFTbinFromFrequency(radarConfig, calcFrequencyFromDistance(radarConfig, minDistance)))
    maxBin = calcFFTbinFromFrequency(radarConfig, calcFrequencyFromDistance(radarConfig, maxDistance))

    print(f"   [Bin Scan] scanning bins {minBin}-{maxBin} over {scanFrames} frames "
          f"({scanFrames/radarConfig.frame_rate_Hz:.1f}s) starting at frame {startFrame}")

    # collect the complex range-FFT value at each candidate bin, for each scanned frame
    candidateBins = np.arange(minBin, maxBin+1)
    complexByBin = np.zeros((scanFrames, len(candidateBins)), dtype=complex)

    for i, frameNum in enumerate(range(startFrame, startFrame+scanFrames)):
        if len(data.shape) == 5:
            frameRaw = data[frameNum, TX_ANTENNA, RX_ANTENNA].copy()
        else:
            frameRaw = data[frameNum, RX_ANTENNA].copy()

        if altCalculation:
            frameFFT = processOneFrameAlt(frameRaw, mean=True, window=window)
        else:
            frameFFT = processOneFrame(frameRaw, mean=True, window=window)

        complexByBin[i, :] = frameFFT[candidateBins]

    frames_per_min = int(radarConfig.frame_rate_Hz * 60)
    energyPerBin = np.zeros(len(candidateBins))

    for col in range(len(candidateBins)):
        phase = np.unwrap(np.angle(complexByBin[:, col]))
        phaseFiltered = filterData(phase, searchBand[0], searchBand[1], frames_per_min, order=4)
        energyPerBin[col] = np.var(phaseFiltered)

    bestIdx  = np.argmax(energyPerBin)
    bestBin  = int(candidateBins[bestIdx])
    bestFreq = bestBin * radarConfig.fft_step_Hz
    bestDist = calcDistanceFromFrequency(radarConfig, bestFreq)

    print(f"   [Bin Scan] best bin: {bestBin}  (~{bestDist:.3f} m)")

    return {
        "bin": bestBin,
        "distance": bestDist,
        "energy_per_bin": energyPerBin,
        "bins": candidateBins,
    }


# --- saveRadarAngles  [was radarLib.py:418-420]
def saveRadarAngles(radarAngles:np.ndarray, dataPath:str):
    anglesFile = os.path.join(dataPath, ANGLES_NAME)
    np.save(anglesFile, radarAngles)


# --- importRadarPhasedata  [was radarLib.py:428-438]
def importRadarPhasedata(dataPath:str):
    '''
    import preprocessed radar-data and radar-configuration from datapath
    :param dataPath:    where to find files
    :return:            deviceConfig, preprocessed radar-data
    '''
    anglesFile = os.path.join(dataPath, ANGLES_NAME)    
    radarConfig = importRadarConfig(dataPath) # read JSON-File with configuration
    # read radar data
    phaseData = np.load(anglesFile)
    return radarConfig, phaseData


# --- calcDisplacement (definition 1 of 2 in the file, line 441)  [was radarLib.py:441-461]
def calcDisplacement(radarConfig:radarConfigData, phaseData:np.ndarray) -> np.ndarray:
    '''
    calculate the distance from the phase shift between consecutive frames

    the calculated distance is relative to the distance equiv to the selected FFT-bin
    for normal use, only the changes of distance are used -> remove the mean of the data
    
    :param radarconfig:     radar-configuration
    :param phaseData:       Numpy-Array with phase data
    :return:    Numpy-Array with the relative distances (one value less than the phase data)
    '''
    # calculate the phase-change
    phaseDelta = phaseData[:-1].copy()
    phaseDelta = phaseDelta -  phaseData[1:]
    # from phase change calculate the freq difference (to the used FFT-bin=)
    freqDelta = (phaseDelta / (2 *np.pi)) * radarConfig.frame_rate_Hz
    # then calculate the RT-time difference form freq difference
    timeDelta = freqDelta / radarConfig.freq_change_speed
    # and convert to distance in m
    distance  = (timeDelta /2) *SPEED_OF_LIGHT
    return distance


# --- calcDisplacementAlt (definition 1 of 2 in the file, line 464)  [was radarLib.py:464-477]
def calcDisplacementAlt(radarConfig: radarConfigData, phaseData: np.ndarray) -> np.ndarray:
    '''
    calculate the distance direct from phase

    :param radarconfig:     radar-configuration
    :param phaseData:       Numpy-Array with phase data
    :return:    Numpy-Array with the relative distances (one value less than the phase data)
    '''
    phaseFixed = detrend(phaseData)
    meanFreq   = (radarConfig.start_frequency_Hz+radarConfig.end_frequency_Hz) /2
    meanWavelength  = SPEED_OF_LIGHT / meanFreq
    distPhase  = (phaseFixed / (4 * np.pi)) * meanWavelength  # calculate distance from phase

    return distPhase - np.mean(distPhase)


# --- window_spectrum (definition 1 of 2 in the file, line 517)  [was radarLib.py:517-524]
def window_spectrum(x, fs, zp=2.0, lo=2.0, hi=30.0):
    '''Power spectrum of one window in units of its own noise floor (median), band lo..hi bpm.'''
    x = normalize(signal.detrend(x)) * signal.get_window('blackman', len(x))
    n = int(round(len(x) * zp))
    p = np.abs(sp.fft.rfft(x, n=n)) ** 2
    f = sp.fft.rfftfreq(n, 1 / (fs * 60))
    b = (f >= lo) & (f <= hi)
    return f[b], p[b] / np.median(p[b])


# --- collect_window_spectra  [was radarLib.py:527-554]
def collect_window_spectra(rec_dirs, win_s=120.0, overlap=0.0, skip_s=10.0, lo=2.0, hi=30.0, zp=2.0,
                           bin_cols=(1, 2, 3), fname='radar_distance2_alt.npy', prepare=None, fs=None):
    '''
    One spectrum per window (average over the target-bin columns).

    prepare: callable applied once to each FULL column before cutting windows (your heart band-pass,
             the same one calcHeartRateOverTime uses), so filter transients are not repeated per window.
    returns: f (bpm), S (n_windows x n_freq, in noise-floor units), meta (list of dicts: rec, t0_s, t1_s)
    '''
    f, spectra, meta = None, [], []
    for rec in rec_dirs:
        path = os.path.join(rec, fname)
        if not os.path.exists(path):
            continue
        d = np.load(path)
        t = d[:, 0] - d[0, 0]
        fs_rec = fs or 1 / np.median(np.diff(t))
        cols = {c: (prepare(d[:, c]) if prepare else d[:, c]) for c in bin_cols}
        n = int(round(win_s * fs_rec))
        step = max(1, int(round(n * (1 - overlap))))
        for a in range(int(round(skip_s * fs_rec)), len(d) - n + 1, step):
            specs = []
            for c in bin_cols:
                f, s = window_spectrum(cols[c][a:a + n], fs_rec, zp, lo, hi)
                specs.append(s)
            spectra.append(np.mean(specs, axis=0))
            meta.append({"rec": rec, "t0_s": float(t[a]), "t1_s": float(t[a] + win_s)})
    return f, np.array(spectra), meta


# --- calcRate (definition 1 of 2 in the file, line 557)  [was radarLib.py:557-568]
def calcRate(distData:np.ndarray, framerate_Hz:float) -> float:
    distNorm    = normalize(distData )
    distWindowed= distNorm * signal.get_window(window='blackman', Nx=len(distData))
    spectrum    = sp.fft.rfft( distWindowed )
    specAbs     = np.abs( spectrum )
    specAbs[0]  = 0     # suppress DC / near-DC leakage so it can never win the argmax
    specFreq    = sp.fft.rfftfreq(len(distData), 1/(framerate_Hz*60))  # display frequencies in 1/min

    peakIdx = np.argmax(specAbs)
    peakFreq = specFreq[peakIdx]
    
    return peakFreq


# --- calcRate (definition 2 of 2 in the file, line 571)  [was radarLib.py:571-581]
def calcRate(distData: np.ndarray, framerate_Hz: float, zero_pad_factor: float = 2.0) -> float:
    distNorm = normalize(distData)
    distWindowed = distNorm * signal.get_window(window='blackman', Nx=len(distData))
    nfft = int(len(distData) * zero_pad_factor)
    spectrum = sp.fft.rfft(distWindowed, n=nfft)
    specAbs = np.abs(spectrum)
    specAbs[0] = 0
    specFreq = sp.fft.rfftfreq(nfft, 1 / (framerate_Hz * 60))

    peakIdx = np.argmax(specAbs)
    return specFreq[peakIdx]


# --- motion_metrics  [was radarLib.py:868-880]
def motion_metrics(dist, fs, win_s=60.0):
    n = int(win_s * fs)
    rows = []
    for a in range(0, len(dist) - n + 1, n):
        seg = signal.detrend(dist[a:a+n])
        rows.append({
            "t0_s": a / fs,
            "rms": seg.std(),
            "ptp": np.ptp(seg),
            "range_95": np.percentile(seg, 97.5) - np.percentile(seg, 2.5),
            "max_step": np.abs(np.diff(seg)).max(),   # > ~1.2e-3 m: unwrap likely failed
        })
    return rows


# --- window_spectrum (definition 2 of 2 in the file, line 883)  [was radarLib.py:883-889]
def window_spectrum(x, fs, zp=2, lo=2, hi=30):
    x = normalize(signal.detrend(x)) * signal.get_window('blackman', len(x))
    n = int(len(x) * zp)
    p = np.abs(sp.fft.rfft(x, n=n)) ** 2
    f = sp.fft.rfftfreq(n, 1 / (fs * 60))
    b = (f >= lo) & (f <= hi)
    return f[b], p[b] / np.median(p[b])      # units of the window's own noise floor




# ====================================================================================================
# from radarTools.py - no caller in any of the six files
# ====================================================================================================

# --- WEIGHTED_PEAK_FIND  [was radarTools.py:15-15]
WEIGHTED_PEAK_FIND  = False


# --- plotFFTdata  [was radarTools.py:22-48]
def plotFFTdata(data_path: str, name: str, FFT_data: np.ndarray, timebase:float, heart_rate_bpm: float = 60.0):
    FFT_freq = np.arange(len(FFT_data)) / timebase
    fig, ax = plt.subplots(1, 1, figsize=(15,8) )
    if len(FFT_data.shape) ==1:
        ax.plot(FFT_freq, FFT_data, color='black')
    else:
        FFT_mean = np.mean(FFT_data, axis=1)
        FFT_count = FFT_data.shape[0]
        for i in range(FFT_count):
            ax.plot(FFT_freq, FFT_data[:,i], color='0.5')
        ax.plot(FFT_freq, FFT_mean, color='black')

    f_hr = heart_rate_bpm / 60.0
    for h in range(1, 5):
        harmonic_freq = h * f_hr
        ax.axvline(
            x=harmonic_freq,
            color="red",
            linestyle="--",
            alpha=1,
            label=f"Harmonics ({f_hr:.2f} Hz)" if h == 1 else None,
        )
    
    ax.set_title('Spectrum: '+name)
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"fft_plot_{name}.png"), dpi=150)
    plt.close(fig)


# --- plotNeighboringWindows  [was radarTools.py:159-248]
def plotNeighboringWindows(distData: np.ndarray, framerate_Hz: float, win_s: float, name: str,
                           col: int = 1, n_windows: int = 6, start_s: float = 0.0,
                           step_s: float = None, band_bpm: tuple = None,
                           max_plot_bpm: float = None, pad_factor: int = 2,
                           data_path: str = None):
    """
    Plot the spectra and picked peaks of several neighbouring windows from one (already filtered) recording.
    :param distData:     array with time in column 0 and displacement columns (same format as plotDistances)
    :param win_s:        window length in s (e.g. 60 for heart, 300 for breathing)
    :param col:          data column to use (e.g. 1 = bin 16)
    :param n_windows:    number of windows to plot
    :param start_s:      start of the first window in s (e.g. 10 to skip the filter start-up transient)
    :param step_s:       shift between windows; None = win_s (neighbouring, no overlap); smaller = overlap
    :param band_bpm:     (low, high) search band in 1/min; peak is picked inside it only
    :param pad_factor:   zero-padding factor (only interpolates, true resolution is 60/win_s bpm)
    """
    if max_plot_bpm is None:
        max_plot_bpm = MAX_PLOT_BPM
    if step_s is None:
        step_s = win_s

    fs      = framerate_Hz
    win_n   = int(round(win_s * fs))
    step_n  = int(round(step_s * fs))
    i0      = int(round(start_s * fs))
    n_fft   = pad_factor * win_n
    freqs   = sp.fft.rfftfreq(n_fft, 1 / (fs * 60))          # 1/min
    res_bpm = 60 / win_s                                      # true frequency resolution
    plot_m  = freqs <= max_plot_bpm
    band_m  = plot_m if band_bpm is None else (freqs >= band_bpm[0]) & (freqs <= band_bpm[1])
    window  = signal.get_window('blackman', win_n)
    color   = PLOT_COLORS[col % len(PLOT_COLORS)]

    mids, specs, peaks, edge = [], [], [], []
    for k in range(n_windows):
        a = i0 + k * step_n
        b = a + win_n
        if b > distData.shape[0]:
            print(f"only {k} windows fit into the recording")
            break
        seg  = normalize(distData[a:b, col])
        spec = np.abs(sp.fft.rfft(seg * window, n=n_fft))
        spec[0] = 0
        spec = spec / spec[plot_m].max()

        idx = np.flatnonzero(band_m)[np.argmax(spec[band_m])]
        f0  = freqs[idx]
        at_edge = band_bpm is not None and (f0 <= band_bpm[0] + res_bpm or f0 >= band_bpm[1] - res_bpm)

        mids.append(distData[a, 0] - distData[0, 0] + win_s / 2)
        specs.append(spec)
        peaks.append(f0)
        edge.append(at_edge)

    fig, (ax_w, ax_p) = plt.subplots(2, 1, figsize=(15, 9), linewidth=0.8)
    fig.suptitle(f"{name} - {win_s:.0f} s windows, step {step_s:.0f} s")

    # top: stacked spectra, one row per window
    for k, spec in enumerate(specs):
        ax_w.plot(freqs[plot_m], spec[plot_m] * 0.9 + k, color=color)
        pk = np.argmin(np.abs(freqs - peaks[k]))
        ax_w.plot(peaks[k], spec[pk] * 0.9 + k, "o", color=color,
                  mfc="none" if edge[k] else color)
    ax_w.set_yticks(range(len(specs)))
    ax_w.set_yticklabels([f"{m - win_s/2:.0f} s" for m in mids])
    ax_w.set_xlim(0, max_plot_bpm)
    ax_w.set_xlabel("Frequency (BPM)")
    ax_w.set_ylabel("Window start")
    ax_w.set_title("Normalized spectra (hollow marker = peak at band edge)")
    ax_w.grid(True)

    # bottom: picked peak vs time
    ax_p.plot(mids, peaks, "-", color=color, alpha=1)
    for m, f0, e in zip(mids, peaks, edge):
        ax_p.plot(m, f0, "o", color=color, mfc="none" if e else color)
    if band_bpm is not None:
        for lim in band_bpm:
            ax_p.axhline(lim, color="gray", linestyle="--")
    ax_p.set_xlabel("Time (s)")
    ax_p.set_ylabel("Peak frequency (BPM)")
    ax_p.set_title("Picked peak per window")
    ax_p.grid(True)

    plt.tight_layout()
    if data_path is not None:
        plt.savefig(os.path.join(data_path, f"windows_{name}_{distData[0,0]:.0f}.png"), dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return peaks, edge




# ====================================================================================================
# from radarTestDefs.py - no caller in any of the six files (not even in commented-out code)
# ====================================================================================================

# --- radarFilePaths4  [was radarTestDefs.py:28-29]
radarFilePaths4 = ["D:/igel/Radar-Test/recording_2026_02_18_10_17_15/RadarIfxBGT60TR13C_00",
                   "D:/igel/Radar-Test/recording_2026_02_26_11_42_11/RadarIfxBGT60TR13C_00"]


# --- radarDistances4  [was radarTestDefs.py:31-31]
radarDistances4 = [0.49, 0.49] 




# ====================================================================================================
# from radarTestWaves.py - no caller in live code (old hibernating pipeline, superseded by radarTestWavesNew; run_batch_processing_awake duplicates runAwakeAnalysis without the plots)
# ====================================================================================================

# --- calcRatesOverTime  [was radarTestWaves.py:281-327]
def calcRatesOverTime(dataPath:str, plotRange:int):
    """
    calculate the breath- and heartrate for different parts of the distanceData, using the
    hibernating-torpor bands (HIB_BREATH_BAND/HIB_HEART_BAND) for every window.
    :param dataPath:    where to find the data
    :param plotRange:   how much time to use for the calculation
    :return:    a list with breathrates and heartrates
    """
    dataFilename    = getDistFilename(dataPath, USE_ALT_PHASE, USE_ALT_DIST)
    distData        = np.load(dataFilename)
    distCount       = distData.shape[0]
    radarConfig     = importRadarConfig(dataPath)

    rates = []
    rateLen = int(radarConfig.frame_rate_Hz * plotRange)    # how much data to use for calculation
    if distCount < rateLen:
        print(f"   WARNING: recording has {distCount} frames, shorter than one "
              f"{plotRange}s window ({rateLen} frames needed) - skipping, 0 rate windows produced.")
        return rates
    for ofs in range(0, distCount-rateLen, rateLen//2):
        # select part of data to calculate rates
        rateData = distData[ofs:ofs+rateLen,:]
        #grab timestamps
        window_timestamp = distData[ofs, 0]
        displacement = rateData[:, 1]

        window_std = np.std(displacement)
        status = "hibernating"

        # calculate rates from selected data
        breath_heart_rate = calcRates(radarConfig, rateData, HIB_BREATH_BAND, HIB_HEART_BAND)
        # append calculated rates to list
        rates.append({
            "Timestamp_s": window_timestamp,
            "Breathing_Rate_BPM": breath_heart_rate['Breathrate'],
            "Heart_Rate_BPM": breath_heart_rate['Heartrate'],
            "Breathing_Std": breath_heart_rate['BreathStd'],      # Added 
            "Heart_Std": breath_heart_rate['HeartStd'],          
            "Breath_SNR_dB": breath_heart_rate['Breath_SNR_dB'],
            "Heart_SNR_dB": breath_heart_rate['Heart_SNR_dB'],
            "Breath_Prominence": breath_heart_rate['Breath_Prominence'],
            "Heart_Prominence": breath_heart_rate['Heart_Prominence'],
            "Arousal_Flag": status,
            "Displacement_Std": window_std
            })
    # return list of calculated rates
    return rates


# --- runHibernatingAnalysis  [was radarTestWaves.py:373-396]
def runHibernatingAnalysis(data_paths:list = None):
    '''
    run rate-extraction + plotting for hibernating recordings, using HIB_BREATH_BAND/
    HIB_HEART_BAND and PLOT_RANGE_HIB (5-min windows). Writes rates.csv + plots into
    each recording's own folder.
    '''
    if data_paths is None:
        data_paths = radarFilePaths1

    for data_path in data_paths:
        print(f"\nProcessing (hibernating config): {data_path}")
        try:
            rates = calcRatesOverTime(data_path, PLOT_RANGE_HIB)
            if len(rates) == 0:
                print(f"   Skipping {data_path} - no rate windows produced (recording too short).")
                continue
            rates_df = pd.DataFrame.from_dict(rates)
            rates_df["Timestamp_UTC"] = pd.to_datetime(rates_df["Timestamp_s"], unit="s", utc=True)
            rates_df.to_csv(os.path.join(data_path, "rates.csv"), index=False)
            plotDistData(data_path, HIB_BREATH_BAND, HIB_HEART_BAND, PLOT_RANGE_HIB, savePlot=True,
                         title="Hibernating Hedgehog Heart and Respiration Rates2")
        except Exception:
            print("Error processing dir:", data_path)
            traceback.print_exc()


# --- buildComparisonCsv  [was radarTestWaves.py:430-454]
def buildComparisonCsv(data_path:str):
    '''
    merge the hibernating-band (rates.csv) and awake-band (awake_rates.csv) estimates for
    the same recording into one CSV, aligned by nearest timestamp. Both
    candidate heart/breath-rate estimates for any given moment side by side, then use 
    CO2-confirmed arousal timestamps to decide - per window - which estimate is the real one.
    '''
    hib_path   = os.path.join(data_path, "rates.csv")
    awake_path = os.path.join(data_path, "awake_rates.csv")
    if not (os.path.exists(hib_path) and os.path.exists(awake_path)):
        print(f"Skipping comparison csv for {data_path} - need both rates.csv and awake_rates.csv present")
        return

    hib_df   = pd.read_csv(hib_path).add_prefix("Hib_").rename(columns={"Hib_Timestamp_s": "Timestamp_s"})
    awake_df = pd.read_csv(awake_path).add_prefix("Awake_").rename(columns={"Awake_Timestamp_s": "Timestamp_s"})

    hib_df   = hib_df.sort_values("Timestamp_s")
    awake_df = awake_df.sort_values("Timestamp_s")

    # awake windows (30s) are much finer than hibernating windows (5min), so align each
    # awake row to its nearest hibernating window rather than trying to force equal timestamps
    merged = pd.merge_asof(awake_df, hib_df, on="Timestamp_s", direction="nearest")
    merged["Timestamp_UTC"] = pd.to_datetime(merged["Timestamp_s"], unit="s", utc=True)
    merged.to_csv(os.path.join(data_path, "rates_comparison.csv"), index=False)
    print(f"Wrote rates_comparison.csv for {data_path}")


# --- runBothAnalyses  [was radarTestWaves.py:457-475]
def runBothAnalyses(data_paths:list = None):
    '''
    for recordings that may contain both torpor and arousal bouts (e.g. radarFilePaths1),
    run BOTH the hibernating-band and awake-band analyses on the same files, then merge
    the two into rates_comparison.csv per recording. Use  CO2 log afterward to pick
    which set of numbers (Hib_* vs Awake_*) is correct for each time window.
    '''
    if data_paths is None:
        data_paths = radarFilePaths1

    runHibernatingAnalysis(data_paths)
    runAwakeAnalysis(data_paths)

    for data_path in data_paths:
        try:
            buildComparisonCsv(data_path)
        except Exception:
            print("Error building comparison csv for dir:", data_path)
            traceback.print_exc()


# --- run_batch_processing_awake  [was radarTestWaves.py:497-523]
def run_batch_processing_awake(paths: list[str], plotRange: int = 30,
                               breathBand: list = AWAKE_BREATH_BAND,
                               heartBand: list = AWAKE_HEART_BAND,
                               status_label: str = "awake/arousal state"):
    """
    Aroused/awake counterpart to run_batch_processing() above. No separate breath/heart
    window merge is needed here - calcRatesOverTimeAwake already computes both rates
    per window in one pass, so each recording just needs its awake_rates.csv written.
    """
    for data_path in paths:
        try:
            print(f"\nProcessing (aroused): {data_path}")

            awake_rates = calcRatesOverTimeAwake(data_path, plotRange, breathBand, heartBand, status_label)
            if not awake_rates:
                print(f"   Skipping {data_path} - no rate windows produced (recording too short).")
                continue

            awake_df = pd.DataFrame(awake_rates)
            awake_df["Timestamp_UTC"] = pd.to_datetime(awake_df["Timestamp_s"], unit="s", utc=True)
            awake_df.to_csv(os.path.join(data_path, "awake_rates.csv"), index=False)

            print(f"  -> Saved awake_rates.csv ({len(awake_df)} rows)")

        except Exception as e:
            print(f"  [ERROR] Failed processing {data_path}: {e}")
            traceback.print_exc()


# --- commented-out example call for run_batch_processing_awake  [was radarTestWaves.py:527-535]
# if __name__ == "__main__":
#     from radarTestDefs import radarFilePaths1
#     run_batch_processing_awake(
#         radarFilePaths1,
#         plotRange=30,
#         breathBand=AWAKE_BREATH_BAND,
#         heartBand=AWAKE_HEART_BAND,
#         status_label="active"
#     )




# ====================================================================================================
# from radarTestWavesNew.py - no caller in any of the six files
# ====================================================================================================

# --- edgeThresholds  [was radarTestWavesNew.py:36-39]
def edgeThresholds(band, window_s, cells):
    """The two BPM thresholds edgeFlags uses, for printing and checking."""
    cell = 60.0 / window_s
    return band[0] + cells["low"] * cell, band[1] - cells["high"] * cell


# --- trimEdgeRuns  [was radarTestWavesNew.py:42-57]
def trimEdgeRuns(is_edge):
    """is_edge: per-window band-edge flags for ONE recording, in time order.
    Returns True only for the contiguous run of edge windows at the very
    start and the contiguous run at the very end of the recording."""
    is_edge = np.asarray(is_edge, dtype=bool)
    out = np.zeros_like(is_edge)
    if not is_edge.any():
        return out
    if is_edge.all():                      # whole recording is edge -> flag all
        return np.ones_like(is_edge)
    if is_edge[0]:                         # leading run
        out[:np.argmin(is_edge)] = True    # argmin = index of first False
    if is_edge[-1]:                        # trailing run
        rev = is_edge[::-1]
        out[len(out) - np.argmin(rev):] = True
    return out


# --- nearBreathHarmonic  [was radarTestWavesNew.py:126-129]
def nearBreathHarmonic(heart, breath, breath_at_edge, bin_bpm, kmax=4):
    if breath_at_edge or not np.isfinite(breath) or breath <= 0:
        return False
    return any(abs(heart - k * breath) <= bin_bpm for k in range(1, kmax + 1))



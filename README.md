# Radar vital-sign extraction (hedgehog hibernation)

Python pipeline that turns raw BGT60TR13C FMCW radar recordings into per-window
heart and breathing rates, plus quality metrics (SNR, prominence, band-edge flags).
Part of my master's thesis on extracting breathing and heart rates for a hibernating hedgehog.
(Parallelly: comparison with dog and active hedgehogs)

## Pipeline overview

raw radar recording
  -> radarTestPreprocess.py   range FFT, phase at target bin   -> radar_phase*.npy
  -> radarTestPreprocess.py   phase -> displacement            -> radar_distance*.npy
  -> radarTestWavesNew.py     windowing, filtering, rate + SNR -> rates.csv
  -> radarTestWaves.py        awake / dog analysis             -> awake_rates.csv
  -> <build_master_table.py>  merge with CO2/temperature       -> <master_table>

## Requirements
Python <version>; numpy, scipy, pandas, matplotlib (`pip install -r requirements.txt`).

## Data layout
Each recording is a folder containing `config.json`, the raw radar data and timestamps.
Paths are set in `radarTestDefs.py` (`radarFilePaths1` = hibernating, `...2` = awake
station, `...3` = dog). Target distance per recording is `radarDistances*`.

## How to run
1. Set paths and distances in `radarTestDefs.py`.
2. `python radarTestPreprocess.py` - writes phase and distance files into each recording folder.
3. `python radarTestWavesNew.py` - writes `rates.csv` per recording and `batch_vitals_summary.csv`.
4. <awake / dog: how you run it>

## Method
- Target bins 16-18 (one column each); <which one is used>
- Hibernating: heart window 60 s, breathing window 300 s, 50 % overlap
- Awake/aroused: 30 s windows
- Butterworth band-pass (order 4), notch on breathing harmonics 2-4 for the heart signal
- Rate = rFFT peak inside the band, zero-padded; SNR and prominence from harmonics
- Bands (bpm): table with hibernating / aroused / awake / dog

## Output columns
Table explaining `Heart_Rate_BPM`, `Breathing_Rate_BPM`, `*_SNR_dB`, `*_Prominence`,
`*_Edge`, `Timestamp_s`, window lengths ...

## Known limitations
- Peaks near the lower band edge are flagged as possible artifacts (`*_Edge`)
- <N = 1 animal, sensor failure on the second box, ...>

## Project layout
radarLib.py (core processing), radarTools.py (plots), radarTestDefs.py (paths), radarTestWaves and *new ()
radarAttic.py (unused code, kept for reference, not imported)

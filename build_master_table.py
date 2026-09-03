'''
build_master_table.py

Combines:
  - radar rates.csv AND awake_rates.csv per recording (Individual = 00/01 folder suffix,
    from radarFilePaths1)
  - box environmental data (CO2, Temp, Hum)
  - outdoor environmental data (Temp, Humidity, Pressure, Light)

Produces one master DataFrame with:
  - State column ("aroused" / "hibernating"), based on CO2 THRESHOLD-CROSSING INTERVALS,
    not single peak timestamps
  - Heart_Rate_BPM / Breathing_Rate_BPM: the CO2-selected estimate per row - Awake_* band
    during confirmed arousal, Hib_* band otherwise (Hib_*/Awake_* originals are kept too)
  - Intervention_Day flag for days with known handling/disturbance
  - all environmental variables merged in via merge_asof

Ready to feed into the mixedlm model
'''

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 1. Arousal Interval Detection & State Labeling

def findArousalIntervals(env: pd.DataFrame, co2_col: str = "CO2", threshold: float = 2000,
                         min_duration: pd.Timedelta = pd.Timedelta("10min"),
                         pad: pd.Timedelta = pd.Timedelta("0min")) -> pd.DataFrame:
    env = env.sort_values("datetime").reset_index(drop=True)
    above = env[co2_col] > threshold
    blockId = (above != above.shift()).cumsum()

    intervals = []
    for _, group in env[above].groupby(blockId[above]):
        start = group["datetime"].iloc[0] - pad
        end   = group["datetime"].iloc[-1] + pad
        if (end - start) >= min_duration:
            intervals.append({
                "start": start,
                "end": end,
                "duration_min": (end - start).total_seconds() / 60,
                "peak_CO2": group[co2_col].max(),
            })

    return pd.DataFrame(intervals)


def labelState(timestamps: pd.Series, arousal_intervals: pd.DataFrame) -> np.ndarray:
    if len(arousal_intervals) == 0:
        return np.full(len(timestamps), "hibernating", dtype=object)
    idx = pd.IntervalIndex.from_arrays(arousal_intervals["start"], arousal_intervals["end"], closed="both")
    return np.where(idx.get_indexer(timestamps) >= 0, "aroused", "hibernating")


# 2. Load Environmental & Rate Data (with ns timestamp standardization)

def loadOutdoorEnv(path=r"D:/igel/Igel-Außentemp.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(
        df["Datum"] + " " + df["Zeit"], format="%d.%m.%Y %H:%M:%S"
    ).astype("datetime64[ns]")
    df = df.rename(columns={
        "Temperatur[°C]": "Outdoor_Temp_C",
        "rel.Feuchte[%]": "Outdoor_Humidity_pct",
        "Barom.Druck[hPa]": "Outdoor_Pressure_hPa",
        "Beleuch.Stärke[lx]": "Light_Lux",
    })
    return df.sort_values("Timestamp")


def loadBoxEnv(path=r"D:/igel/SD-Card - ohne/H-BOX.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d.%m.%Y %H:%M:%S"
    ).astype("datetime64[ns]")
    df = df.rename(columns={"CO2": "CO2_ppm", "Temp": "Box_Temp_C", "Hum": "Box_Humidity_pct"})
    return df.sort_values("Timestamp")


DEVICE_TO_INDIVIDUAL = {"00": "Hedgehog_A", "01": "Hedgehog_B"}


def parseIdentifiers(data_path: str):
    p = Path(data_path)
    device_suffix = p.name.split("_")[-1]
    individual = DEVICE_TO_INDIVIDUAL.get(device_suffix, "Unknown")
    session = p.parent.name
    return individual, session


def loadRecordingRates(data_path: str, filename: str = "rates.csv", col_prefix: str = "Hib_"):
    path = Path(data_path) / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    individual, session = parseIdentifiers(data_path)
    df["Individual"] = individual
    df["Session"] = session
    df["Recording"] = f"{session}_{individual}"
    
    # Force ns datetime precision
    df["Timestamp"] = (
        pd.to_datetime(df["Timestamp_s"], unit="s", utc=True)
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
    )
    
    rate_cols = ["Breathing_Rate_BPM", "Heart_Rate_BPM", "Breathing_Std", "Heart_Std", "Displacement_Std"]
    df = df.rename(columns={c: col_prefix + c for c in rate_cols if c in df.columns})
    return df


# 3. Master Table Builder

def buildMasterTable(radar_file_paths: list, arousal_threshold: float = 2000,
                     min_arousal_duration: pd.Timedelta = pd.Timedelta("10min"),
                     special_days: list = None, merge_tolerance: pd.Timedelta = pd.Timedelta("15min")):

    # Radar rates (Hib-band)
    hib_dfs = [loadRecordingRates(p, "rates.csv", col_prefix="Hib_") for p in radar_file_paths]
    master = pd.concat([d for d in hib_dfs if d is not None], ignore_index=True)
    master = master.sort_values("Timestamp")

    # Radar rates (Awake-band)
    awake_dfs = [loadRecordingRates(p, "awake_rates.csv", col_prefix="Awake_") for p in radar_file_paths]
    awake_dfs = [d for d in awake_dfs if d is not None]
    if awake_dfs:
        awake_all = pd.concat(awake_dfs, ignore_index=True).sort_values("Timestamp")
        awake_cols = ["Timestamp", "Recording", "Awake_Breathing_Rate_BPM", "Awake_Heart_Rate_BPM",
                      "Awake_Breathing_Std", "Awake_Heart_Std", "Awake_Displacement_Std"]
        awake_cols = [c for c in awake_cols if c in awake_all.columns]
        merged_parts = []
        for recording, group in master.groupby("Recording"):
            awake_sub = awake_all[awake_all["Recording"] == recording][awake_cols]
            if len(awake_sub) == 0:
                merged_parts.append(group)
                continue
            merged_parts.append(pd.merge_asof(group.sort_values("Timestamp"),
                                              awake_sub.drop(columns=["Recording"]).sort_values("Timestamp"),
                                              on="Timestamp", direction="nearest"))
        master = pd.concat(merged_parts, ignore_index=True).sort_values("Timestamp")
    else:
        print("Warning: no awake_rates.csv files found - Awake_* columns will be missing.")

    # Environmental merges
    outdoor = loadOutdoorEnv()
    box     = loadBoxEnv()

    master = pd.merge_asof(master, outdoor[["Timestamp", "Outdoor_Temp_C", "Outdoor_Humidity_pct",
                                             "Outdoor_Pressure_hPa", "Light_Lux"]],
                           on="Timestamp", direction="nearest", tolerance=merge_tolerance)
    master = pd.merge_asof(master, box[["Timestamp", "CO2_ppm", "Box_Temp_C", "Box_Humidity_pct"]],
                           on="Timestamp", direction="nearest", tolerance=merge_tolerance)

    # State classification
    arousal_intervals = findArousalIntervals(box.rename(columns={"Timestamp": "datetime", "CO2_ppm": "CO2"}),
                                             threshold=arousal_threshold, min_duration=min_arousal_duration)
    master["State"] = labelState(master["Timestamp"], arousal_intervals)

    # State-based rate selection
    is_aroused = master["State"] == "aroused"
    has_awake_data = master["Awake_Heart_Rate_BPM"].notna() if "Awake_Heart_Rate_BPM" in master.columns else pd.Series(False, index=master.index)
    master["Missing_Awake_Data"] = is_aroused & ~has_awake_data

    for metric in ["Heart_Rate_BPM", "Breathing_Rate_BPM"]:
        hib_col, awake_col = f"Hib_{metric}", f"Awake_{metric}"
        if awake_col in master.columns:
            conditions = [master["Missing_Awake_Data"], is_aroused & master[awake_col].notna()]
            choices    = [np.nan, master[awake_col]]
            master[metric] = np.select(conditions, choices, default=master[hib_col])
        else:
            master[metric] = np.where(is_aroused, np.nan, master[hib_col])

    # Intervention flag
    if special_days is not None:
        special_days_norm = pd.to_datetime(special_days).normalize()
        master["Intervention_Day"] = master["Timestamp"].dt.normalize().isin(special_days_norm)
    else:
        master["Intervention_Day"] = False

    return master, arousal_intervals


# 4. Execution Entry Point

if __name__ == "__main__":
    from radarTestDefs import radarFilePaths1

    special_days = ["2026-02-20", "2026-02-27", "2026-03-01", "2026-03-10", "2026-03-12"]

    master, arousal_intervals = buildMasterTable(radarFilePaths1, special_days=special_days)

    print(f"Detected {len(arousal_intervals)} arousal bouts:")
    print(arousal_intervals)

    print(f"\nMaster table: {len(master)} rows, states: {master['State'].value_counts().to_dict()}")
    print(f"Intervention-day rows: {master['Intervention_Day'].sum()}")
    print(f"Rows flagged Missing_Awake_Data: {master['Missing_Awake_Data'].sum()}")

    # Export master output
    master.to_csv("master_table.csv", index=False)
    print("Saved master_table.csv")
    
    master_a = master[master["Individual"] == "Hedgehog_A"].copy()
    master_a.to_csv("master_tableA.csv", index=False)

    # Plot summary histograms 
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    metrics = [("Heart_Rate_BPM", "Heart Rate (bpm)"),
        ("Breathing_Rate_BPM", "Breathing Rate (bpm)"),
    ]
    states = ["hibernating", "aroused"]

    # Color map for each (metric, state) combination
    color_map = {
        ("Heart_Rate_BPM", "hibernating"): "#fcb4b4",      # Light red
        ("Heart_Rate_BPM", "aroused"):  "#f02727",          # Dark red
        ("Breathing_Rate_BPM", "hibernating"):  "#b4d9dc", # Light blue
        ("Breathing_Rate_BPM", "aroused"):  "#178d97",          # Darker blue
    }

    for col_idx, (rate_col, label) in enumerate(metrics):
        for row_idx, state in enumerate(states):
            ax = axes[row_idx, col_idx]
            color = color_map[(rate_col, state)]
            sub = master_a[master_a["State"] == state][rate_col].dropna()

            if len(sub):
                ax.hist(
                    sub,
                    bins=30,
                    alpha=0.7,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                )

            ax.set_title(f"{label} — {state.capitalize()} (n={len(sub)})")
            ax.set_xlabel(label)
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("summary_histograms.png", dpi=150) 
    print("Saved summary_histograms.png")

    # Save summary stats table
    rows = []
    for state in ["hibernating", "aroused"]:
        for rate_col in ["Heart_Rate_BPM", "Breathing_Rate_BPM"]:
            sub = master_a[master_a["State"] == state][rate_col].dropna()
            if len(sub):
                rows.append({
                    "State": state, "Metric": rate_col, "N": len(sub),
                    "Median": sub.median(), "Mean": sub.mean(), "SD": sub.std(),
                    "Min": sub.min(), "Max": sub.max()
                })
    summary_df = pd.DataFrame(rows)
    print("\nSummary Statistics:")
    print(summary_df.round(2).to_string(index=False))
    summary_df.to_csv("summary_stats_table.csv", index=False)
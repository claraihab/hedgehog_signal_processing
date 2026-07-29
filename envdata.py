'''
Environmental data plotting
'''
import sys
sys.path.append(r"C:\Users\Clara Guindy\Documents\GitHub\hedgehog_signal_processing")

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load CSV
env = pd.read_csv(r"D:/igel/SD-Card - ohne/H-BOX.csv") 


# Combine Date + Time into datetime
env['datetime'] = pd.to_datetime(env['Date'] + ' ' + env['Time'], format="%d.%m.%Y %H:%M:%S")

# Quick summary 
print(env.describe())
print(env.info())

peaks, _ = find_peaks(env['CO2'], height=5000, distance=100) 
arousal_times = env['datetime'].iloc[peaks]

##add specific line colors for these days: 20.2, 27.2, 1.3, 10.3, 12.3 (5 intervention days)

special_days = [
    "2026-02-20",
    "2026-02-27",
    "2026-03-01",
    "2026-03-10",
    "2026-03-12",
]
line_color = "darkorange"

# Plot CO2, Temp, Hum in 3 stacked subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# CO2 plot
axes[0].plot(env['datetime'], env['CO2'], color='black')
axes[0].set_ylabel('CO2 level [ppm]')

# Temp plot
axes[1].plot(env['datetime'], env['Temp'], color='red')
axes[1].set_ylabel('Temperature [°C]')

# Humidity plot
axes[2].plot(env['datetime'], env['Hum'], color='blue')
axes[2].set_ylabel('Relative Humidity [%]')
axes[2].set_xlabel('Date')


# Loop through detected peaks to add vertical lines 

for peak_time in arousal_times:
  for ax in axes:
    ax.axvline(
        x=peak_time, color="gray", linestyle="-", linewidth=2.5, alpha=0.3
    )

for day_str in special_days:
  day_dt = pd.to_datetime(day_str)
  for ax in axes:
    ax.axvline(
        x=day_dt,
        color=line_color,
        linestyle="--",
        linewidth=2.0,
        alpha=0.85,
    )


axes[2].set_xlim(env['datetime'].min(), env['datetime'].max())


# Adjust layout and show the final plot
fig.tight_layout(pad=3)
 
start_plot_date = pd.to_datetime('2026-02-20')
axes[2].set_xlim(left=start_plot_date)

# Show the final plot
plt.show()

print(arousal_times)


# Define the start and end dates (inclusive of the whole end day)
start_date = "2026-02-20"
end_date = "2026-03-27 10:00:00"

# Filter dataframe using boolean mask
mask = (env["datetime"] >= start_date) & (env["datetime"] <= end_date)
env_filtered = env.loc[mask]

# Summary table for all three environmental variables
stats = env_filtered[["Temp", "CO2", "Hum"]].agg(
    ["mean", "std", "min", "max", "count"]
)
print("\n--- Environmental Summary Table ---")
print(stats.round(2))


# Non-parametric correlation spearman
spearman_corr = env_filtered[['Temp', 'CO2', 'Hum']].corr(method='spearman')
print(spearman_corr)



'''
MACRO - Environmental data plotting
'''

# Load CSV
outside_env = pd.read_csv(r"D:/igel/Igel-Außentemp.csv") 


# Combine Date + Time into datetime
outside_env['datetime'] = pd.to_datetime(outside_env['Datum'] + ' ' + outside_env['Zeit'], format="%d.%m.%Y %H:%M:%S")

# Quick summary 
print(outside_env.describe())
print(outside_env.info())



fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# pressure plot
axes[0].plot(outside_env['datetime'], outside_env['Barom.Druck[hPa]'], color='black')
axes[0].set_ylabel('Air Pressure [hPa]')

# Temp plot
axes[1].plot(outside_env['datetime'], outside_env['Temperatur[°C]'], color='red')
axes[1].set_ylabel('Temperature [°C]')

# Humidity plot
axes[2].plot(outside_env['datetime'], outside_env['rel.Feuchte[%]'], color='blue')
axes[2].set_ylabel('Relative Humidity [%]')

axes[3].plot(outside_env['datetime'], outside_env['Beleuch.Stärke[lx]'], color='orange')
axes[3].set_ylabel('light intensity (lux)')

axes[3].set_xlabel('Date')



axes[3].set_xlim(outside_env['datetime'].min(), env['datetime'].max())


# Adjust layout and show the final plot
fig.tight_layout(pad=3)
 
start_plot_date = pd.to_datetime('2026-02-20')
axes[3].set_xlim(left=start_plot_date)

# Show the final plot
plt.show()


# Filter dataframe using boolean mask
mask_out = (outside_env["datetime"] >= start_date) & (outside_env["datetime"] <= end_date)
outside_env_filtered = outside_env.loc[mask_out]

# Summary table for all three environmental variables
outside_stats = outside_env_filtered[["Temperatur[°C]", "rel.Feuchte[%]", "Barom.Druck[hPa]", "Beleuch.Stärke[lx]"]].agg(
    ["mean", "std", "min", "max", "count"]
)
print("\n--- Environmental Summary Table ---")
print(outside_stats.round(2))


# Non-parametric correlation spearman
spearman_corr_out = outside_env_filtered[["Temperatur[°C]", "rel.Feuchte[%]", "Barom.Druck[hPa]", "Beleuch.Stärke[lx]"]].corr(method='spearman')
print(spearman_corr_out)
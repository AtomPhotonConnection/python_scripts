"""
Used for analysing the data produced by the Thorlabs "Optical Parameter Monitor" software.
Set the file_path to the location that the csv data is saved to, and the script will 
automatically find the data table, read it in, and produce a plot of the power fluctuations
over time. It will also calculate the average fluctuation size as a percentage of the local
mean.
"""


import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# 1. Read the file to find where the data table begins
file_path = r"C:\Users\LabUser\Documents\1 Matt K\data\2026-06 Power stability measurements\new_fibre\18_deg_test_2.csv"  # Replace with your actual file path
base_path, _ = os.path.splitext(file_path)
save_path = base_path + "_plot.png"


# Read the file as lines to find the row index of the header
with open(file_path, 'r') as f:
    lines = f.readlines()

skip_rows = 0
for i, line in enumerate(lines):
    if "Samples" in line and "Power (W)" in line:
        skip_rows = i
        break

# 2. Load the data table, splitting on whitespace/tabs
df = pd.read_csv(
    file_path, 
    skiprows=skip_rows, 
    sep=r',', 
    engine='python'
)
print(f"Data head:\n{df.head()}")


# 3. Clean up column names and convert power to mW to match your plot
# Columns are Samples  Date (MM/dd/yyyy)  Time of day (hh:mm:ss)   Power (W)  Unnamed: 4
df = df.rename(columns={df.columns[3]: 'power_W'})
df = df.rename(columns={df.columns[2]: 'time'})
df = df.rename(columns={df.columns[1]: 'date'})
print(f"Renamed columns:\n{df.columns}")
df['power_mW'] = df['power_W'] * 1000  # Convert to mW for the analysis script
df['power_mW'] = pd.to_numeric(df['power_mW'], errors='coerce')

combined_dt = df["date"].str.strip() + ' ' + df["time"].str.strip()
df['timestamp'] = pd.to_datetime(combined_dt, format='%m/%d/%Y %H:%M:%S.%f')

power = df['power_mW'].to_numpy(dtype=float)

# Find indices of peaks and troughs
# Adjust distance/prominence based on your exact sampling rate
peaks, _ = find_peaks(power, distance=15, prominence=0.01)
troughs, _ = find_peaks(-power, distance=15, prominence=0.01)

# Ensure we pair them up correctly (e.g., matching length)
min_cycles = min(len(peaks), len(troughs))
peaks = peaks[:min_cycles]
troughs = troughs[:min_cycles]

# Calculate local amplitudes and means
p_peaks = power[peaks]
p_troughs = power[troughs]

amplitudes = np.abs(p_peaks - p_troughs) / 2
local_means = (p_peaks + p_troughs) / 2
mean_timestamps = df['timestamp'].iloc[peaks]

# Relative fluctuation per cycle
relative_fluctuations = amplitudes / local_means

# Average across the entire run
avg_relative_fluctuation = np.mean(relative_fluctuations)

print(f"Average Fluctuation Size: {avg_relative_fluctuation * 100:.2f}% of the local mean")


# 5. Plotting (Styled to match your original screenshot)
fig, ax = plt.subplots(figsize=(11, 8))

# Match the light-gray background color from your software
fig.patch.set_facecolor('#f4f4f4')
ax.set_facecolor('#f4f4f4')


# Plot the fluctuation curve in the same dark red color
ax.plot(df['timestamp'], df['power_mW'], color='#b10000', linewidth=1.5)

ax.plot(mean_timestamps, local_means, color='#0055ff', linestyle='--', linewidth=2, label='Local Mean (Oscillation Center)')

ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # Limit number of ticks on X-axis for clarity
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format X-axis as HH:MM:SS on top and MM/DD/YY below

avg_pct = np.mean(relative_fluctuations) * 100
info_text = f"Avg Fluctuation Size:\n{avg_pct:.2f}% of local mean"

# Places text using relative axes coordinates (0,0 is bottom-left, 1,1 is top-right)
ax.text(
    0.95, 0.95, info_text,
    transform=ax.transAxes,
    fontsize=11,
    fontweight='bold',
    color='#333333',
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#999999', alpha=0.8)
)


# Formatting the axes and labels
ax.set_ylabel('Power (mW)', fontsize=10)
ax.set_xlabel('Time', fontsize=10)

# Add horizontal-only gridlines like the original plot
ax.grid(axis='y', color='#999999', linestyle='-', linewidth=0.6)
ax.grid(False,axis='x')  # Turn off vertical grid lines

# Format the X-axis ticks to show "HH:MM:SS" on top and "DD/MM/YY" underneath
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S\n%d/%m/%y'))
ax.tick_params(axis='both', labelsize=9)

# Adjust margins to look clean
plt.tight_layout()

plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
print(f"Plot saved to: {save_path}")

# Display the plot window
plt.show()
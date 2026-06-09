import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# ==========================================
# 1. Configuration
# ==========================================
# Adjust these based on your experimental parameters
fs = 50e6  # Sample rate in Hz (e.g., 50 MHz to match the APD110A bandwidth)
fc = 200e3 # Cutoff frequency in Hz 
order = 5  # 5th order (matches Thorlabs EF504 specifications)

# ==========================================
# 2. Data Loading
# ==========================================
# To test with your actual collected data, uncomment the pandas lines below 
# and update the column names to match your file:
# 
PATH_TO_DATA = r"C:\pulse_shaping_data\2026-03-12\16-49-51_700ns_0.43ms_channel2_delay_RABI(15,15,15,15)_STIRAP_low_intensity\no_pulses\shot002\iteration_1_data.csv"
df = pd.read_csv(PATH_TO_DATA)
t = df['Time (s)'].values
sig = df['Channel 3 Voltage (V)'].values

# --- Synthetic Data Generation (for testing out of the box) ---
# t = np.arange(0, 50e-6, 1/fs)  # 50 us time window
# clean_sig = np.zeros_like(t)
# clean_sig[t > 10e-6] = 1.0     # Ideal step edge representing a fast atom pulse

# # Add Gaussian noise to simulate the 50MHz broadband photodiode noise
# noise_amplitude = 0.3
# sig = clean_sig + np.random.normal(0, noise_amplitude, len(t))
# # --------------------------------------------------------------

# ==========================================
# 3. Filter Design
# ==========================================
# Normalize frequency for digital filter design (Nyquist = fs / 2)
nyq = 0.5 * fs
normal_cutoff = fc / nyq

sos_ellip = signal.ellip(order, rp=1, rs=40, Wn=normal_cutoff, btype='low', analog=False, output='sos')
zi_ellip = signal.sosfilt_zi(sos_ellip)
filtered_ellip, _ = signal.sosfilt(sos_ellip, sig, zi=zi_ellip*sig[0])

sos_bessel = signal.bessel(order, normal_cutoff, btype='low', analog=False, output='sos')
zi_bessel = signal.sosfilt_zi(sos_bessel)
filtered_bessel, _ = signal.sosfilt(sos_bessel, sig, zi=zi_bessel*sig[0])

sos_butter = signal.butter(1, normal_cutoff, btype='low', output='sos') # 1st order Butterworth acts as a simple RC filter for comparison
zi_butter = signal.sosfilt_zi(sos_butter)
filtered_butter, _ = signal.sosfilt(sos_butter, sig, zi=zi_butter*sig[0])

# ==========================================
# 5. Plotting
# ==========================================
plt.figure(figsize=(12, 7))

# Plot raw signal
plt.plot(t * 1e6, sig, label='Raw APD Signal', color='lightgray', alpha=0.8, linewidth=1)

# Plot filtered signals
plt.plot(t * 1e6, filtered_ellip, label='Elliptic Filter (Notice the ringing)', color='red', linewidth=2)
plt.plot(t * 1e6, filtered_bessel, label='Bessel Filter (Smooth edge)', color='blue', linewidth=2)
plt.plot(t * 1e6, filtered_butter, label='1st Order Butterworth (Simple RC)', color='green', linewidth=2)

# Plot formatting
plt.title('Hardware Filter Simulation: Elliptic vs Bessel on a Fast Rising Edge', fontsize=14)
plt.xlabel('Time (µs)', fontsize=12)
plt.ylabel('Amplitude (V)', fontsize=12)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Display the plot
plt.show()
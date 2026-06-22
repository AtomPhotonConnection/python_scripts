#import csv
import argparse
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from typing import Optional

# --- Configuration ---
DEFAULT_DATE = datetime.now().strftime("%Y-%m-%d")
DEFAULT_TIME = None
INPUT_FILENAME = "laser_power_log.csv"
PLOT_FILENAME = "laser_power_plot.png"
STATS_FILENAME = "laser_power_summary.txt"
DETAILS_FILENAME = "details.txt"


def find_latest_run_time(run_date: str) -> Optional[str]:
    data_root = Path(__file__).resolve().parent / "data" / run_date
    if not data_root.exists():
        return None

    time_dirs = [
        entry.name
        for entry in data_root.iterdir()
        if entry.is_dir() and re.fullmatch(r"\d{2}-\d{2}-\d{2}", entry.name)
    ]
    if not time_dirs:
        return None

    return max(time_dirs)

def analyze_laser_data(run_date: str, run_time: Optional[str] = None):
    if run_time is None:
        run_time = find_latest_run_time(run_date)
        if run_time is None:
            print(f"Error: No run folders found for date '{run_date}'.")
            return

    run_dir = Path(__file__).resolve().parent / "data" / run_date / run_time
    input_path = run_dir / INPUT_FILENAME
    plot_path = run_dir / PLOT_FILENAME
    stats_path = run_dir / STATS_FILENAME
    details_path = run_dir / DETAILS_FILENAME

    # Check if the data file exists
    if not input_path.exists():
        print(f"Error: '{input_path}' not found. Please run the collection script first or check the date/time values.")
        return

    print(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path)

    if df.empty:
        print("Error: The CSV file is empty.")
        return

    # --- 1. Calculate Summary Statistics ---
    mean_power = df["Power (W)"].mean()
    std_power = df["Power (W)"].std()
    max_power = df["Power (W)"].max()
    min_power = df["Power (W)"].min()
    
    # Calculate actual duration based on the data points
    total_duration_hours = df["Elapsed Time (s)"].iloc[-1] / 3600 if len(df) > 1 else 0

    # Format the summary statistics
    summary_text = (
        f"======================================\n"
        f"    LASER POWER SUMMARY STATISTICS    \n"
        f"======================================\n"
        f"Analysis Time:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Total Data Points:  {len(df)}\n"
        f"Total Duration:     {total_duration_hours:.2f} hours\n"
        f"--------------------------------------\n"
        f"Mean Power:         {mean_power:.6e} W\n"
        f"Std Deviation:      {std_power:.6e} W\n"
        f"Std Dev %:          {(std_power / mean_power * 100) if mean_power != 0 else np.inf:.2f}%\n"
        f"Max Power:          {max_power:.6e} W\n"
        f"Min Power:          {min_power:.6e} W\n"
        f"======================================\n"
    )

    # Print to console
    print("\n" + summary_text)

    # Save statistics to a text file
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Summary statistics successfully saved to: {stats_path}")

    # Create a blank details file only if it does not already exist
    if not details_path.exists():
        details_path.touch()
        print(f"Created blank details file: {details_path}")
    else:
        print(f"Details file already exists, leaving unchanged: {details_path}")

    # --- 2. Generate and Save the Plot ---
    print("Generating power stability plot...")
    plt.figure(figsize=(10, 6))
    
    # Convert elapsed seconds to hours for a cleaner X-axis layout
    elapsed_hours = df["Elapsed Time (s)"] / 3600
    
    # Plot the raw power data
    plt.plot(elapsed_hours, df["Power (W)"], label="Measured Power", color="#1f77b4", linewidth=1.5)
    
    # Add a horizontal dashed line representing the mean power
    plt.axhline(mean_power, color="red", linestyle="--", alpha=0.8, 
                label=f"Mean Power ({mean_power:.3e} W)")
    
    # Add shaded region for ±5% if variations exist
    if mean_power != 0:
        pct_band = mean_power * 0.05
        plt.fill_between(elapsed_hours, mean_power - pct_band, mean_power + pct_band,
                         color="green", alpha=0.1, label="±5% Band")

    # Add dashed lines indicating standard deviation
    if std_power > 0:
        plt.axhline(mean_power + std_power, color="red", linestyle="--", alpha=0.8,
                    label="+1 Std Dev")
        plt.axhline(mean_power - std_power, color="red", linestyle="--", alpha=0.8,
                    label="-1 Std Dev")

    # Styling the plot
    plt.title(f"Laser Power Stability Over Time ({run_date} {run_time})", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Elapsed Time (hours)", fontsize=12)
    plt.ylabel("Optical Power (Watts)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best", framealpha=0.9)
    
    # Ensure labels and title fit nicely inside the image boundaries
    plt.tight_layout()
    
    # Save the plot high resolution (300 DPI)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plot successfully saved to: {plot_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze laser power measurements from a date/time folder.")
    parser.add_argument("--date", default=DEFAULT_DATE, help="Run date folder in YYYY-MM-DD format.")
    parser.add_argument(
        "--time",
        default=DEFAULT_TIME,
        help="Run time folder in HH-MM-SS format. Defaults to the latest folder for the selected date.",
    )
    args = parser.parse_args()
    analyze_laser_data(args.date, args.time)
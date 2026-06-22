import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from io import StringIO

# --- Configuration ---
DEFAULT_DATE = datetime.now().strftime("%Y-%m-%d")
DEFAULT_TIME = "16-00-00"#datetime.now().strftime("%H-%M-%S")
INPUT_FILENAME = "environment_log.txt"
PLOT_FILENAME = "environment_stability_plot.png"
STATS_FILENAME = "environment_summary.txt"
DETAILS_FILENAME = "details.txt"

def analyze_environment_data(run_date: str, run_time: str):
    run_dir = Path(__file__).resolve().parent / "data" / run_date / run_time
    input_path = run_dir / INPUT_FILENAME
    plot_path = run_dir / PLOT_FILENAME
    stats_path = run_dir / STATS_FILENAME
    details_path = run_dir / DETAILS_FILENAME

    try:
        run_end_time = datetime.strptime(f"{run_date} {run_time}", "%Y-%m-%d %H-%M-%S")
    except ValueError:
        print("Error: date/time arguments must match YYYY-MM-DD and HH-MM-SS formats.")
        return

    # Check if the data file exists
    if not input_path.exists():
        print(f"Error: '{input_path}' not found. Please place your log file there or check the date/time values.")
        return

    print(f"Reading data from {input_path}...")
    
    # Read the entire file content as a string
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_content = f.read()

    if not raw_content.strip():
        print(f"Error: The log file '{input_path}' is completely empty (0 bytes).")
        return

    # Dynamic Header Locator: Scan for known columns to find where the table starts
    header_markers = ["Temperature\\Humidity Graph", "Celsius(°C)", "Humidity(%rh)", "Time,Celsius"]
    start_idx = -1
    for marker in header_markers:
        start_idx = raw_content.find(marker)
        if start_idx != -1:
            break

    # Extract table content
    if start_idx == -1:
        print("Warning: Could not locate standard column headers. Attempting to parse from line 1...")
        csv_content = raw_content
    else:
        csv_content = raw_content[start_idx:]

    # Clean up any potential copy-paste text injections or artifact strings
    csv_content = csv_content.replace("", "").replace("", "")

    # Load data into DataFrame
    try:
        df = pd.read_csv(StringIO(csv_content.strip()))
    except Exception as e:
        print(f"Error parsing CSV data: {e}")
        return

    # Debug check if DataFrame is empty
    if df.empty:
        print("\nError: The data section of the file parsed into an empty table.")
        print("--- DEBUG: First 300 characters found in data section ---")
        print(csv_content[:300])
        print("---------------------------------------------------------")
        return

    # --- 1. Data Processing ---
    # Parse the 'Time' column into datetime objects
    df["Time"] = pd.to_datetime(df["Time"], format="%m/%d/%Y %H:%M:%S")
    
    # Calculate elapsed time in hours relative to the first log point
    start_time = df["Time"].iloc[0]
    df["Elapsed Time (s)"] = (df["Time"] - start_time).dt.total_seconds()
    elapsed_hours = df["Elapsed Time (s)"] / 3600
    total_duration_hours = elapsed_hours.iloc[-1] if len(df) > 1 else 0
    inferred_run_start = run_end_time - timedelta(hours=total_duration_hours)

    # Calculate Summary Statistics for Temperature
    mean_temp = df["Celsius(°C)"].mean()
    std_temp = df["Celsius(°C)"].std()
    max_temp = df["Celsius(°C)"].max()
    min_temp = df["Celsius(°C)"].min()

    # Calculate Summary Statistics for Humidity
    mean_hum = df["Humidity(%rh)"].mean()
    std_hum = df["Humidity(%rh)"].std()
    max_hum = df["Humidity(%rh)"].max()
    min_hum = df["Humidity(%rh)"].min()

    # Format the summary statistics text
    summary_text = (
        f"======================================\n"
        f"    ENVIRONMENTAL LOG SUMMARY         \n"
        f"======================================\n"
        f"Analysis Time:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Total Data Points:  {len(df)}\n"
        f"Run Start Time:     {inferred_run_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Run End Time:       {run_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Total Duration:     {total_duration_hours:.2f} hours\n"
        f"--------------------------------------\n"
        f"TEMPERATURE STATISTICS\n"
        f"Mean Temp:          {mean_temp:.2f} °C\n"
        f"Std Deviation:      {std_temp:.2f} °C\n"
        f"Max Temp:           {max_temp:.2f} °C\n"
        f"Min Temp:           {min_temp:.2f} °C\n"
        f"--------------------------------------\n"
        f"HUMIDITY STATISTICS\n"
        f"Mean Humidity:      {mean_hum:.2f} %rh\n"
        f"Std Deviation:      {std_hum:.2f} %rh\n"
        f"Max Humidity:       {max_hum:.2f} %rh\n"
        f"Min Humidity:       {min_hum:.2f} %rh\n"
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

    # --- 2. Generate and Save the Plot ---
    print("Generating temperature and humidity stability plot...")
    
    # Set up 2 stacked subplots sharing the same X axis (Time)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    plot_x_max = total_duration_hours if total_duration_hours > 0 else 1

    # Top Plot: Temperature
    ax1.plot(elapsed_hours, df["Celsius(°C)"], label="Measured Temp", color="#d62728", linewidth=1.5)
    ax1.plot([0, plot_x_max], [mean_temp, mean_temp], color="black", linestyle="--", alpha=0.7, label=f"Mean ({mean_temp:.2f} °C)")
    if std_temp > 0:
        ax1.plot([0, plot_x_max], [mean_temp + std_temp, mean_temp + std_temp], color="gray", linestyle=":", alpha=0.6, label="+1 Std Dev")
        ax1.plot([0, plot_x_max], [mean_temp - std_temp, mean_temp - std_temp], color="gray", linestyle=":", alpha=0.6, label="-1 Std Dev")
    ax1.set_title(f"Environmental Stability Over Time ({run_date} {run_time})", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Temperature (°C)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Bottom Plot: Humidity
    ax2.plot(elapsed_hours, df["Humidity(%rh)"], label="Measured Humidity", color="#1f77b4", linewidth=1.5)
    ax2.plot([0, plot_x_max], [mean_hum, mean_hum], color="black", linestyle="--", alpha=0.7, label=f"Mean ({mean_hum:.2f} %rh)")
    if std_hum > 0:
        ax2.plot([0, plot_x_max], [mean_hum + std_hum, mean_hum + std_hum], color="gray", linestyle=":", alpha=0.6, label="+1 Std Dev")
        ax2.plot([0, plot_x_max], [mean_hum - std_hum, mean_hum - std_hum], color="gray", linestyle=":", alpha=0.6, label="-1 Std Dev")
    ax2.set_xlabel("Elapsed Time (hours)", fontsize=11)
    ax2.set_ylabel("Humidity (%rh)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9)

    ax2.set_xlim(0, plot_x_max)

    top_axis = ax2.twiny()
    top_axis.set_xlim(ax2.get_xlim())
    top_axis.set_xlabel("Actual Date / Time")

    if total_duration_hours <= 6:
        tick_step = 1
    elif total_duration_hours <= 24:
        tick_step = 4
    elif total_duration_hours <= 72:
        tick_step = 12
    else:
        tick_step = 24

    top_ticks = [0]
    next_tick = tick_step
    while next_tick < total_duration_hours:
        top_ticks.append(next_tick)
        next_tick += tick_step
    if total_duration_hours > 0 and top_ticks[-1] != total_duration_hours:
        top_ticks.append(total_duration_hours)

    top_axis.set_xticks(top_ticks)
    top_axis.set_xticklabels(
        [(inferred_run_start + timedelta(hours=tick)).strftime("%Y-%m-%d\n%H:%M") for tick in top_ticks]
    )

    # Adjust layout and save
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.90, hspace=0.20)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plot successfully saved to: {plot_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze environmental log data from a date/time folder.")
    parser.add_argument("--date", default=DEFAULT_DATE, help="Run date folder in YYYY-MM-DD format.")
    parser.add_argument("--time", default=DEFAULT_TIME, help="Run time folder in HH-MM-SS format.")
    args = parser.parse_args()
    analyze_environment_data(args.date, args.time)
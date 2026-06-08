import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO
from PIL import Image, ImageDraw, ImageFont

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
    plot_x_max = total_duration_hours if total_duration_hours > 0 else 1

    width, height = 1600, 1100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def load_font(size: int, bold: bool = False):
        candidates = [
            r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    title_font = load_font(24, bold=True)
    axis_font = load_font(18)
    label_font = load_font(16)
    small_font = load_font(13)
    legend_font = load_font(15)

    def text_size(text: str, font):
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def format_tick_label(hours_value: float):
        return (inferred_run_start + timedelta(hours=hours_value)).strftime("%Y-%m-%d\n%H:%M")

    def choose_ticks(duration_hours: float):
        if duration_hours <= 6:
            step = 1
        elif duration_hours <= 24:
            step = 4
        elif duration_hours <= 72:
            step = 12
        else:
            step = 24

        ticks = [0.0]
        current = float(step)
        while current < duration_hours:
            ticks.append(current)
            current += step
        return ticks

    def draw_rotated_label(position, text, font, fill):
        label_w, label_h = text_size(text, font)
        label_image = Image.new("RGBA", (label_w + 8, label_h + 8), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label_image)
        label_draw.text((4, 4), text, font=font, fill=fill)
        rotated = label_image.rotate(90, expand=True)
        image.paste(rotated, position, rotated)

    def draw_panel(x0, y0, x1, y1, y_values, line_color, mean_value, std_value, ylabel, legend_label):
        panel_width = x1 - x0
        panel_height = y1 - y0
        y_min = min(min(y_values), mean_value - (std_value * 1.5 if std_value > 0 else 0.5))
        y_max = max(max(y_values), mean_value + (std_value * 1.5 if std_value > 0 else 0.5))
        y_padding = max((y_max - y_min) * 0.08, 0.25)
        y_min -= y_padding
        y_max += y_padding

        def x_to_px(value):
            return x0 + (value / plot_x_max) * panel_width

        def y_to_px(value):
            return y1 - ((value - y_min) / (y_max - y_min)) * panel_height

        draw.rectangle([x0, y0, x1, y1], outline="#222222", width=2)

        y_ticks = 5
        for index in range(y_ticks + 1):
            tick_value = y_min + (y_max - y_min) * index / y_ticks
            tick_y = y_to_px(tick_value)
            draw.line([(x0, tick_y), (x1, tick_y)], fill="#d9d9d9", width=1)
            tick_label = f"{tick_value:.1f}"
            label_w, label_h = text_size(tick_label, small_font)
            draw.text((x0 - label_w - 12, tick_y - label_h / 2), tick_label, font=small_font, fill="#222222")

        x_ticks = choose_ticks(plot_x_max)
        for tick_value in x_ticks:
            tick_x = x_to_px(tick_value)
            draw.line([(tick_x, y0), (tick_x, y1)], fill="#ededed", width=1)

        points = list(zip((x_to_px(value) for value in elapsed_hours), (y_to_px(value) for value in y_values)))
        if len(points) > 1:
            draw.line(points, fill=line_color, width=3)

        mean_y = y_to_px(mean_value)
        draw.line([(x0, mean_y), (x1, mean_y)], fill="#111111", width=2)
        if std_value > 0:
            draw.line([(x0, y_to_px(mean_value + std_value)), (x1, y_to_px(mean_value + std_value))], fill="#777777", width=2)
            draw.line([(x0, y_to_px(mean_value - std_value)), (x1, y_to_px(mean_value - std_value))], fill="#777777", width=2)

        title_w, title_h = text_size(legend_label, axis_font)
        draw.text((x0, y0 - title_h - 10), legend_label, font=axis_font, fill="#111111")
        draw_rotated_label((20, int(y0 + panel_height / 2 - 70)), ylabel, label_font, "#111111")

        legend_items = [(line_color, "Measured"), ("#111111", f"Mean ({mean_value:.2f})")]
        if std_value > 0:
            legend_items.extend([("#777777", "+/- 1 Std Dev")])
        legend_x = x1 - 270
        legend_y = y0 + 20
        for color, label in legend_items:
            draw.line([(legend_x, legend_y + 10), (legend_x + 30, legend_y + 10)], fill=color, width=3)
            draw.text((legend_x + 40, legend_y), label, font=legend_font, fill="#111111")
            legend_y += 28

        return x_to_px, y_to_px, x_ticks

    title_text = f"Environmental Stability Over Time ({run_date} {run_time})"
    title_w, title_h = text_size(title_text, title_font)
    draw.text(((width - title_w) / 2, 24), title_text, font=title_font, fill="#111111")

    top_x0, top_y0, top_x1, top_y1 = 120, 90, 1540, 450
    bottom_x0, bottom_y0, bottom_x1, bottom_y1 = 120, 600, 1540, 960

    draw_panel(top_x0, top_y0, top_x1, top_y1, df["Celsius(°C)"].tolist(), "#d62728", mean_temp, std_temp, "Temperature (°C)", "Temperature")
    x_to_px, _, bottom_ticks = draw_panel(bottom_x0, bottom_y0, bottom_x1, bottom_y1, df["Humidity(%rh)"].tolist(), "#1f77b4", mean_hum, std_hum, "Humidity (%rh)", "Humidity")

    bottom_axis_y = bottom_y1 + 24
    draw.line([(bottom_x0, bottom_y1), (bottom_x1, bottom_y1)], fill="#222222", width=2)
    bottom_label = "Elapsed Time (hours)"
    bottom_label_w, bottom_label_h = text_size(bottom_label, axis_font)
    draw.text(((bottom_x0 + bottom_x1 - bottom_label_w) / 2, bottom_axis_y + 26), bottom_label, font=axis_font, fill="#111111")

    actual_axis_label = "Actual Date / Time"
    actual_axis_label_w, _ = text_size(actual_axis_label, axis_font)
    draw.text(((bottom_x0 + bottom_x1 - actual_axis_label_w) / 2, bottom_y0 - 56), actual_axis_label, font=axis_font, fill="#111111")
    for tick_value in bottom_ticks:
        label = format_tick_label(tick_value)
        tick_x = x_to_px(tick_value)
        label_w, label_h = text_size(label, small_font)
        draw.multiline_text((tick_x - label_w / 2, bottom_y0 - label_h - 16), label, font=small_font, fill="#111111", align="center", spacing=2)
        draw.line([(tick_x, bottom_y0 - 6), (tick_x, bottom_y0)], fill="#222222", width=2)
        hour_label = f"{tick_value:.0f}" if tick_value.is_integer() else f"{tick_value:.1f}"
        hour_label_w, _ = text_size(hour_label, small_font)
        draw.text((tick_x - hour_label_w / 2, bottom_y1 + 8), hour_label, font=small_font, fill="#111111")

    image.save(plot_path)
    
    print(f"Plot successfully saved to: {plot_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze environmental log data from a date/time folder.")
    parser.add_argument("--date", default=DEFAULT_DATE, help="Run date folder in YYYY-MM-DD format.")
    parser.add_argument("--time", default=DEFAULT_TIME, help="Run time folder in HH-MM-SS format.")
    args = parser.parse_args()
    analyze_environment_data(args.date, args.time)
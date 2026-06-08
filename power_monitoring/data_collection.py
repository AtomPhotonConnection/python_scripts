import pyvisa
import time
import csv
from pathlib import Path
from datetime import datetime
from typing import cast, Optional
from pyvisa.resources import MessageBasedResource

# --- Configuration ---
DURATION_HOURS = 2
INTERVAL_SECONDS = 0.5  # Time between measurements
OUTPUT_FILENAME = "laser_power_log.csv"
DURATION_SECONDS = DURATION_HOURS * 3600

def find_pm100a(rm) -> Optional[MessageBasedResource]:
    """Automatically finds the Thorlabs PM100A from available VISA resources."""
    # List all devices connected
    for resource in rm.list_resources():
        try:
            # Open resource with a timeout of 2 seconds
            inst = rm.open_resource(resource, timeout=2000)
            idn = inst.query("*IDN?")
            if "PM100A" in idn:
                print(f"Connected to: {idn.strip()} at {resource}")
                return cast(MessageBasedResource, inst)
            inst.close()
        except Exception:
            # Ignore resources that timeout or don't respond to *IDN?
            pass
    return None

def main():
    rm = pyvisa.ResourceManager()
    pm100a = find_pm100a(rm)
    
    if pm100a is None:
        print("Error: Could not find a Thorlabs PM100A connected. Please check the USB connection.")
        return

    # Configure the instrument to measure optical power
    pm100a.write("CONF:POW")

    run_start = datetime.now()
    output_dir = Path(__file__).resolve().parent / "data" / run_start.strftime("%Y-%m-%d") / run_start.strftime("%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    
    # Open CSV file and write headers
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Elapsed Time (s)", "Power (W)"])
        
        start_time = time.time()
        print(f"Starting measurement for {DURATION_HOURS} hours... (Press Ctrl+C to stop early)")
        
        try:
            # Run the loop until the 3-hour duration is reached
            while time.time() - start_time < DURATION_SECONDS:
                current_time = time.time()
                elapsed = current_time - start_time
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Query the power reading
                power_str = pm100a.query("MEAS:POW?")
                
                try:
                    power = float(power_str.strip())
                except ValueError:
                    power = 0.0
                    print(f"Warning: Could not parse power value: '{power_str}'")
                
                # Log to CSV
                writer.writerow([timestamp_str, f"{elapsed:.2f}", power])
                file.flush() # Ensure data is written to disk immediately to prevent loss in case of a crash
                
                # Optional: Print to console so you can monitor progress
                print(f"[{timestamp_str}] Power: {power:.6e} W")
                
                # Wait until it is time for the next interval
                time.sleep(INTERVAL_SECONDS)
                
        except KeyboardInterrupt:
            print("\nMeasurement manually interrupted by user.")
            
    # Always safely close the connection to the instrument
    pm100a.close()
    print(f"Measurement complete. Data successfully saved to {output_path}")

if __name__ == "__main__":
    main()
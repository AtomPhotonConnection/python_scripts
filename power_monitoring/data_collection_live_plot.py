import csv
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

import pyqtgraph as pg
import pyvisa
from pyqtgraph.Qt import QtCore, QtWidgets
from pyvisa.resources import MessageBasedResource

# --- Configuration ---
DURATION_HOURS = 3
INTERVAL_SECONDS = 0.1  # Time between measurements
OUTPUT_FILENAME = "laser_power_log.csv"
DURATION_SECONDS = DURATION_HOURS * 3600


pg.setConfigOptions(antialias=True)


def find_pm100a(rm) -> Optional[MessageBasedResource]:
    """Automatically finds the Thorlabs PM100A from available VISA resources."""
    for resource in rm.list_resources():
        try:
            inst = rm.open_resource(resource, timeout=2000)
            idn = inst.query("*IDN?")
            if "PM100A" in idn:
                print(f"Connected to: {idn.strip()} at {resource}")
                return cast(MessageBasedResource, inst)
            inst.close()
        except Exception:
            pass
    return None


class LivePowerMonitor:
    def __init__(self):
        self.rm = pyvisa.ResourceManager()
        pm100a = find_pm100a(self.rm)

        if pm100a is None:
            raise RuntimeError(
                "Error: Could not find a Thorlabs PM100A connected. Please check the USB connection."
            )

        self.pm100a = cast(MessageBasedResource, pm100a)

        self.pm100a.write("CONF:POW")

        self.run_start = datetime.now()
        self.output_dir = (
            Path(__file__).resolve().parent
            / "data"
            / self.run_start.strftime("%Y-%m-%d")
            / self.run_start.strftime("%H-%M-%S")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / OUTPUT_FILENAME

        self.output_file = open(self.output_path, mode="w", newline="")
        self.writer = csv.writer(self.output_file)
        self.writer.writerow(["Timestamp", "Elapsed Time (s)", "Power (W)"])

        self.start_time = time.time()
        self.elapsed_history = []
        self.power_history = []
        self.finished = False

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.app = cast(QtWidgets.QApplication, app)

        self.window = pg.PlotWidget(title="Live Laser Power Measurement")
        self.window.resize(1100, 650)

        self.plot = cast(Any, self.window.getPlotItem())
        self.plot.setTitle("Live Laser Power Measurement", size="14pt", bold=True)
        self.plot.setLabel("bottom", "Elapsed Time", units="hours")
        self.plot.setLabel("left", "Optical Power", units="W")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.getViewBox().setMouseEnabled(x=True, y=True)

        self.curve = self.plot.plot([], [], pen=pg.mkPen(color=(31, 119, 180), width=2), name="Measured Power")
        self.curve.setClipToView(True)
        self.curve.setDownsampling(auto=True, method="peak")

        self.window.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.collect_sample)
        self.timer.start(int(INTERVAL_SECONDS * 1000))

        self.app.aboutToQuit.connect(self.cleanup)
        signal.signal(signal.SIGINT, lambda *_: self.app.quit())

    def collect_sample(self):
        if self.finished:
            return

        elapsed = time.time() - self.start_time
        if elapsed >= DURATION_SECONDS:
            self.finish("Measurement complete.")
            return

        elapsed_hours = elapsed / 3600
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            power_str = self.pm100a.query("MEAS:POW?")
        except Exception as exc:
            self.finish(f"Error querying instrument: {exc}")
            return

        try:
            power = float(power_str.strip())
        except ValueError:
            power = 0.0
            print(f"Warning: Could not parse power value: '{power_str}'")

        self.writer.writerow([timestamp_str, f"{elapsed:.2f}", power])
        self.output_file.flush()

        self.elapsed_history.append(elapsed_hours)
        self.power_history.append(power)
        self.curve.setData(self.elapsed_history, self.power_history)

        latest_x = self.elapsed_history[-1]
        x_max = max(latest_x * 1.05, 0.01)
        self.plot.setXRange(0, x_max, padding=0)

        y_min = min(self.power_history)
        y_max = max(self.power_history)
        y_span = y_max - y_min
        y_pad = max(y_span * 0.1, abs(y_max) * 0.05, 1e-12)
        self.plot.setYRange(y_min - y_pad, y_max + y_pad, padding=0)

        print(f"[{timestamp_str}] Power: {power:.6e} W")

    def finish(self, message: str):
        if self.finished:
            return

        self.finished = True
        print(message)
        self.timer.stop()
        self.cleanup()
        self.app.quit()

    def cleanup(self):
        if getattr(self, "output_file", None) and not self.output_file.closed:
            self.output_file.close()

        if getattr(self, "pm100a", None) is not None:
            try:
                self.pm100a.close()
            except Exception:
                pass

        if getattr(self, "rm", None) is not None:
            try:
                self.rm.close()
            except Exception:
                pass

    def run(self):
        print(f"Starting measurement for {DURATION_HOURS} hours... (Close the window to stop early)")
        self.app.exec()
        print(f"Measurement complete. Data successfully saved to {self.output_path}")


def main():
    try:
        monitor = LivePowerMonitor()
    except RuntimeError as exc:
        print(exc)
        return

    monitor.run()


if __name__ == "__main__":
    main()
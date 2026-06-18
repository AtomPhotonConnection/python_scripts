#!/usr/bin/env python3
"""
lowpass_filter_sim.py — Low-Pass Filter Simulation
====================================================
Applies physically-accurate digital equivalents of analogue low-pass
filters to voltage/time oscilloscope data captured at ~2.5 GSa/s.

Filter types simulated
-----------------------
  RC (1st order)      Simple resistor-capacitor circuit:
                        H(s) = ωc / (s + ωc),  ωc = 2π·fc
                        Identical to a 1st-order Butterworth [Ref 3].

  Butterworth         2nd, 4th, and 8th order.  Maximally flat (no ripple)
  (N = 2, 4, 8)       monotone passband; -3 dB exactly at fc [Ref 1].
                        Roll-off: -20·N dB/decade beyond fc.

  Elliptic 5th order  Equiripple in both passband and stopband.  Achieves
                        the steepest possible transition band for a given
                        filter order (Cauer filter) [Ref 4].
                        Default: ≤ 0.1 dB passband ripple, ≥ 60 dB stopband.

Key accuracy decisions
-----------------------
  * All IIR filters output SOS (second-order sections).  Cascading biquads
    avoids the coefficient-cancellation problem that makes high-order
    direct-form IIR numerically unstable when fc << fs/2 [Ref 5].

  * Bilinear transform with pre-warping ensures the analogue -3 dB break
    frequency is preserved exactly in the digital filter [Ref 2, §7.1].

  * Causal mode (default): scipy.signal.sosfilt with DC steady-state
    initial conditions (sosfilt_zi × data[0]) to eliminate the long
    start-up transient that would otherwise appear over ~1/fc seconds.
    This models what a real hardware filter would do [Ref 5].

  * Zero-phase mode (--zero-phase): sosfiltfilt — forward + backward pass.
    Doubles effective order, removes all phase shift.  Useful for
    magnitude-only analysis but physically non-realisable [Ref 6].

Nyquist note
------------
  For fs = 2.5 GSa/s, the Nyquist limit is 1.25 GHz.  Any requested
  cutoff ≥ Nyquist is skipped with a warning.  A 2 GHz LPF cannot be
  represented at this sample rate; you would need fs ≥ 4 GSa/s.

Usage
-----
  python lowpass_filter_sim.py data.csv
  python lowpass_filter_sim.py data.csv --cutoffs 1e6 20e6 200e6 1e9
  python lowpass_filter_sim.py data.csv --filter-type rc --causal
    python lowpass_filter_sim.py data.csv --separate-filter-plots
  python lowpass_filter_sim.py data.csv --time-window 1e-6 --output results

Requirements
------------
  pip install numpy scipy pandas matplotlib

References
----------
[1] Butterworth, S. (1930). On the theory of filter amplifiers.
    Wireless Engineer, 7, 536–541.
    https://www.techrxiv.org/doi/10.36227/techrxiv.17137302

[2] Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal
    Processing (3rd ed.). Prentice Hall.
    [Bilinear transform with pre-warping: §7.1; IIR design: §7.2–7.4]

[3] Williams, A. B., & Taylor, F. J. (2006). Electronic Filter Design
    Handbook (4th ed.). McGraw-Hill.
    [RC prototype §2.1; Butterworth §2.3; Elliptic §2.6]

[4] Zverev, A. I. (1967). Handbook of Filter Synthesis. Wiley.
    [Complete elliptic / Cauer filter pole-zero tables and theory]
    Also: Proakis, J. G., & Manolakis, D. G. (2007). Digital Signal
    Processing (4th ed.). Pearson. [Elliptic IIR: §8.3.4]

[5] SciPy developers (2024). scipy.signal — Signal processing.
    https://docs.scipy.org/doc/scipy/reference/signal.html
    "sosfilt / sosfiltfilt recommended over lfilter for numerical
    stability, especially for high-order filters or small ωc/ωs ratios."

[6] Gustafsson, F. (1996). Determining the initial states in
    forward-backward filtering. IEEE Transactions on Signal Processing,
    44(4), 988–992.  [Initial conditions for sosfiltfilt]
"""

# ── Standard library ─────────────────────────────────────────────────────────
import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from scipy import signal
except ImportError as _e:
    sys.exit(
        f"Missing dependency: {_e}\n"
        "Install with:  pip install numpy scipy pandas matplotlib"
    )

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

MAX_PLOT_SAMPLES: int   = 250_000   # decimate time traces for plotting
BODE_N_POINTS:    int   = 16_384    # frequency resolution for Bode curves
WELCH_NPERSEG:    int   = 65_536    # Welch PSD segment length

# Default cutoff sweep (200 kHz → 1 GHz; 2 GHz skipped if fs = 2.5 GSa/s)
DEFAULT_CUTOFFS: List[float] = [100e3]

ALL_FILTER_NAMES: List[str] = [
    "RC (1st)", "Butter 2nd", "Butter 4th", "Butter 8th", "Elliptic 5th"
]

# Colour / line styles per filter name
STYLES = {
    "Original":     dict(color="#1a1a1a", lw=1.5, ls="-",  alpha=0.80, zorder=10),
    "RC (1st)":     dict(color="#e41a1c", lw=1.2, ls="-",  alpha=1.0,  zorder=6),
    "Butter 2nd":   dict(color="#ff7f00", lw=1.2, ls="-",  alpha=1.0,  zorder=6),
    "Butter 4th":   dict(color="#4daf4a", lw=1.2, ls="-",  alpha=1.0,  zorder=6),
    "Butter 8th":   dict(color="#377eb8", lw=1.2, ls="-",  alpha=1.0,  zorder=6),
    "Elliptic 5th": dict(color="#984ea3", lw=1.2, ls="-",  alpha=1.0,  zorder=6),
}

FILTER_TYPE_MAP = {
    "rc":        ["RC (1st)"],
    "butter2":   ["Butter 2nd"],
    "butter4":   ["Butter 4th"],
    "butter8":   ["Butter 8th"],
    "elliptic5": ["Elliptic 5th"],
    "all":       ALL_FILTER_NAMES,
}


# ═════════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ═════════════════════════════════════════════════════════════════════════════

def fmt_hz(f: float) -> str:
    """Return a human-readable frequency string (Hz / kHz / MHz / GHz)."""
    if f >= 1e9:  return f"{f/1e9:.4g} GHz"
    if f >= 1e6:  return f"{f/1e6:.4g} MHz"
    if f >= 1e3:  return f"{f/1e3:.4g} kHz"
    return f"{f:.4g} Hz"


def fmt_time(t: float) -> str:
    """Return a human-readable time string."""
    a = abs(t)
    if a >= 1e-3:  return f"{t*1e3:.4g} ms"
    if a >= 1e-6:  return f"{t*1e6:.4g} µs"
    if a >= 1e-9:  return f"{t*1e9:.4g} ns"
    return f"{t:.4g} s"


def filter_slug(name: str) -> str:
    """Return a filename-friendly slug for a filter name."""
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    return slug.strip("_")


# ═════════════════════════════════════════════════════════════════════════════
# CSV Loading
# ═════════════════════════════════════════════════════════════════════════════

def load_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load time and voltage arrays from a CSV file.

    Handles common oscilloscope export formats (Tektronix, Keysight/Agilent,
    Rigol, LeCroy) which often have metadata header rows, varying delimiters
    (comma / tab / semicolon), and non-numeric comment lines.

    Column auto-detection tries names like 'time', 't', 'x', 'second' for
    time and 'voltage', 'v', 'ch1', 'amplitude', 'y' for voltage.
    Falls back to first two numeric columns.

    Returns
    -------
    t  : np.ndarray  — time in seconds
    v  : np.ndarray  — voltage in volts
    fs : float       — sample rate in Hz (derived from median Δt)
    """
    path = Path(filepath)
    if not path.exists():
        sys.exit(f"Error: file not found — '{filepath}'")

    # ── Try skipping 0..24 leading header/metadata rows ──────────────────────
    df: Optional[pd.DataFrame] = None
    for skip in range(0, 25):
        try:
            d = pd.read_csv(
                filepath,
                sep=None, engine="python",
                skiprows=skip,
                header=0 if skip == 0 else None,
                on_bad_lines="skip",
                skip_blank_lines=True,
            )
            d_num = d.apply(pd.to_numeric, errors="coerce").dropna()
            if len(d_num) >= 10 and d_num.shape[1] >= 2:
                df = d_num
                if skip:
                    print(f"  ↳ Skipped {skip} header line(s).")
                break
        except Exception:
            continue

    if df is None or len(df) < 10:
        sys.exit(
            "Could not extract numeric data from the CSV.\n"
            "Ensure the file has at least two numeric columns: time (s) and voltage (V)."
        )

    # ── Identify time and voltage columns ────────────────────────────────────
    cols_lower = [str(c).strip().lower() for c in df.columns]
    time_kw = {"time", "t", "x", "sec", "second", "seconds", "time_s", "time(s)"}
    volt_kw = {"voltage", "volt", "v", "y", "amplitude", "ampl",
               "ch1", "ch2", "signal", "value", "voltage_v", "voltage(v)"}

    t_idx = next(
        (i for i, c in enumerate(cols_lower) if any(k in c for k in time_kw)), 0
    )
    v_idx = next(
        (i for i, c in enumerate(cols_lower)
         if any(k in c for k in volt_kw) and i != t_idx),
        1 if t_idx == 0 else 0,
    )

    t = df.iloc[:, t_idx].to_numpy(np.float64)
    v = df.iloc[:, v_idx].to_numpy(np.float64)

    # Sort by time, remove non-finite values
    order = np.argsort(t)
    t, v   = t[order], v[order]
    ok     = np.isfinite(t) & np.isfinite(v)
    t, v   = t[ok], v[ok]

    if len(t) < 10:
        sys.exit("Too few valid samples after cleaning. Check the CSV.")

    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        sys.exit("Error: time column is not monotonically increasing.")
    fs = 1.0 / dt

    print(f"  Samples  : {len(t):,}")
    print(f"  Duration : {fmt_time(t[-1] - t[0])}")
    print(f"  fs       : {fmt_hz(fs)}")
    print(f"  Nyquist  : {fmt_hz(fs / 2)}")
    print(f"  V range  : [{v.min():.4g},  {v.max():.4g}] V")

    if len(t) > 20_000_000:
        print(
            f"\n  ⚠  Large dataset ({len(t):,} samples). "
            "Use --time-window to limit scope and reduce memory usage."
        )

    return t, v, fs


# ═════════════════════════════════════════════════════════════════════════════
# Filter design — all return SOS (second-order sections) arrays
# ═════════════════════════════════════════════════════════════════════════════

def design_rc(fc: float, fs: float) -> np.ndarray:
    """
    Simple RC low-pass filter.

    Analogue prototype: H(s) = ωc / (s + ωc),  ωc = 2π·fc.
    Physically equivalent to a series-R / shunt-C voltage divider with
    time constant τ = RC = 1/(2π·fc).

    The 1st-order Butterworth is mathematically identical to the RC filter
    [Ref 3, §2.1].  scipy.signal.butter(1, …) applies the bilinear transform
    with pre-warping so the digital -3 dB point is exactly fc [Ref 2, §7.1].
    """
    return signal.butter(1, fc, btype="low", fs=fs, output="sos")


def design_butterworth(order: int, fc: float, fs: float) -> np.ndarray:
    """
    Butterworth low-pass filter of the specified order.

    Analogue prototype: N poles equally spaced in angle on the left unit
    semicircle of the s-plane, giving maximally flat (no passband ripple),
    monotonically decreasing magnitude [Ref 1].

    -3 dB point is exactly fc regardless of order.
    Asymptotic roll-off: -20·N dB/decade beyond fc.
    """
    return signal.butter(order, fc, btype="low", fs=fs, output="sos")


def design_elliptic(
    order: int, fc: float, fs: float,
    rp_db: float = 0.1, rs_db: float = 60.0,
) -> np.ndarray:
    """
    Elliptic (Cauer) low-pass filter.

    Equiripple in both passband (≤ rp_db dB) and stopband (≥ rs_db dB).
    For a given order, this achieves the narrowest possible transition band
    of any IIR filter — the sharpest roll-off at the cost of both passband
    and stopband ripple [Ref 4].

    scipy places the -rp_db dB passband edge at fc.

    Default parameters: rp_db = 0.1 dB passband ripple (barely perceptible),
    rs_db = 60 dB stopband attenuation (~1000× amplitude reduction).
    """
    return signal.ellip(order, rp_db, rs_db, fc,
                        btype="low", fs=fs, output="sos")


def build_filter(
    name: str, fc: float, fs: float,
    rp_db: float = 0.1, rs_db: float = 60.0,
) -> Optional[np.ndarray]:
    """
    Design a named filter.  Returns an SOS array or None on failure.

    Uses SOS output throughout for [Ref 5] numerical stability — critical
    when fc/fs is very small (e.g. 200 kHz / 2.5 GHz ≈ 1.6 × 10⁻⁴).
    """
    try:
        if name == "RC (1st)":
            return design_rc(fc, fs)
        elif name == "Butter 2nd":
            return design_butterworth(2, fc, fs)
        elif name == "Butter 4th":
            return design_butterworth(4, fc, fs)
        elif name == "Butter 8th":
            return design_butterworth(8, fc, fs)
        elif name == "Elliptic 5th":
            return design_elliptic(5, fc, fs, rp_db=rp_db, rs_db=rs_db)
        else:
            warnings.warn(f"Unknown filter name: '{name}'")
    except Exception as exc:
        warnings.warn(f"  ✗ Could not design '{name}' at {fmt_hz(fc)}: {exc}")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Filtering
# ═════════════════════════════════════════════════════════════════════════════

def apply_filter(
    sos: np.ndarray, data: np.ndarray, causal: bool = True
) -> np.ndarray:
    """
    Apply an SOS filter to a 1-D signal.

    causal = True  (default — physical simulation)
        scipy.signal.sosfilt: single forward pass.
        Initial state set to the DC steady-state for data[0] using
        sosfilt_zi [Ref 5].  This eliminates the long start-up transient
        that a real filter would take ~τ = 1/(2π·fc) seconds to clear.
        Faithfully represents what a hardware low-pass filter would do to
        the waveform, including phase delay / group delay.

    causal = False  (--zero-phase flag — magnitude analysis)
        scipy.signal.sosfiltfilt: forward then backward pass [Ref 6].
        Zero net phase shift; effective order doubled.  Useful for
        comparing the pure amplitude effect of each filter, but is
        non-causal and physically non-realisable.
    """
    if causal:
        zi       = signal.sosfilt_zi(sos) * data[0]   # DC steady-state IC
        y, _     = signal.sosfilt(sos, data, zi=zi)
    else:
        y = signal.sosfiltfilt(sos, data)
    return y


# ═════════════════════════════════════════════════════════════════════════════
# Frequency response
# ═════════════════════════════════════════════════════════════════════════════

def freq_response(
    sos: np.ndarray, fs: float, n: int = BODE_N_POINTS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (freq_Hz, magnitude_dB, phase_deg) for an SOS digital filter.
    Uses scipy.signal.sosfreqz evaluated at n evenly-spaced frequencies
    from 0 to the Nyquist rate.
    """
    w, h   = signal.sosfreqz(sos, worN=n, fs=fs)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-150))
    ph_deg = np.unwrap(np.angle(h)) * (180.0 / np.pi)
    return w, mag_db, ph_deg


# ═════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═════════════════════════════════════════════════════════════════════════════

def _decimate(arr: np.ndarray, max_pts: int) -> np.ndarray:
    """Stride-decimate arr to at most max_pts samples."""
    step = max(1, len(arr) // max_pts)
    return arr[::step]


def _freq_axis(ax: plt.Axes, xlim: Optional[Tuple] = None) -> None:
    """Apply frequency axis formatter and grid."""
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: fmt_hz(x) if x > 0 else "0"))
    ax.set_xlabel("Frequency")
    ax.grid(True, which="both", alpha=0.2, linewidth=0.5)
    if xlim:
        ax.set_xlim(*xlim)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 — Bode magnitude plots
# ═════════════════════════════════════════════════════════════════════════════

def figure_bode(
    valid_cutoffs: List[float],
    filter_names:  List[str],
    fs:            float,
    rp_db:         float,
    rs_db:         float,
) -> plt.Figure:
    """
    One subplot per cutoff frequency.
    Shows the digital filter magnitude response (dB vs log-frequency)
    for every requested filter type at that cutoff.

    Reference lines at -3, -20, -40, -60 dB.
    Vertical dashed line marks fc.
    """
    n   = len(valid_cutoffs)
    nyq = fs / 2.0

    fig, axs = plt.subplots(1, n, figsize=(max(5, 4.8 * n), 5.5),
                             constrained_layout=True)
    if n == 1:
        axs = [axs]

    fig.suptitle(
        "Fig 1 — Filter Magnitude Response (Bode Plots)\n"
        "Designed via bilinear transform with pre-warping [Ref 2]; "
        "SOS for numerical stability [Ref 5]",
        fontsize=10, fontweight="bold",
    )

    ref_lines = [(-3, "--", 0.75, "−3 dB"),
                 (-20, ":",  0.55, "−20 dB"),
                 (-40, ":",  0.40, "−40 dB"),
                 (-60, ":",  0.30, "−60 dB")]

    for ax, fc in zip(axs, valid_cutoffs):
        for fname in filter_names:
            sos = build_filter(fname, fc, fs, rp_db=rp_db, rs_db=rs_db)
            if sos is None:
                continue
            w, mag, _ = freq_response(sos, fs)
            s = STYLES[fname]
            ax.semilogx(w, mag, color=s["color"], lw=s["lw"],
                        ls=s["ls"], label=fname, zorder=s["zorder"])

        # Reference lines
        x_left = max(fc / 500.0, 1e3)
        for ref_db, ref_ls, ref_a, ref_label in ref_lines:
            ax.axhline(ref_db, color="gray", lw=0.6,
                       ls=ref_ls, alpha=ref_a, zorder=1)
            ax.text(x_left * 1.15, ref_db + 1.0, ref_label,
                    color="gray", fontsize=6, va="bottom", zorder=2)

        ax.axvline(fc, color="black", lw=0.9, ls=":", alpha=0.45)

        ax.set_ylim(-80, 5)
        ax.set_ylabel("Magnitude (dB)")
        ax.legend(fontsize=7, loc="lower left", framealpha=0.85)
        ax.set_title(f"fc = {fmt_hz(fc)}", fontsize=9, fontweight="bold")
        _freq_axis(ax, xlim=(x_left, nyq))

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 — Phase response
# ═════════════════════════════════════════════════════════════════════════════

def figure_phase(
    valid_cutoffs: List[float],
    filter_names:  List[str],
    fs:            float,
    rp_db:         float,
    rs_db:         float,
) -> plt.Figure:
    """
    Phase response (degrees vs log-frequency) for each filter at each fc.
    Helps visualise group-delay / time-delay distortion differences between
    filter types.
    """
    n   = len(valid_cutoffs)
    nyq = fs / 2.0

    fig, axs = plt.subplots(1, n, figsize=(max(5, 4.8 * n), 4.5),
                             constrained_layout=True)
    if n == 1:
        axs = [axs]

    fig.suptitle(
        "Fig 2 — Filter Phase Response\n"
        "RC / Butterworth: minimum phase.  Elliptic: minimum phase with non-linear phase.",
        fontsize=10, fontweight="bold",
    )

    for ax, fc in zip(axs, valid_cutoffs):
        for fname in filter_names:
            sos = build_filter(fname, fc, fs, rp_db=rp_db, rs_db=rs_db)
            if sos is None:
                continue
            w, _, ph = freq_response(sos, fs)
            s = STYLES[fname]
            ax.semilogx(w, ph, color=s["color"], lw=s["lw"],
                        ls=s["ls"], label=fname, zorder=s["zorder"])

        ax.axvline(fc, color="black", lw=0.9, ls=":", alpha=0.45)
        ax.set_ylabel("Phase (degrees)")
        ax.legend(fontsize=7, loc="lower left", framealpha=0.85)
        ax.set_title(f"fc = {fmt_hz(fc)}", fontsize=9, fontweight="bold")
        _freq_axis(ax, xlim=(max(fc / 500.0, 1e3), nyq))

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3 — Time-domain waveforms
# ═════════════════════════════════════════════════════════════════════════════

def figure_time(
    t:             np.ndarray,
    v:             np.ndarray,
    valid_cutoffs: List[float],
    filter_names:  List[str],
    fs:            float,
    causal:        bool,
    rp_db:         float,
    rs_db:         float,
) -> plt.Figure:
    """
    Time-domain overlay: original waveform + all filtered outputs.
    Filtered signal for each (filter, fc) pair is computed on the full
    data array with DC steady-state initial conditions [Ref 5] (causal mode)
    or sosfiltfilt (zero-phase mode).
    """
    n = len(valid_cutoffs)

    fig, axs = plt.subplots(1, n, figsize=(max(5, 4.8 * n), 4.5),
                             constrained_layout=True)
    if n == 1:
        axs = [axs]

    mode_str = "causal (sosfilt, DC IC)" if causal else "zero-phase (sosfiltfilt)"
    fig.suptitle(
        f"Fig 3 — Time-Domain Waveforms  [{mode_str}]\n"
        "Decimated for display; filter applied to full dataset.",
        fontsize=10, fontweight="bold",
    )

    t_plt = _decimate(t, MAX_PLOT_SAMPLES)
    v_plt = _decimate(v, MAX_PLOT_SAMPLES)
    t_in_ms  = t_plt * 1e3     # display in milliseconds

    for ax, fc in zip(axs, valid_cutoffs):
        s0 = STYLES["Original"]
        ax.plot(t_in_ms, v_plt,
                color=s0["color"], lw=s0["lw"], ls=s0["ls"],
                alpha=s0["alpha"], label="Original", zorder=s0["zorder"])

        for fname in filter_names:
            sos = build_filter(fname, fc, fs, rp_db=rp_db, rs_db=rs_db)
            if sos is None:
                continue
            v_filt = apply_filter(sos, v, causal=causal)
            s = STYLES[fname]
            ax.plot(t_in_ms, _decimate(v_filt, MAX_PLOT_SAMPLES),
                    color=s["color"], lw=s["lw"], ls=s["ls"],
                    alpha=s["alpha"], label=fname, zorder=s["zorder"])

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (V)")
        ax.legend(fontsize=7, framealpha=0.85)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_title(f"fc = {fmt_hz(fc)}", fontsize=9, fontweight="bold")

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4 — Power spectral density (Welch)
# ═════════════════════════════════════════════════════════════════════════════

def figure_psd(
    v:             np.ndarray,
    valid_cutoffs: List[float],
    filter_names:  List[str],
    fs:            float,
    causal:        bool,
    rp_db:         float,
    rs_db:         float,
) -> plt.Figure:
    """
    Welch power spectral density estimate (V²/Hz, log-log axes).
    Shows how each filter attenuates spectral content above fc.

    nperseg = min(N, 65536) — chosen to give good frequency resolution while
    retaining enough segments for variance reduction.  For a 200 kHz cutoff
    with fs = 2.5 GHz the frequency bin width is 2.5e9/65536 ≈ 38 kHz,
    which resolves the filter roll-off adequately.
    """
    n   = len(valid_cutoffs)
    nyq = fs / 2.0
    nperseg = min(len(v), WELCH_NPERSEG)

    # Pre-compute original PSD once
    f_w, pxx_orig = signal.welch(
        v, fs=fs, nperseg=nperseg, window="hann", scaling="density"
    )
    dc = f_w > 0   # exclude DC bin for log-log plot

    fig, axs = plt.subplots(1, n, figsize=(max(5, 4.8 * n), 4.5),
                             constrained_layout=True)
    if n == 1:
        axs = [axs]

    mode_str = "causal" if causal else "zero-phase"
    fig.suptitle(
        f"Fig 4 — Power Spectral Density — Welch Estimate  [{mode_str}]\n"
        f"nperseg = {nperseg:,}   →   Δf ≈ {fmt_hz(fs/nperseg)} per bin",
        fontsize=10, fontweight="bold",
    )

    for ax, fc in zip(axs, valid_cutoffs):
        s0 = STYLES["Original"]
        ax.loglog(f_w[dc], pxx_orig[dc],
                  color=s0["color"], lw=s0["lw"],
                  alpha=s0["alpha"], label="Original", zorder=s0["zorder"])

        for fname in filter_names:
            sos = build_filter(fname, fc, fs, rp_db=rp_db, rs_db=rs_db)
            if sos is None:
                continue
            v_filt = apply_filter(sos, v, causal=causal)
            _, pxx_f = signal.welch(
                v_filt, fs=fs, nperseg=nperseg, window="hann", scaling="density"
            )
            s = STYLES[fname]
            ax.loglog(f_w[dc], pxx_f[dc],
                      color=s["color"], lw=s["lw"], ls=s["ls"],
                      alpha=s["alpha"], label=fname, zorder=s["zorder"])

        ax.axvline(fc, color="black", lw=0.9, ls=":", alpha=0.5,
                   label=f"fc = {fmt_hz(fc)}")
        ax.set_ylabel("PSD (V² / Hz)")
        ax.legend(fontsize=7, framealpha=0.85)
        ax.set_title(f"fc = {fmt_hz(fc)}", fontsize=9, fontweight="bold")
        _freq_axis(ax, xlim=(max(fc / 100.0, f_w[dc][0]), nyq))

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Main orchestration
# ═════════════════════════════════════════════════════════════════════════════

def run(
    t:            np.ndarray,
    v:            np.ndarray,
    fs:           float,
    cutoffs:      List[float],
    filter_names: List[str],
    causal:       bool,
    time_window:  Optional[float],
    rp_db:        float,
    rs_db:        float,
    separate_filter_plots: bool,
    output:       Optional[str],
) -> None:
    nyq = fs / 2.0

    # ── Validate cutoff frequencies ──────────────────────────────────────────
    valid_cutoffs: List[float] = []
    for fc in cutoffs:
        if fc <= 0:
            print(f"  ✗  Ignoring non-positive cutoff: {fc}")
        elif fc >= nyq:
            print(
                f"  ✗  {fmt_hz(fc)} ≥ Nyquist ({fmt_hz(nyq)}) — cannot be "
                f"represented at this sample rate.  Skipping."
            )
        else:
            valid_cutoffs.append(fc)

    if not valid_cutoffs:
        sys.exit(
            "No valid cutoff frequencies remain after Nyquist check.\n"
            f"Nyquist for fs = {fmt_hz(fs)} is {fmt_hz(nyq)}.\n"
            "Lower your cutoff frequencies or use higher-rate data."
        )

    # ── Apply time window ────────────────────────────────────────────────────
    if time_window is not None:
        mask = (t - t[0]) <= time_window
        t, v = t[mask], v[mask]
        print(f"  Time window : first {fmt_time(time_window)}"
              f"  ({len(t):,} samples)")

    print(f"\n  Valid cutoffs : {', '.join(fmt_hz(f) for f in valid_cutoffs)}")
    print(f"  Filter types  : {', '.join(filter_names)}")
    print(f"  Filter mode   : {'causal — sosfilt with DC IC [Ref 5]' if causal else 'zero-phase — sosfiltfilt [Ref 6]'}")
    print(f"  Elliptic spec : rp = {rp_db} dB passband ripple, "
          f"rs = {rs_db} dB stopband attenuation")
    if separate_filter_plots and len(valid_cutoffs) == 1 and len(filter_names) > 1:
        print("  Plot layout   : separate figure per filter type")
    print()

    # ── Generate figures ─────────────────────────────────────────────────────
    split_filter_plots = (
        separate_filter_plots
        and len(valid_cutoffs) == 1
        and len(filter_names) > 1
    )

    if split_filter_plots:
        output_path = Path(output) if output else None
        for fname in filter_names:
            slug = filter_slug(fname)
            print(f"  Building plots for {fname} …")
            bode_fig = figure_bode(valid_cutoffs, [fname], fs, rp_db, rs_db)
            phase_fig = figure_phase(valid_cutoffs, [fname], fs, rp_db, rs_db)
            time_fig = figure_time(t, v, valid_cutoffs, [fname], fs, causal, rp_db, rs_db)
            psd_fig = figure_psd(v, valid_cutoffs, [fname], fs, causal, rp_db, rs_db)

            if output_path:
                for tag, fig in (
                    (f"bode_{slug}", bode_fig),
                    (f"phase_{slug}", phase_fig),
                    (f"time_{slug}", time_fig),
                    (f"psd_{slug}", psd_fig),
                ):
                    out_path = output_path.parent / f"{output_path.stem}_{tag}.png"
                    fig.savefig(out_path, dpi=150, bbox_inches="tight")
                    print(f"  ✓  Saved: {out_path}")
        if output_path:
            print()
        else:
            plt.show()
        return
    else:
        print("  Building Bode plots …")
        fig1 = figure_bode(valid_cutoffs, filter_names, fs, rp_db, rs_db)

        print("  Building phase plots …")
        fig2 = figure_phase(valid_cutoffs, filter_names, fs, rp_db, rs_db)

        print("  Applying filters and building time-domain plots …")
        fig3 = figure_time(t, v, valid_cutoffs, filter_names, fs, causal, rp_db, rs_db)

        print("  Computing Welch PSD …")
        fig4 = figure_psd(v, valid_cutoffs, filter_names, fs, causal, rp_db, rs_db)

        if output:
            output_path = Path(output)
            for tag, fig in (
                ("bode", fig1),
                ("phase", fig2),
                ("time", fig3),
                ("psd", fig4),
            ):
                out_path = output_path.parent / f"{output_path.stem}_{tag}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"  ✓  Saved: {out_path}")
            print()
        else:
            plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate RC, Butterworth, and Elliptic low-pass filters "
            "on high-speed oscilloscope data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n"
            "--------\n"
            "  # All filters, default cutoffs (200 kHz → 1 GHz)\n"
            "  python lowpass_filter_sim.py data.csv\n\n"
            "  # Custom cutoffs in Hz (scientific notation supported)\n"
            "  python lowpass_filter_sim.py data.csv --cutoffs 1e6 50e6 500e6\n\n"
            "  # Show only the RC filter, causal mode, first 1 µs\n"
            "  python lowpass_filter_sim.py data.csv --filter-type rc "
            "--causal --time-window 1e-6\n\n"
            "  # Save four PNG files to a subdirectory\n"
            "  python lowpass_filter_sim.py data.csv --output results/sim\n\n"
            "  # Stricter elliptic spec (0.05 dB / 80 dB)\n"
            "  python lowpass_filter_sim.py data.csv --rp 0.05 --rs 80\n"
        ),
    )

    parser.add_argument(
        "filepath",
        help="Path to CSV file with time (s) and voltage (V) columns.",
    )
    parser.add_argument(
        "--cutoffs", nargs="+", type=float, metavar="HZ", default=None,
        help=(
            "Cutoff frequencies in Hz (space-separated). "
            f"Default: {', '.join(fmt_hz(f) for f in DEFAULT_CUTOFFS)}. "
            "Frequencies ≥ Nyquist (fs/2) are skipped with a warning."
        ),
    )
    parser.add_argument(
        "--filter-type",
        choices=list(FILTER_TYPE_MAP.keys()), default="all",
        help=(
            "Which filter(s) to simulate. "
            "Choices: rc, butter2, butter4, butter8, elliptic5, all. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--causal", action="store_true",
        help=(
            "Use causal (one-pass sosfilt) filtering — physically realisable, "
            "includes phase delay. Default is zero-phase sosfiltfilt."
        ),
    )
    parser.add_argument(
        "--separate-filter-plots", action="store_true",
        help=(
            "When exactly one cutoff is active, render each filter type in its "
            "own figures instead of overlaying them on a single plot."
        ),
    )
    parser.add_argument(
        "--time-window", type=float, metavar="SECONDS", default=None,
        help=(
            "Analyse only the first SECONDS of data (e.g. 1e-6 for 1 µs). "
            "Reduces memory and computation for long captures."
        ),
    )
    parser.add_argument(
        "--rp", type=float, default=0.1, metavar="dB",
        help="Elliptic passband ripple in dB (default 0.1).",
    )
    parser.add_argument(
        "--rs", type=float, default=60.0, metavar="dB",
        help="Elliptic stopband attenuation in dB (default 60).",
    )
    parser.add_argument(
        "--output", metavar="STEM", default=None,
        help=(
            "Save figures to STEM_bode.png, STEM_phase.png, "
            "STEM_time.png, STEM_psd.png instead of displaying interactively."
        ),
    )

    args = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print("═" * 60)
    print("  Low-Pass Filter Simulation")
    print("  (RC · Butterworth · Elliptic)  —  SOS / bilinear transform")
    print("═" * 60)
    print(f"\n  File : {args.filepath}")

    # ── Load ──────────────────────────────────────────────────────────────────
    t, v, fs = load_csv(args.filepath)

    cutoffs      = args.cutoffs or DEFAULT_CUTOFFS
    filter_names = FILTER_TYPE_MAP[args.filter_type]

    run(
        t=t, v=v, fs=fs,
        cutoffs=cutoffs,
        filter_names=filter_names,
        causal=args.causal,
        time_window=args.time_window,
        rp_db=args.rp,
        rs_db=args.rs,
        separate_filter_plots=args.separate_filter_plots,
        output=args.output,
    )

    print("  Done.")
    print()


if __name__ == "__main__":
    main()

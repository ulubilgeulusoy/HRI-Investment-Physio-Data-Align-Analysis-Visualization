import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

# =========================
# EDIT THESE PATHS
# =========================
INPUT_FOLDER = r"C:\Users\ulubi\Desktop\Data\Data Align"          # folder with merged/aligned CSVs
OUTPUT_FOLDER = r"C:\Users\ulubi\Desktop\Data\Data Clean"      # where cleaned CSVs will be written
OVERWRITE = False  # if True, writes back into INPUT_FOLDER instead of OUTPUT_FOLDER

# =========================
# NAMING CONVENTION SUPPORT
# =========================
# Each signal can appear under multiple possible column names.
ECG_CANDIDATES = ["ECG", "ACQ_ECG"]
RSP_CANDIDATES = ["RSP", "ACQ_RSP"]
EDA_CANDIDATES = ["EDA", "ACQ_EDA"]

TIME_CANDIDATES = ["time", "Time", "timestamp", "Timestamp"]  # most common

# =========================
# FILTER SETTINGS (match your MATLAB intent)
# =========================
ECG_HIGHPASS_HZ = 1.0
ECG_LOWPASS_HZ = 100.0
NOTCH_BANDSTOP = (59.0, 61.0)  # 60 Hz removal

RSP_BANDPASS = (0.05, 3.0)
RSP_ORDER = 2

EDA_MIN = 5.0
EDA_MAX = 40.0
EDA_DERIV_THRESH = 0.5
EDA_INFLECT_THRESH = 0.5
EDA_SG_ORDER = 3
EDA_SG_FRAMELEN = 2001  # will auto-shrink if signal is shorter


# =========================
# HELPERS
# =========================
def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def estimate_fs_from_time(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    if len(t) < 3:
        raise ValueError("Time vector too short to estimate Fs.")
    dt = float(t[-1] - t[0])
    if dt <= 0:
        raise ValueError("Non-positive duration from time column.")
    return float(len(t) / dt)

def butter_filter(x: np.ndarray, fs: float, kind: str, freq, order: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    nyq = 0.5 * fs

    if kind in ("highpass", "lowpass"):
        wn = float(freq) / nyq
        wn = min(max(wn, 1e-6), 0.999999)
        b, a = butter(order, wn, btype=kind)
        return filtfilt(b, a, x)

    if kind == "bandpass":
        lo, hi = freq
        lo = min(max(lo / nyq, 1e-6), 0.999999)
        hi = min(max(hi / nyq, 1e-6), 0.999999)
        if hi <= lo:
            return x
        b, a = butter(order, [lo, hi], btype="bandpass")
        return filtfilt(b, a, x)

    if kind == "bandstop":
        lo, hi = freq
        lo = min(max(lo / nyq, 1e-6), 0.999999)
        hi = min(max(hi / nyq, 1e-6), 0.999999)
        if hi <= lo:
            return x
        b, a = butter(order, [lo, hi], btype="bandstop")
        return filtfilt(b, a, x)

    raise ValueError(f"Unknown filter kind: {kind}")

def clean_ecg(ecg: np.ndarray, fs: float) -> np.ndarray:
    y = np.asarray(ecg, dtype=float)
    y = butter_filter(y, fs, "highpass", ECG_HIGHPASS_HZ, order=4)
    y = butter_filter(y, fs, "lowpass", ECG_LOWPASS_HZ, order=4)
    y = butter_filter(y, fs, "bandstop", NOTCH_BANDSTOP, order=2)  # like your MATLAB bandstop iir order 2
    return y

def clean_rsp(rsp: np.ndarray, fs: float) -> np.ndarray:
    y = np.asarray(rsp, dtype=float)
    y = butter_filter(y, fs, "bandpass", RSP_BANDPASS, order=RSP_ORDER)
    return y

def clean_eda(eda: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(eda, dtype=float).copy()

    # remove physiologically impossible values
    x[(x > EDA_MAX) | (x < EDA_MIN)] = np.nan

    # remove high derivatives and inflection points
    d1 = np.gradient(x)
    d2 = np.gradient(d1)
    x[np.abs(d1) > EDA_DERIV_THRESH] = np.nan
    x[np.abs(d2) > EDA_INFLECT_THRESH] = np.nan

    # fill NaNs using linear interpolation (like your MATLAB fillmissing linear)
    s = pd.Series(x)
    x_filled = s.interpolate(method="linear", limit_direction="both").to_numpy(dtype=float)

    # Savitzky-Golay smoothing (framelen must be odd and <= len)
    n = len(x_filled)
    framelen = min(EDA_SG_FRAMELEN, n if n % 2 == 1 else n - 1)
    if framelen < (EDA_SG_ORDER + 2):
        # too short to apply SG meaningfully; just return filled
        return x_filled

    y = savgol_filter(x_filled, window_length=framelen, polyorder=EDA_SG_ORDER)
    return y


def main():
    in_dir = Path(INPUT_FOLDER)
    out_dir = in_dir if OVERWRITE else Path(OUTPUT_FOLDER)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in: {in_dir}")
        return

    for csv_path in csv_files:
        print(f"\nProcessing: {csv_path.name}")
        df = pd.read_csv(csv_path)

        # Find time column
        time_col = pick_first_existing_column(df, TIME_CANDIDATES)
        if time_col is None:
            print("  -> Skipping (no time column found).")
            continue

        t = df[time_col].to_numpy(dtype=float)
        try:
            fs = estimate_fs_from_time(t)
        except Exception as e:
            print(f"  -> Skipping (Fs estimate failed): {e}")
            continue

        print(f"  Fs estimated: {fs:.3f} Hz using column '{time_col}'")

        # Detect naming convention (this is how you “double check”)
        ecg_col = pick_first_existing_column(df, ECG_CANDIDATES)
        rsp_col = pick_first_existing_column(df, RSP_CANDIDATES)
        eda_col = pick_first_existing_column(df, EDA_CANDIDATES)

        print(f"  Found ECG column: {ecg_col}")
        print(f"  Found RSP column: {rsp_col}")
        print(f"  Found EDA column: {eda_col}")

        # Clean if exists, and add back to same df as new columns
        if ecg_col is not None:
            df["ECG_clean"] = clean_ecg(df[ecg_col].to_numpy(dtype=float), fs)
            print("  -> Wrote ECG_clean")

        if rsp_col is not None:
            df["RSP_clean"] = clean_rsp(df[rsp_col].to_numpy(dtype=float), fs)
            print("  -> Wrote RSP_clean")

        if eda_col is not None:
            df["EDA_clean"] = clean_eda(df[eda_col].to_numpy(dtype=float), fs)
            print("  -> Wrote EDA_clean")

        out_path = out_dir / csv_path.name
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

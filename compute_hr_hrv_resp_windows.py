import os
import glob
import numpy as np
import pandas as pd

try:
    import neurokit2 as nk
except ImportError as e:
    raise SystemExit("Missing neurokit2. Install with: pip install neurokit2") from e


# =========================
# EDIT THESE
# =========================
INPUT_FOLDER = r"C:\Users\ulubi\Desktop\Data\Data Clean"
OUTPUT_FOLDER = r"C:\Users\ulubi\Desktop\Data\Metrics"

WINDOW_S = 20.0
START_S = 0.0
END_S = None

ECG_METHOD = "neurokit"  # try "pantompkins1985", "hamilton2002", "christov2004" if needed

EMO_COLS = [
    "XDF_OpenFaceRealtime_emo_Neutral_pct",
    "XDF_OpenFaceRealtime_emo_Happy_pct",
    "XDF_OpenFaceRealtime_emo_Sad_pct",
    "XDF_OpenFaceRealtime_emo_Surprise_pct",
    "XDF_OpenFaceRealtime_emo_Fear_pct",
    "XDF_OpenFaceRealtime_emo_Disgust_pct",
    "XDF_OpenFaceRealtime_emo_Anger_pct",
    "XDF_OpenFaceRealtime_emo_Contempt_pct",
]


# =========================
# Helpers
# =========================
def _pick_signal_column(df: pd.DataFrame, base: str) -> str | None:
    cols = list(df.columns)

    if base in cols:
        return base
    acq = f"ACQ_{base}"
    if acq in cols:
        return acq

    base_lower = base.lower()
    candidates = []
    for c in cols:
        cl = c.lower()
        if cl == base_lower:
            candidates.append(c)
        elif cl == f"acq_{base_lower}":
            candidates.append(c)
        elif cl.endswith(f"_{base_lower}"):
            candidates.append(c)
        elif base_lower in cl and ("acq" in cl or "_" in cl):
            candidates.append(c)

    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]
    return candidates[0] if candidates else None


def _estimate_fs(t_rel: np.ndarray) -> float:
    dt = np.diff(t_rel)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if len(dt) == 0:
        return np.nan
    return float(1.0 / np.median(dt))


def _compute_ecg_metrics(ecg: np.ndarray, fs: float):
    if not np.isfinite(fs) or fs <= 0 or len(ecg) < max(3, int(2 * fs)):
        return (np.nan, np.nan, np.nan, 0)

    try:
        signals, info = nk.ecg_process(ecg, sampling_rate=fs, method=ECG_METHOD)
        rpeaks = info.get("ECG_R_Peaks", None)
        if rpeaks is None or len(rpeaks) < 2:
            return (np.nan, np.nan, np.nan, int(len(rpeaks) if rpeaks is not None else 0))

        hr = signals["ECG_Rate"].to_numpy(dtype=float)
        mean_hr = float(np.nanmean(hr))

        hrv = nk.hrv_time(rpeaks, sampling_rate=fs, show=False)
        rmssd = float(hrv.get("HRV_RMSSD", [np.nan])[0])
        sdnn = float(hrv.get("HRV_SDNN", [np.nan])[0])

        return (mean_hr, rmssd, sdnn, int(len(rpeaks)))
    except Exception:
        return (np.nan, np.nan, np.nan, 0)


def _compute_rsp_rate(rsp: np.ndarray, fs: float):
    if not np.isfinite(fs) or fs <= 0 or len(rsp) < max(3, int(5 * fs)):
        return (np.nan, 0)

    try:
        rsp_signals, rsp_info = nk.rsp_process(rsp, sampling_rate=fs)
        rate = rsp_signals["RSP_Rate"].to_numpy(dtype=float)
        mean_rate = float(np.nanmean(rate))

        peaks = rsp_info.get("RSP_Peaks", None)
        n_breaths = int(len(peaks) if peaks is not None else 0)

        return (mean_rate, n_breaths)
    except Exception:
        return (np.nan, 0)


def _window_edges(t_rel: np.ndarray, start_s: float, end_s: float | None, win_s: float):
    t0 = float(start_s)
    t1 = float(t_rel[-1]) if end_s is None else float(end_s)
    if t1 <= t0:
        return []
    n = int(np.floor((t1 - t0) / win_s))
    return [(t0 + k * win_s, t0 + (k + 1) * win_s) for k in range(n)]


def _safe_mean(series: pd.Series) -> float:
    # robust to non-numeric
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def process_file(csv_path: str):
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise RuntimeError(f"{os.path.basename(csv_path)}: missing 'time' column")

    t = df["time"].to_numpy(dtype=float)
    if len(t) < 5:
        return None

    t_rel = t - t[0]  # windowing in relative seconds
    fs = _estimate_fs(t_rel)

    # Your cleaned columns (as you edited earlier)
    ecg_col = _pick_signal_column(df, "ECG_clean")
    rsp_col = _pick_signal_column(df, "RSP_clean")

    # Emotion columns that actually exist in this file
    emo_present = [c for c in EMO_COLS if c in df.columns]

    if ecg_col is None and rsp_col is None and len(emo_present) == 0:
        raise RuntimeError(
            f"{os.path.basename(csv_path)}: could not find ECG_clean / RSP_clean or any emotion columns."
        )

    rows = []
    for (a, b) in _window_edges(t_rel, START_S, END_S, WINDOW_S):
        m = (t_rel >= a) & (t_rel < b)
        if np.sum(m) < 3:
            continue

        row = {
            "file": os.path.basename(csv_path),
            "win_start_s": a,
            "win_end_s": b,
            "fs_hz_est": fs,
            "ecg_col": ecg_col or "",
            "rsp_col": rsp_col or "",
        }

        # ECG metrics
        if ecg_col is not None:
            ecg_seg = pd.to_numeric(df.loc[m, ecg_col], errors="coerce").to_numpy(dtype=float)
            mean_hr, rmssd, sdnn, n_beats = _compute_ecg_metrics(ecg_seg, fs)
            row.update({
                "mean_hr_bpm": mean_hr,
                "hrv_rmssd_ms": rmssd,
                "hrv_sdnn_ms": sdnn,
                "n_rpeaks": n_beats,
            })
        else:
            row.update({
                "mean_hr_bpm": np.nan,
                "hrv_rmssd_ms": np.nan,
                "hrv_sdnn_ms": np.nan,
                "n_rpeaks": 0,
            })

        # RSP metrics
        if rsp_col is not None:
            rsp_seg = pd.to_numeric(df.loc[m, rsp_col], errors="coerce").to_numpy(dtype=float)
            mean_rr, n_breaths = _compute_rsp_rate(rsp_seg, fs)
            row.update({
                "mean_resp_bpm": mean_rr,
                "n_breaths": n_breaths,
            })
        else:
            row.update({
                "mean_resp_bpm": np.nan,
                "n_breaths": 0,
            })

        # Emotion means
        for emo_col in EMO_COLS:
            out_col = "mean_" + emo_col.replace("XDF_OpenFaceRealtime_", "")
            if emo_col in df.columns:
                row[out_col] = _safe_mean(df.loc[m, emo_col])
            else:
                row[out_col] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in: {INPUT_FOLDER}")

    for f in files:
        try:
            out = process_file(f)
            if out is None or out.empty:
                print(f"Skip (no windows): {os.path.basename(f)}")
                continue

            out_name = os.path.splitext(os.path.basename(f))[0] + "_metrics_10s_with_emotions.csv"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            out.to_csv(out_path, index=False)
            print(f"Wrote: {out_path}  (rows={len(out)})")

        except Exception as e:
            print(f"ERROR processing {os.path.basename(f)}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()

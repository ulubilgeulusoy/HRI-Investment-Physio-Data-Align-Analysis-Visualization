# -*- coding: utf-8 -*-
"""
align_acq_xdf_dropout_merge_all_xdf.py

Based on your dropout-only script fileciteturn6file0L1-L28, but now the output CSV can include
ALL XDF streams (e.g., OpenFace) aligned onto the same time axis as the aligned ACQ ECG/RSP.

How it works (high level)
-------------------------
1) Find dropout/flatline anchor in the chosen XDF physio stream.
2) Find dropout/flatline anchor in the ACQ.
3) Compute offset t_opt = acq_anchor_t - xdf_anchor_t.
4) Build the aligned ACQ time vector in *absolute XDF time*.
5) Merge other XDF streams onto the aligned ACQ time base:
   - If the XDF stream has nominal_srate > 0: interpolate each channel onto ACQ time.
   - Else (irregular): merge_asof (nearest sample) with a tolerance.

You do NOT need any command line args. Just edit the PATHS + settings at the top, then run:
    python align_acq_xdf_dropout_merge_all_xdf.py
"""

import numpy as np
import pandas as pd
import neurokit2 as nk
import pyxdf

# =========================
# EDIT THESE PATHS
# =========================
ACQ_PATH = r"C:\Users\ulubi\Desktop\Data\block_ulu_new.acq"
XDF_PATH = r"C:\Users\ulubi\Desktop\Data\block_ulu_new.xdf"
OUT_CSV  = r"C:\Users\ulubi\Desktop\Data\Data Align\aligned_dropout_merged.csv"

# =========================
# WHICH XDF STREAM IS "PHYSIO" (used ONLY to detect dropout anchor + define trim window)
# =========================
# If you know it exactly, set it; else None (auto-pick)
XDF_PHYSIO_STREAM_NAME = None  # e.g., "Biopac ECG-RSP"

# If physio XDF columns are unlabeled/mis-mapped, set them manually (must match channel count).
# Example for 2 channels: ["ECG", "RSP"]
XDF_PHYSIO_MANUAL_COLS = None  # or ["ECG", "RSP"]

# =========================
# MERGE OTHER XDF STREAMS INTO CSV?
# =========================
INCLUDE_OTHER_XDF_STREAMS = True
INCLUDE_PHYSIO_STREAM_IN_OUTPUT = True
ADD_TIMESTAMP_AND_TIMESEC = True

# Option A: include ALL other streams
INCLUDE_STREAM_NAMES = None   # None means "include everything except the physio stream"

# Option B: include only some streams by name (uncomment and edit)
# INCLUDE_STREAM_NAMES = ["OpenFaceRealtime"]

# Optional: exclude some streams by name
EXCLUDE_STREAM_NAMES = []  # e.g., ["Audio", "Video"]

# For irregular streams (nominal_srate == 0): nearest-neighbor merge tolerance (seconds)
ASOF_TOL_S = 0.050  # 50 ms

# =========================
# ACQ settings
# =========================
FORCE_ACQ_SR = None  # set to a number (e.g., 2000.0) or keep None to use nk.read_acqknowledge() SR

ACQ_RENAME_CANDIDATES = [
    {"RSP, X, RSPEC-R": "RSP", "RSP, Y, RSPEC-R": "RSP",
     "EDA, X, PPGED-R": "EDA", "EDA, Y, PPGED-R": "EDA",
     "ECG, X, RSPEC-R": "ECG", "ECG, Y, RSPEC-R": "ECG",
     "DTU100 - Trigger View, AMI / HLT - A11": "TRIG"},
]

# =========================
# DROPOUT (flatline) detection tuning
# =========================
FLAT_WIN_S = 2.0
FLAT_STD_ABS = 1e-4
FLAT_STD_REL = 0.02
DT_MAX_FOR_SR_EST = 1.0


# -------------------------
# Helpers
# -------------------------
def _safe_np(x) -> np.ndarray:
    return np.asarray(x).astype(float)

def _has_col(df: pd.DataFrame, name: str) -> bool:
    return df is not None and name in df.columns

def _estimate_xdf_sr_from_timestamps(xdf_times0: np.ndarray) -> float | None:
    if len(xdf_times0) < 10:
        return None
    dt = np.diff(xdf_times0)
    dt = dt[(dt > 0) & (dt < DT_MAX_FOR_SR_EST)]
    if len(dt) == 0:
        return None
    return float(1.0 / np.median(dt))

def _detect_flatline_anchor_index(signal: np.ndarray, sr: float, win_s: float) -> int | None:
    x = _safe_np(signal)
    win_n = max(1, int(round(win_s * sr)))
    if len(x) < win_n + 2:
        return None

    global_std = float(np.std(x))
    thr = max(FLAT_STD_ABS, FLAT_STD_REL * global_std)

    c1 = np.cumsum(x)
    c2 = np.cumsum(x * x)
    sum_w = c1[win_n:] - c1[:-win_n]
    sumsq_w = c2[win_n:] - c2[:-win_n]
    mean_w = sum_w / win_n
    var_w = (sumsq_w / win_n) - (mean_w * mean_w)
    var_w[var_w < 0] = 0
    std_w = np.sqrt(var_w)

    idx = np.where(std_w < thr)[0]
    if len(idx) == 0:
        return None
    return int(idx[0])

def _extract_xdf_channel_labels(st) -> list[str] | None:
    """Best-effort extraction of XDF channel labels (if present)."""
    try:
        desc = st["info"].get("desc", None)
        if not desc:
            return None
        desc0 = desc[0]
        channels = desc0.get("channels", None)
        if not channels:
            return None
        ch_list = channels[0].get("channel", None)
        if not ch_list:
            return None
        labels = []
        for ch in ch_list:
            lab = ch.get("label", None) or ch.get("name", None)
            if isinstance(lab, list) and lab:
                labels.append(str(lab[0]))
            elif isinstance(lab, str):
                labels.append(lab)
            else:
                labels.append("")
        return labels
    except Exception:
        return None

def _clean_colname(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch in ["_", "-"])
    return s if s else "ch"

def _list_and_load_all_xdf(xdf_path: str):
    streams, _ = pyxdf.load_xdf(xdf_path, dejitter_timestamps=False)
    meta = []
    for i, st in enumerate(streams):
        info = st["info"]
        name = info.get("name", [""])[0]
        stype = info.get("type", [""])[0]
        chn = int(info.get("channel_count", ["0"])[0])
        srate = float(info.get("nominal_srate", ["0"])[0] or 0.0)
        meta.append((i, name, stype, chn, srate))
    return streams, meta

def _pick_physio_stream_index(meta, preferred_name: str | None):
    if preferred_name:
        for (i, name, *_rest) in meta:
            if name == preferred_name:
                return i
        return None
    for (i, name, stype, *_rest) in meta:
        t = (name or "").lower() + " " + (stype or "").lower()
        if "biopac" in t or "psychophys" in t or "phys" in t or "ecg" in t or "rsp" in t or "eda" in t or "gsr" in t:
            return i
    return max(meta, key=lambda x: x[3])[0] if meta else None

def _map_physio_columns(xdf_raw: pd.DataFrame, manual_cols: list[str] | None):
    xdf = xdf_raw.copy()
    if manual_cols is not None:
        if len(manual_cols) != xdf.shape[1]:
            raise RuntimeError(f"XDF_PHYSIO_MANUAL_COLS length ({len(manual_cols)}) != physio channel_count ({xdf.shape[1]}).")
        xdf.columns = manual_cols
        return xdf
    # fallback common mapping
    if xdf.shape[1] == 2:
        xdf.columns = ["ECG", "RSP"]
    elif xdf.shape[1] >= 3:
        xdf.columns = ["RSP", "EDA", "ECG"] + [f"XDF_{i}" for i in range(3, xdf.shape[1])]
    else:
        xdf.columns = ["ECG"]
    return xdf

def _interp_stream_to_timebase(times: np.ndarray, series2d: np.ndarray, target_times: np.ndarray):
    """Interpolate each column of series2d to target_times."""
    out = np.empty((len(target_times), series2d.shape[1]), dtype=float)
    for j in range(series2d.shape[1]):
        y = series2d[:, j].astype(float, copy=False)
        out[:, j] = np.interp(target_times, times, y)
    return out


def main():
    # --- Load ACQ
    acq_df, acq_sr_reported = nk.read_acqknowledge(ACQ_PATH)
    for m in ACQ_RENAME_CANDIDATES:
        acq_df = acq_df.rename(columns=m)
    acq_sr = float(FORCE_ACQ_SR) if FORCE_ACQ_SR is not None else float(acq_sr_reported)

    # --- Load all XDF streams
    streams, meta = _list_and_load_all_xdf(XDF_PATH)
    if not meta:
        raise RuntimeError("No streams found in XDF.")

    phys_idx = _pick_physio_stream_index(meta, XDF_PHYSIO_STREAM_NAME)
    if phys_idx is None:
        raise RuntimeError("Could not pick physio stream. Set XDF_PHYSIO_STREAM_NAME exactly.")

    phys_st = streams[phys_idx]
    xdf_physio_raw = pd.DataFrame(phys_st["time_series"])
    xdf_physio_times = _safe_np(phys_st["time_stamps"])
    xdf_physio_times0 = xdf_physio_times - xdf_physio_times[0]
    xdf_sr = _estimate_xdf_sr_from_timestamps(xdf_physio_times0)
    if xdf_sr is None:
        raise RuntimeError("Could not estimate physio XDF sampling rate from timestamps.")

    xdf_physio_df = _map_physio_columns(xdf_physio_raw, XDF_PHYSIO_MANUAL_COLS)

    # --- Find flatline anchor in physio XDF
    xdf_anchor_t = None
    xdf_anchor_col = None
    preferred = [c for c in ["ECG", "RSP", "EDA"] if _has_col(xdf_physio_df, c)]
    scan_cols = preferred + [c for c in xdf_physio_df.columns if c not in preferred]

    for col in scan_cols:
        idx = _detect_flatline_anchor_index(xdf_physio_df[col].values, xdf_sr, FLAT_WIN_S)
        if idx is not None:
            xdf_anchor_t = float(xdf_physio_times0[idx])
            xdf_anchor_col = col
            break

    if xdf_anchor_t is None:
        raise RuntimeError(
            "No flatline found in physio XDF. This script ONLY supports dropout/flatline alignment.\n"
            "Try increasing FLAT_WIN_S and/or FLAT_STD_REL."
        )

    # --- Find matching flatline in ACQ
    acq_anchor_t = None
    acq_anchor_col = None
    order = []
    if xdf_anchor_col in acq_df.columns:
        order.append(xdf_anchor_col)
    order += [c for c in ["ECG", "RSP", "EDA"] if _has_col(acq_df, c) and c not in order]
    order += [c for c in acq_df.columns if c not in order]

    for col in order:
        idx = _detect_flatline_anchor_index(acq_df[col].values, acq_sr, FLAT_WIN_S)
        if idx is not None:
            acq_anchor_t = float(idx / acq_sr)
            acq_anchor_col = col
            break

    if acq_anchor_t is None:
        raise RuntimeError(
            "Found a flatline in XDF physio, but could not find a flatline in ACQ.\n"
            "Try increasing FLAT_WIN_S and/or FLAT_STD_REL."
        )

    # --- Compute offset
    t_opt = acq_anchor_t - xdf_anchor_t

    # --- Build aligned ACQ time base (absolute XDF time) trimmed to physio XDF duration
    total_time = float(xdf_physio_times0[-1])
    acq_times_full = np.arange(len(acq_df), dtype=float) / acq_sr
    acq_times_shifted = acq_times_full - t_opt

    start_i = int(np.sum(acq_times_shifted < 0))
    end_i = int(np.sum(acq_times_shifted < total_time))
    if end_i <= start_i + 2:
        raise RuntimeError("After trimming, ACQ became too short. Check file pairing and flatline anchors.")

    acq_trim = acq_df.iloc[start_i:end_i].copy()
    acq_times_trim_rel = acq_times_shifted[start_i:end_i]
    acq_times_abs = acq_times_trim_rel + float(xdf_physio_times[0])

    out = pd.DataFrame({"time": acq_times_abs})
    for col in acq_trim.columns:
        # keep everything from ACQ (not only ECG/RSP/EDA)
        out[f"ACQ_{_clean_colname(col)}"] = acq_trim[col].values

    # --- Merge other XDF streams
    if INCLUDE_OTHER_XDF_STREAMS:
        t_start = float(acq_times_abs[0])
        t_end = float(acq_times_abs[-1])
        base_time = out["time"].values.astype(float)

        for (i, name, stype, chn, srate) in meta:
            if i == phys_idx and not INCLUDE_PHYSIO_STREAM_IN_OUTPUT:
                continue

            if INCLUDE_STREAM_NAMES is not None and name not in INCLUDE_STREAM_NAMES:
                continue
            if name in EXCLUDE_STREAM_NAMES:
                continue

            st = streams[i]
            ts = _safe_np(st["time_stamps"])
            if len(ts) < 2:
                continue

            # trim to overlap window
            m = (ts >= t_start) & (ts <= t_end)
            n_overlap = int(np.sum(m))
            if n_overlap < 1:
                ys_all = np.asarray(st["time_series"])
                n_ch = int(ys_all.shape[1]) if ys_all.ndim > 1 else 1
                labels = _extract_xdf_channel_labels(st)
                if labels is not None and len(labels) == n_ch:
                    colnames = [f"XDF_{_clean_colname(name)}_{_clean_colname(lab)}" for lab in labels]
                else:
                    colnames = [f"XDF_{_clean_colname(name)}_ch{j}" for j in range(n_ch)]
                for cn in colnames:
                    out[cn] = np.nan
                continue

            ts = ts[m]
            ys = np.asarray(st["time_series"])[m]

            # build column names
            labels = _extract_xdf_channel_labels(st)
            if labels is not None and len(labels) == ys.shape[1]:
                colnames = [f"XDF_{_clean_colname(name)}_{_clean_colname(lab)}" for lab in labels]
            else:
                colnames = [f"XDF_{_clean_colname(name)}_ch{j}" for j in range(ys.shape[1])]

            # Decide interpolation vs nearest merge
            if float(srate) > 0 and ys.shape[0] >= 3:
                # Ensure timestamps monotonic for interp
                order_idx = np.argsort(ts)
                ts2 = ts[order_idx]
                ys2 = ys[order_idx]

                # Drop duplicate timestamps (np.interp needs increasing x)
                uniq_mask = np.concatenate([[True], np.diff(ts2) > 0])
                ts2 = ts2[uniq_mask]
                ys2 = ys2[uniq_mask]

                # Interpolate numeric columns; non-numeric falls back to asof
                try:
                    ys2 = ys2.astype(float)
                    interp = _interp_stream_to_timebase(ts2, ys2, base_time)
                    for j, cn in enumerate(colnames):
                        out[cn] = interp[:, j]
                except Exception:
                    # Fallback: nearest merge
                    df_s = pd.DataFrame({"time": ts2})
                    for j, cn in enumerate(colnames):
                        df_s[cn] = ys2[:, j]
                    out = pd.merge_asof(
                        out.sort_values("time"),
                        df_s.sort_values("time"),
                        on="time",
                        direction="nearest",
                        tolerance=ASOF_TOL_S,
                    )
            else:
                # Irregular stream: nearest neighbor merge
                df_s = pd.DataFrame({"time": ts})
                for j, cn in enumerate(colnames):
                    df_s[cn] = ys[:, j]
                out = pd.merge_asof(
                    out.sort_values("time"),
                    df_s.sort_values("time"),
                    on="time",
                    direction="nearest",
                    tolerance=ASOF_TOL_S,
                )

        # restore time order
        out = out.sort_values("time").reset_index(drop=True)

    if ADD_TIMESTAMP_AND_TIMESEC:
        out["timestamp"] = out["time"]
        out["time_sec"] = out["time"] - float(out["time"].iloc[0])

    out.to_csv(OUT_CSV, index=False)

    print("Done.")
    print("Method used: DROPOUT_FLATLINE_ONLY + MERGE_XDF_STREAMS")
    print(f"ACQ SR used: {acq_sr} Hz")
    print(f"Physio XDF SR estimate: {xdf_sr}")
    print(f"Physio XDF stream: idx={phys_idx}, name='{meta[phys_idx][1]}', type='{meta[phys_idx][2]}'")
    print(f"XDF flatline found in column: {xdf_anchor_col} at t={xdf_anchor_t:.6f}s (relative to physio XDF start)")
    print(f"ACQ flatline found in column: {acq_anchor_col} at t={acq_anchor_t:.6f}s (relative to ACQ start)")
    print(f"t_opt (s): {t_opt:.6f}")
    print(f"Wrote: {OUT_CSV}")
    print(f"Columns: {len(out.columns)} total")


if __name__ == "__main__":
    main()

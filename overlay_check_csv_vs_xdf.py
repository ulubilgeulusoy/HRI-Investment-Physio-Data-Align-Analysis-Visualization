import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyxdf

# =========================
# EDIT THESE PATHS
# =========================
CSV_PATH = r"C:\Users\ulubi\Desktop\Data\Data Align\aligned_dropout_merged.csv"
XDF_PATH = r"C:\Users\ulubi\Desktop\Data\block_ulu_new.xdf"

# If you know the exact XDF stream name, set it (e.g., "Biopac ECG-RSP" or "Biopac Data").
# If None, the script will auto-pick a likely physio stream.
XDF_STREAM_NAME = None

# If XDF channel labels are missing/unreliable, you can FORCE the channel order for that stream:
# Example from your screenshot (2 channels): ["ECG", "RSP"]
XDF_MANUAL_COLS = None  # e.g., ["ECG", "RSP"]

# =========================
# HELPERS
# =========================
def list_xdf_streams(xdf_path):
    streams, _ = pyxdf.load_xdf(xdf_path, dejitter_timestamps=False)
    meta = []
    for i, st in enumerate(streams):
        info = st["info"]
        name = info.get("name", [""])[0]
        stype = info.get("type", [""])[0]
        chn = int(info.get("channel_count", ["0"])[0])
        srate = info.get("nominal_srate", ["0"])[0]
        meta.append((i, name, stype, chn, srate))
    return streams, meta

def pick_stream_index(meta, preferred_name=None):
    if preferred_name:
        for (i, name, *_rest) in meta:
            if name == preferred_name:
                return i
        raise ValueError(f"Could not find XDF stream with name='{preferred_name}'.")
    # auto-pick: prefer streams that look like biopac/physio
    for (i, name, stype, *_rest) in meta:
        t = (name or "").lower() + " " + (stype or "").lower()
        if any(k in t for k in ["biopac", "ecg", "rsp", "eda", "gsr", "phys"]):
            return i
    # fallback: largest channel count
    return max(meta, key=lambda x: x[3])[0]

def extract_channel_labels(st):
    try:
        desc = st["info"].get("desc", None)
        if not desc:
            return None
        channels = desc[0].get("channels", None)
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

def canonicalize(label: str) -> str:
    t = (label or "").lower()
    if "ecg" in t:
        return "ECG"
    if "rsp" in t or "resp" in t:
        return "RSP"
    if "eda" in t or "gsr" in t:
        return "EDA"
    if "trig" in t or "trigger" in t or "marker" in t:
        return "TRIG"
    return ""

def map_xdf_columns(df, labels, manual_cols=None):
    out = df.copy()

    if manual_cols is not None:
        if len(manual_cols) != out.shape[1]:
            raise ValueError(f"XDF_MANUAL_COLS length {len(manual_cols)} != channel_count {out.shape[1]}")
        out.columns = manual_cols
        return out

    if labels is not None and len(labels) == out.shape[1]:
        canon = [canonicalize(l) for l in labels]
        if any(c != "" for c in canon):
            out.columns = [c if c != "" else f"XDF_{i}" for i, c in enumerate(canon)]
            return out

    # fallback heuristics (common for 2ch biopac in your screenshot)
    if out.shape[1] == 2:
        out.columns = ["ECG", "RSP"]
    elif out.shape[1] >= 3:
        out.columns = ["RSP", "EDA", "ECG"] + [f"XDF_{i}" for i in range(3, out.shape[1])]
    else:
        out.columns = ["ECG"]
    return out

def interp_to_csv_time(xdf_times, xdf_signal, csv_time):
    # Ensure ascending times for np.interp
    order = np.argsort(xdf_times)
    t = xdf_times[order]
    y = xdf_signal[order]
    return np.interp(csv_time, t, y)

# =========================
# MAIN
# =========================
# Load aligned CSV
csv = pd.read_csv(CSV_PATH)
if "time" not in csv.columns:
    raise ValueError("CSV must contain a 'time' column.")
t_csv = csv["time"].to_numpy(dtype=float)

# Load XDF, pick the stream, map columns, and compute time vector
streams, meta = list_xdf_streams(XDF_PATH)
idx = pick_stream_index(meta, XDF_STREAM_NAME)
st = streams[idx]
xdf_data = np.asarray(st["time_series"])
xdf_times = np.asarray(st["time_stamps"], dtype=float)
labels = extract_channel_labels(st)

xdf_df = pd.DataFrame(xdf_data)
xdf_df = map_xdf_columns(xdf_df, labels, XDF_MANUAL_COLS)

# ---- NEW: normalize time to start at 0 ----
t0 = float(t_csv[0])
t_csv = t_csv - t0
xdf_times = xdf_times - t0

print("Picked XDF stream:")
print(f"  index={idx}, name='{st['info'].get('name',[''])[0]}', type='{st['info'].get('type',[''])[0]}', channels={xdf_df.shape[1]}")
print("Mapped XDF columns:", list(xdf_df.columns))
print("CSV columns:", list(csv.columns))

def pick_csv_signal_col(csv_columns, base_name):
    """
    Returns the column name to use for a signal.
    Prefers exact match (ECG), else ACQ_ECG, else None.
    """
    if base_name in csv_columns:
        return base_name
    pref = f"ACQ_{base_name}"
    if pref in csv_columns:
        return pref
    return None

csv_cols = list(csv.columns)
csv_ecg_col = pick_csv_signal_col(csv_cols, "ECG")
csv_rsp_col = pick_csv_signal_col(csv_cols, "RSP")

print("Detected CSV signal columns:")
print("  ECG ->", csv_ecg_col)
print("  RSP ->", csv_rsp_col)


# Overlay ECG
if csv_ecg_col is not None and "ECG" in xdf_df.columns:
    xdf_ecg_on_csv = interp_to_csv_time(xdf_times, xdf_df["ECG"].to_numpy(dtype=float), t_csv)

    plt.figure()
    plt.plot(t_csv, csv[csv_ecg_col].to_numpy(dtype=float), label=f"CSV ({csv_ecg_col})")
    plt.plot(t_csv, xdf_ecg_on_csv, label="XDF (ECG interpolated)")
    plt.title("ECG overlay (CSV vs XDF)")
    plt.xlabel("Time (seconds, XDF clock)")
    plt.ylabel("ECG (raw units)")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Skipping ECG overlay: ECG not found in both CSV and XDF.")


# Overlay RSP
if csv_rsp_col is not None and "RSP" in xdf_df.columns:
    xdf_rsp_on_csv = interp_to_csv_time(xdf_times, xdf_df["RSP"].to_numpy(dtype=float), t_csv)

    plt.figure()
    plt.plot(t_csv, csv[csv_rsp_col].to_numpy(dtype=float), label=f"CSV ({csv_rsp_col})")
    plt.plot(t_csv, xdf_rsp_on_csv, label="XDF (RSP interpolated)")
    plt.title("RSP overlay (CSV vs XDF)")
    plt.xlabel("Time (seconds, XDF clock)")
    plt.ylabel("RSP (raw units)")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Skipping RSP overlay: RSP not found in both CSV and XDF.")


print("\nAll XDF streams in file (for debugging):")
for (i, name, stype, chn, srate) in meta:
    print(f"  [{i}] name='{name}' type='{stype}' channels={chn} nominal_srate={srate}")

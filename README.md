# HRI Investment Physio Data Pipeline (Script-Based)

This branch is focused on running the pipeline through standalone Python scripts.

## Scripts

- `align_acq_xdf_dropout_merge_all_xdf.py`
: Aligns BIOPAC `.acq` and XDF `.xdf` by dropout/flatline anchor, then merges selected/all XDF streams into one aligned CSV.

- `clean_physio_csv_folder.py`
: Reads aligned CSV files and creates cleaned signals (`ECG_clean`, `RSP_clean`, `EDA_clean`) with filtering and artifact handling.

- `compute_hr_hrv_resp_windows.py`
: Computes windowed HR/HRV/respiration metrics from cleaned CSV files and optionally aggregates OpenFace emotion columns.

- `overlay_check_csv_vs_xdf.py`
: Plots visual overlays between aligned CSV signals and raw XDF stream signals for quality checking.

## Requirements

Python 3.9+ recommended.

Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas scipy matplotlib neurokit2 pyxdf
```

## Script Order

Run scripts in this order:

1. `python align_acq_xdf_dropout_merge_all_xdf.py`
2. `python clean_physio_csv_folder.py`
3. `python compute_hr_hrv_resp_windows.py`
4. `python overlay_check_csv_vs_xdf.py` (optional QA)

## Configuration Style

Each script uses editable constants near the top (no CLI args required). Before running, update path/config blocks in each file.

### 1) Alignment Script

File: `align_acq_xdf_dropout_merge_all_xdf.py`

Edit these groups:
- Paths:
  - `ACQ_PATH`
  - `XDF_PATH`
  - `OUT_CSV`
- Physio stream selection:
  - `XDF_PHYSIO_STREAM_NAME`
  - `XDF_PHYSIO_MANUAL_COLS`
- Merge behavior:
  - `INCLUDE_OTHER_XDF_STREAMS`
  - `INCLUDE_PHYSIO_STREAM_IN_OUTPUT`
  - `INCLUDE_STREAM_NAMES` / `EXCLUDE_STREAM_NAMES`
  - `ASOF_TOL_S`
- Flatline detection tuning:
  - `FLAT_WIN_S`
  - `FLAT_STD_ABS`
  - `FLAT_STD_REL`

Output:
- One aligned merged CSV at `OUT_CSV`.

### 2) Cleaning Script

File: `clean_physio_csv_folder.py`

Edit these groups:
- Paths:
  - `INPUT_FOLDER`
  - `OUTPUT_FOLDER`
  - `OVERWRITE`
- Signal naming candidates:
  - `ECG_CANDIDATES`
  - `RSP_CANDIDATES`
  - `EDA_CANDIDATES`
  - `TIME_CANDIDATES`
- Filter settings:
  - ECG high/low/notch
  - RSP bandpass
  - EDA clipping/derivative/smoothing thresholds

Output:
- Cleaned CSV files in `OUTPUT_FOLDER` (or `INPUT_FOLDER` if `OVERWRITE=True`).

### 3) Metrics Script

File: `compute_hr_hrv_resp_windows.py`

Edit these groups:
- Paths:
  - `INPUT_FOLDER`
  - `OUTPUT_FOLDER`
- Windowing:
  - `WINDOW_S`
  - `START_S`
  - `END_S`
- ECG detector choice:
  - `ECG_METHOD`
- Emotion columns list:
  - `EMO_COLS`

Output:
- Per-file metrics CSVs in `OUTPUT_FOLDER`.

### 4) Overlay QA Script

File: `overlay_check_csv_vs_xdf.py`

Edit these groups:
- Paths:
  - `CSV_PATH`
  - `XDF_PATH`
- Stream/column mapping:
  - `XDF_STREAM_NAME`
  - `XDF_MANUAL_COLS`

Output:
- Overlay plots (matplotlib windows) for visual alignment checks.

## Notes

- This script workflow is path-constant driven; each script is run independently.
- `.gitignore` excludes data artifacts (`*.csv`, `*.acq`, `*.xdf`) and Python cache files.
- If alignment fails to find a dropout/flatline anchor, relax detection settings in the alignment script.

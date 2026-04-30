# HRI Investment Physio Data Align / Analysis / Visualization

This `main` branch contains the unified GUI-based pipeline for:

1. Aligning BIOPAC `.acq` and LSL/XDF `.xdf` recordings (dropout/flatline based)
2. Cleaning ECG/RSP/EDA channels
3. Computing marker-based HR, HRV, and respiration features
4. Visual quality checks and overlay plotting

**Note:** To ensure a flatline signal appears at the end of your recordings, turn off the Bionomadix transmitter before stopping and saving the data.

## Repository Contents (main)

- `pipeline_gui.py`: end-to-end GUI application
- `README.md`: branch documentation

Data/output folders are created automatically by the GUI under the repository root:

- `raw_xdf_acq_Data/`
- `aligned_Data/`
- `aligned_cleaned_Data/`
- `feature_extracted_Data/`

## Requirements

Python 3.9+ recommended.

Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas neurokit2 pyxdf scipy matplotlib
```

## Run the GUI

```powershell
python pipeline_gui.py
```

Startup behavior:
- GUI opens in a blank/reset state (no auto-filled file paths).
- `Load Latest From Folders` can populate recent files from repo data folders.
- ACQ/XDF browse defaults to `raw_xdf_acq_Data/`.
- CSV browse defaults to `aligned_cleaned_Data/` (fallback: `aligned_Data/`).

## Workflow

1. Select `.acq` and `.xdf` files.
2. Click `Import/Copy Raw Files`.
3. Run `Run Alignment` or `Run Full Pipeline (Align+Clean)`.
4. Open `Open Feature GUI` to compute marker-based features.
5. Use visualizer and overlay sections for QA.

## GUI Sections

1. `Raw import`
- Select/copy raw ACQ + XDF files into repo raw folder.

2. `Pipeline`
- Alignment output written to `aligned_Data/`.
- Cleaning output written to `aligned_cleaned_Data/`.
- Feature outputs written to `feature_extracted_Data/`.
- Progress bar/status messages are shown for long tasks.

3. `Visualizers`
- Load CSV columns and create X-Y/scatter plots.
- Optional scope by marker start/end.
- Export selected scope CSV.

4. `Overlay (Configurable)`
- Load overlay options from selected CSV + XDF.
- Select CSV signal, XDF stream, XDF signal.
- Plot selected overlay for alignment QA.

## Outputs

- Alignment: `<acq_stem>__<xdf_stem>_aligned_dropout_merged.csv` in `aligned_Data/`
- Cleaning: same stem in `aligned_cleaned_Data/`
- Marker features: `<cleaned_stem>_marker_features.csv` in `feature_extracted_Data/`

## Cleaning

Cleaning is performed by `run_clean()` in `pipeline_gui.py` on the aligned CSV.

Signal/time selection:
- Time column is chosen from `time`, `Time`, `timestamp`, or `Timestamp`.
- Sampling rate is estimated as `fs = len(t) / (t[-1] - t[0])`.
- Source channels are detected with fallbacks:
  - ECG: `ECG` or `ACQ_ECG`
  - Respiration: `RSP` or `ACQ_RSP`
  - EDA: `EDA` or `ACQ_EDA`

ECG cleaning (`ECG_clean`):
- 4th-order Butterworth high-pass at `1.0 Hz`
- 4th-order Butterworth low-pass at `100.0 Hz`
- 2nd-order Butterworth band-stop (`59.0–61.0 Hz`) for line-noise suppression
- Filtering uses zero-phase `filtfilt` to reduce phase distortion

RSP cleaning (`RSP_clean`):
- 2nd-order Butterworth band-pass at `0.05–3.0 Hz`
- Also applied with zero-phase `filtfilt`

EDA cleaning (`EDA_clean`):
- Values outside `[5, 40]` are set to `NaN`
- First derivative (`d1`) and second derivative (`d2`) are computed
- Samples with `|d1| > 0.5` or `|d2| > 0.5` are set to `NaN`
- Missing points are linearly interpolated (`limit_direction='both'`)
- Final smoothing uses Savitzky-Golay (`polyorder=3`, `window_length` up to `2001`, auto-adjusted to signal length and odd-window requirement)

Output behavior:
- If filename already ends with `_cleaned`, it is not duplicated
- Otherwise output name appends `_cleaned.csv`
- File is written to `aligned_cleaned_Data/`

## Features

Feature extraction is performed in the `FeatureWindow` path in `pipeline_gui.py`.
It works on marker/event-delimited segments from cleaned CSV data.

### Segment selection modes

1. `Compute Selected Segment Features`
- You choose marker column, start marker, end marker, and segment label
- The segment is sliced between selected marker indices

2. `Compute Features Automatically for Experiment Phases`
- Marker columns are scanned for phase-style events
- Start/end markers are inferred using suffix patterns:
  - start: `_start`, `_begin`, `onset`
  - end: `_end`, `_end_auto`, `_stop`, `offset`
  - reset handling: `_reset` / `...reset`
- Phase types are grouped as `briefing`, `baseline`, `trial`
- Matching start/end pairs produce one feature row per phase segment

### Features computed per segment

Base segment metadata:
- `fs_hz_est`: estimated sampling rate from segment time deltas
- `n_rows`: number of rows in segment
- `seg_start_time`, `seg_end_time`
- `file`, `segment_label`, `marker_col`, `start_marker`, `end_marker`, `start_idx`, `end_idx`

ECG-derived metrics (from `ECG_clean`, if present):
- `mean_hr_bpm`: mean heart rate from NeuroKit ECG rate signal
- `hrv_rmssd_ms`: RMSSD from `nk.hrv_time(...)`
- `hrv_sdnn_ms`: SDNN from `nk.hrv_time(...)`
- `n_rpeaks`: number of detected R-peaks
- Processing uses `nk.ecg_process(..., method='neurokit')`
- If detection fails or too few peaks exist, values are set to `NaN` (and `n_rpeaks=0`)

Respiration-derived metrics (from `RSP_clean`, if present):
- `mean_resp_bpm`: mean respiration rate from `nk.rsp_process(...)`
- `n_breaths`: count of detected respiration peaks
- If processing fails, values default to `NaN` / `0`

Emotion summary features (if OpenFace columns exist):
- Mean value for each of:
  - `emo_Neutral_pct`
  - `emo_Happy_pct`
  - `emo_Sad_pct`
  - `emo_Surprise_pct`
  - `emo_Fear_pct`
  - `emo_Disgust_pct`
  - `emo_Anger_pct`
  - `emo_Contempt_pct`
- Each is computed as a finite-value `nanmean` over the segment

Feature output file:
- Saved to `feature_extracted_Data/<cleaned_stem>_marker_features.csv`
- Selected-segment runs append rows if file already exists
- Auto-phase runs write the computed table for detected phases

## Notes

- Alignment is dropout/flatline-anchor based.
- For irregular XDF streams (`nominal_srate == 0`), nearest-asof merge with tolerance is used.
- `.gitignore` excludes `.csv`, `.acq`, `.xdf`, and Python cache artifacts.

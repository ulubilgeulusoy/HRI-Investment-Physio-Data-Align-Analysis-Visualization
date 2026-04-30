# HRI Investment Physio Data Align / Analysis / Visualization

This repository contains a complete physiological data pipeline for:

1. Aligning BIOPAC `.acq` and LSL/XDF `.xdf` recordings by using the flatline signals at the end of recordings. Note: To be able to have this signal, you need to turn off the Bionomadix transmitter before you stop and save recordings.
2. Cleaning ECG/RSP/EDA channels
3. Computing marker-based HR, HRV, and respiration features
4. Visual quality checks and plotting

You can run this either from the new unified GUI (`pipeline_gui.py`) or from individual scripts.

## Folder Structure

- `raw_xdf_acq_Data/`:
  - Raw `.acq` and `.xdf` files imported/copied from the GUI
- `aligned_Data/`:
  - Alignment outputs (merged CSV after ACQ/XDF synchronization)
- `aligned_cleaned_Data/`:
  - Cleaned CSV outputs (`ECG_clean`, `RSP_clean`, `EDA_clean`)
- `feature_extracted_Data/`:
  - HR/HRV/Respiration + emotion metric CSV outputs

Main scripts:

- `pipeline_gui.py`: unified end-to-end GUI
- `align_acq_xdf_dropout_merge_all_xdf.py`: alignment script
- `clean_physio_csv_folder.py`: signal cleaning script
- `compute_hr_hrv_resp_windows.py`: feature extraction script
- `overlay_check_csv_vs_xdf.py`: overlay check utility

## Requirements

Python 3.9+ recommended.

Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas neurokit2 pyxdf scipy matplotlib
```

## Unified GUI 

Launch:

```powershell
python pipeline_gui.py
```

Startup behavior:
- The GUI opens in a reset/blank state (no auto-filled file paths).
- Use `Load Latest From Folders` only if you want to auto-populate from repository folders.
- ACQ/XDF `Browse` dialogs default to `raw_xdf_acq_Data/`.
- CSV `Browse` defaults to `aligned_cleaned_Data/` (fallback: `aligned_Data/`).

### GUI Sections

1. **Raw import**
   - Select `.acq` and `.xdf`
   - Click `Import/Copy Raw Files`
   - Files are copied into `raw_xdf_acq_Data/`

2. **Pipeline**
   - `Run Alignment`: creates aligned CSV in `aligned_Data/`
     - Includes ACQ columns (`ACQ_*`) and XDF streams (`XDF_*`), including the XDF physio stream
     - Adds `timestamp` and `time_sec` columns in addition to `time`
   - `Run Cleaning`: creates cleaned CSV in `aligned_cleaned_Data/`
   - `Open Feature GUI`: opens marker-based feature extraction window
   - `Run Full Pipeline (Align+Clean)`: runs alignment and cleaning only
   - A progress bar shows stage/percentage while tasks run, and pipeline buttons are temporarily disabled to prevent duplicate runs.

3. **Visualizers**
   - Load any CSV, then plot X-Y or scatter
   - Optional marker-based scope (`Between markers`)

3. **Overlay/Alignment Check**
   - Overlay is in a separate configurable section (BIOPAC-only):
     - Click `Load Overlay Options`
     - Choose BIOPAC-relevant CSV signal (`ECG/RSP/EDA` variants), BIOPAC-like XDF stream, and BIOPAC XDF signal
     - Click `Plot Selected Overlay`
   - `Export Selected Scope CSV`: exports current scoped CSV to `aligned_cleaned_Data/`
   - Large CSV/XDF loading uses background execution with progress updates, so the GUI remains responsive.
   - Both main GUI and Feature GUI are vertically scrollable.

### GUI Controls

- **Alignment options**:
  - `Include extra XDF streams (markers/OpenFace/robot states) in aligned CSV`:
    - ON: aligned CSV includes non-physio XDF streams such as experiment markers, OpenFace outputs, and robot state channels.
    - OFF: aligned CSV excludes those extra non-physio streams, resulting in a smaller/narrower file focused on core aligned signals.
  - Advanced alignment thresholds are intentionally fixed in code (not exposed in GUI) for simpler operation.

- **Feature extraction approach**:
  - Feature extraction is marker-based and done in the separate Feature GUI.
  - When you click `Open Feature GUI`, a small loading window appears first.
  - The Feature GUI opens only after marker loading completes.
  - You can:
    - compute one selected marker segment (`Compute Selected Segment Features`)
    - compute all detected experiment phases (`Compute Features Automatically for Experiment Phases`)
    - load prior results (`Load Previously Calculated CSV`)
  - Feature GUI includes:
    - progress bar during compute
    - an in-window `Computed Feature Rows` table
    - `Load Previously Calculated CSV` button to reload saved feature rows

## Typical GUI Workflow

1. Open GUI: `python pipeline_gui.py`
2. Select raw `.acq` and `.xdf`
3. Click `Import/Copy Raw Files`
4. Click `Run Full Pipeline (Align+Clean)`
5. Click `Open Feature GUI`, load cleaned CSV, and compute features for selected marker segments
6. Review outputs:
   - `aligned_Data/`
   - `aligned_cleaned_Data/`
   - `feature_extracted_Data/`
7. Use visualizer buttons for QA plots

## Script-Only Workflow (Legacy)

If you prefer script-by-script usage:

1. Alignment:

```powershell
python align_acq_xdf_dropout_merge_all_xdf.py
```

2. Cleaning:

```powershell
python clean_physio_csv_folder.py
```

3. Metrics:

```powershell
python compute_hr_hrv_resp_windows.py
```

4. Visual checks:

```powershell
python overlay_check_csv_vs_xdf.py
```

Note: these legacy scripts use in-file path/config constants, while the GUI handles paths and folder routing automatically.

## Output Naming and Routing

- Alignment outputs are generated in `aligned_Data/` with a filename derived from ACQ and XDF stems.
- Cleaning preserves aligned CSV filename and writes to `aligned_cleaned_Data/`.
- Metrics preserves cleaned CSV stem and writes `<stem>_metrics_with_emotions.csv` to `feature_extracted_Data/`.
- Marker-based feature GUI writes/updates `<cleaned_stem>_marker_features.csv` in `feature_extracted_Data/`.
- Repo `.gitignore` excludes all `.csv`, `.acq`, `.xdf`, and Python cache/bytecode artifacts.

## Troubleshooting

- **No flatline found**:
  - Increase `Flat win(s)` and/or `Flat rel` in GUI alignment controls.

- **Overlay loading returns no options**:
  - Ensure CSV contains BIOPAC signal columns (e.g., `ACQ_ECG`, `ACQ_RSP`, `ACQ_EDA`, `ECG_clean`, `RSP_clean`, `EDA_clean`).
  - Ensure the XDF contains a BIOPAC/physio stream.

- **No numeric points in plotting**:
  - Choose numeric X/Y columns and reload CSV columns.

- **Feature row computed but not visible**:
  - Check the `Computed Feature Rows` table in the Feature GUI.
  - Use `Load Saved Feature CSV` to reload existing saved rows.

- **Missing package errors**:
  - Re-check the pip install command in Requirements.

## Notes

- Alignment is dropout/flatline-anchor based.
- For irregular XDF streams (nominal srate `0`), nearest-asof merge with tolerance is used.
- Emotion aggregation uses OpenFace emotion columns when they exist in cleaned/aligned CSV.

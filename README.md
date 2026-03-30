# NTU-Navigator
A smart Nanyang Technological University (NTU) navigator, including localization and path routing.
The app combines:
- graph-based pathfinding (Dijkstra fallback),
- a UVFA-based RL policy model,
- visual place recognition (VPR) from uploaded videos,
- weather-aware routing behavior (rain can block the right stair),
- optional OCR and 360-view expansion for stronger localization.

This README documents **app.py only**.

## Features in app.py

- Interactive route planning UI built with Streamlit.
- Start point selection by:
  - manual room selection, or
  - video localization using VPR.
- Destination selection among fixed mapped locations.
- Weather modes:
  - Live Singapore precipitation API,
  - force rain,
  - force clear.
- Time-aware vertical transition cost (peak/off-peak elevator penalty).
- RL brain management in sidebar:
  - load pretrained model from disk,
  - train a new model and save it.
- VPR database management in sidebar:
  - ingest baseline videos,
  - delete by location,
  - clear database.

## Prerequisites

- Python 3.9+ recommended.
- macOS/Linux/Windows (macOS is supported in code).
- Internet access recommended for:
  - live weather query,
  - first-time DINOv2 model fetch via `torch.hub`.

## Installation

Create and activate your environment first (example with conda):

```bash
conda create -n RL python=3.10 -y
conda activate RL
```

Install required packages:

```bash
pip install streamlit torch torchvision numpy opencv-python pillow plotly requests faiss-cpu
```

Optional packages used by app.py for enhanced localization:

```bash
pip install easyocr py360convert
```

Notes:
- If you have Apple Silicon and want better PyTorch performance, install a wheel variant suitable for your platform.
- `easyocr` is optional; if missing, text extraction gracefully degrades.
- `py360convert` is optional; if missing, 360 expansion falls back to single-view processing.

## Quick Start (app.py)

From repository root:

```bash
streamlit run app.py
```

Then open the URL shown by Streamlit (typically `http://localhost:8501`).

## Required / Important Files for app.py

The app can run with partial functionality even if some files are missing, but these files control key behaviors:

- `app.py`: main application entrypoint.
- `ntu_8d_uvfa_brain.pth`: optional pretrained RL brain loaded from sidebar button.
- `vpr_360_database.index` + `vpr_360_labels.npy`: core VPR index and labels.
- `vpr_360_ocr.npy`: optional OCR metadata for localization refinement.
- `vpr_360_thumb.npy`: optional thumbnail cache (auto-created when ingesting videos).

If VPR files are absent, app.py still starts, but video localization will report database missing/empty until you ingest baseline videos.

## Typical Workflow

1. Start app with `streamlit run app.py`.
2. In sidebar, choose weather mode.
3. (Optional) Load pretrained brain if `ntu_8d_uvfa_brain.pth` exists.
4. (Optional) Upload baseline videos to build/expand VPR library.
5. Select start and destination.
6. Click `Route Me!`.

## Troubleshooting

- App starts but localization fails:
  - Build VPR library first using sidebar video upload.
  - Ensure location names are selected from fixed options.

- First run is slow:
  - `torch.hub` may download DINOv2 weights on first encoder load.

- Weather API unavailable:
  - Switch to `Force Rain` or `Force Clear` mode.

- macOS OpenMP conflict warnings:
  - `app.py` already sets `KMP_DUPLICATE_LIB_OK=TRUE` on Darwin.

## Scope Note

This README intentionally covers only `app.py` and excludes other scripts in this repository.

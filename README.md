# Infer-RainPredRNN2

A small inference utility for running RainPredRNN — a recurrent neural network that predicts future radar reflectivity frames from a sequence of past frames. This repository contains scripts to run a trained model on a stack of georeferenced TIFF frames, measure inference performance, and save the predicted frames.

This README explains the repository layout, how to prepare inputs, how to run inference, and what each file does.

---

## Repository structure

- `.gitignore` — standard ignore file.
- `LICENSE` — license for the repository (please read before reuse).
- `README.md` — this file.
- `extract_tiff.py` — helper script to extract/prepare TIFF frames (expected to create the input stack). See notes below.
- `infer.py` — main inference script (includes configuration, model loading, timing, normalization and saving predictions).
- `infer.sbatch` — example Slurm batch script to run the inference on a cluster (adjust paths and parameters before use).

---

## Overview of the inference workflow

1. Prepare a directory with 18 input TIFF frames (single-band radar reflectivity frames). The inference script expects exactly 18 frames as input.
2. Load a trained model checkpoint (`best_model.pth`) and the model definition (`RainPredRNN`) implemented in `app9.py`.
3. Run the model to predict the next PRED_LENGTH frames (commonly 6).
4. Save each predicted frame as a TIFF (or TIFF-like image) in the specified output directory.
5. The script also measures execution timings (mean, std, p50, p95) across multiple repeats and prints predictions-per-second.

---

## Important files detailed

### infer.py
- Purpose: load a trained RainPredRNN model, run inference on a prepared 18-frame sequence and save predicted frames.
- Key behaviors and configuration (at top of `infer.py`):
  - `CKPT_PATH` — path to the PyTorch checkpoint `.pth` that contains `'model_state_dict'`.
  - `INPUT_DIR` — directory containing the 18 TIFF/TIFF files (the script checks there are exactly 18).
  - `OUTPUT_DIR` — directory where predicted frames will be saved.
  - `REPEATS` — number of repeated forward passes used for measuring timing.
  - `WARMUP` — number of warm-up forward passes to stabilize timings.
  - `USE_AMP` — whether to use mixed precision (autocast) on CUDA.
  - `RESIZE_HW` — (height, width) used to resize input frames (default is 224×224 to match training).
  - `STEP_MIN` — temporal step in minutes used to generate inferred timestamps for outputs (used if input filenames contain timestamps).
- Input format:
  - The script looks for files `*.tif` and `*.tiff` inside `INPUT_DIR`.
  - It expects exactly 18 files (the typical 18-frame input used by the model).
  - Each file is read with `rasterio` and only the first band is used.
  - Pixel normalization steps:
    - NaN / inf values are coerced to 0.
    - Values clipped to [0, +inf], then further clipped to [0, 70] dBZ (configurable inside the function).
    - Rescaled to [0,1], converted to 8-bit image, resized to `RESIZE_HW`, and re-normalized to [-1,1] with mean=0.5 and std=0.5 used by the model.
- Model:
  - The script imports `RainPredRNN` and `PRED_LENGTH` from `app9.py`.
  - You must have `app9.py` present in the same directory (or make `RainPredRNN` and `PRED_LENGTH` available in `PYTHONPATH`).
  - Model weights are loaded from a checkpoint dictionary containing `'model_state_dict'`.
- Inference:
  - Device selection is automatic (`cuda` if available, else `cpu`).
  - Files are converted into a tensor of shape (B=1, T=18, C=1, H, W) and fed to the model.
  - Warm-up iterations are run then `REPEATS` measurements are taken to compute timing statistics.
  - Predictions are output by the model in the range [-1,1] and converted to [0,1] and then saved as 8-bit frames.
- Output naming:
  - The script attempts to preserve timestamp style when generating prediction filenames. It uses a regex pattern to detect timestamps of the form `YYYYmmddZHHMM` in the last input filename (pattern: `.*_(\d{8}Z\d{4})_.*`).
  - If detected, it will generate output names incrementing that timestamp by `STEP_MIN` per predicted frame.
  - If not detected, it will fall back to generic names like `stem_t+01_pred.tiff`, etc.
- Metrics printed:
  - Mean time, standard deviation, median (p50), p95, and predictions-per-second (fps).

### extract_tiff.py
- Purpose: helper script to extract and/or prepare TIFF frames for inference.
- Behavior: the repository includes `extract_tiff.py` as a convenience tool to prepare the 18 TIFF files expected by `infer.py`. Typical uses:
  - Convert radar data files (NetCDF, HDF, or other archive formats) into single-band TIFF frames.
  - Crop/resample/extract areas of interest.
  - Save the resulting images into a directory that can be used as `INPUT_DIR` for `infer.py`.
- Notes:
  - The exact inputs/outputs and usage depend on your local radar data format. Check the script's top-level docstring or inline comments for usage details, required libraries and example commands.
  - If you need help adapting the script to your data, provide a small example of your dataset and filename pattern and the script can be adjusted.

### infer.sbatch
- Purpose: batch job script for Slurm-based clusters.
- Usage: adapt the paths to `python`, `CKPT_PATH`, and `INPUT_DIR`, then submit with:
  - sbatch infer.sbatch

---

## Dependencies

At minimum, the following Python packages are required (versions are indicative — use compatible versions from your training environment):

- Python 3.8+
- numpy
- Pillow (PIL)
- torch (PyTorch) — same major version used during training (GPU version if using CUDA)
- rasterio
- (optionally) other dependencies used by `app9.py` (e.g., custom layers or utils)

Install via pip (example):
pip install numpy pillow torch rasterio

Note: For GPU inference, install the appropriate CUDA-enabled PyTorch build from https://pytorch.org.

---

## Quick start: run inference locally

1. Place your 18 prepared single-band TIFF frames in a directory, e.g. `/path/to/extracted_tiff`.
   - Ensure they are ordered so that the last file corresponds to the most recent observed timestamp.
2. Put the trained checkpoint at `CKPT_PATH`.
3. Ensure `app9.py` (which defines `RainPredRNN` and `PRED_LENGTH`) is present alongside `infer.py`.
4. Edit the top of `infer.py` to point `CKPT_PATH`, `INPUT_DIR`, and `OUTPUT_DIR` to desired locations (or change them via environment management).
5. Run:
   python infer.py
6. Predicted frames will be saved in `OUTPUT_DIR` and timing metrics printed.

---

## Example input/output filename behavior

- If your last input frame is named like:
  mysite_radar_20251008Z0110_tileX.tiff
  the script will attempt to produce outputs:
  mysite_radar_20251008Z0120_tileX_pred.tiff
  mysite_radar_20251008Z0130_tileX_pred.tiff
  ...
  (incrementing by `STEP_MIN` minutes; adjust `STEP_MIN` as needed)
- If the timestamp pattern is not matched, output files will be named:
  <input_stem>_t+01_pred.tiff, <input_stem>_t+02_pred.tiff, ...

---

## Tips & troubleshooting

- If `infer.py` raises “Attesi 18 TIFF ... trovati N” make sure the input directory contains exactly 18 TIFF/TIF files (no hidden files, no other formats).
- If you get model loading errors:
  - Verify `CKPT_PATH` points to a dictionary containing `'model_state_dict'`.
  - Confirm `RainPredRNN` definition in `app9.py` matches the architecture used during training.
- For faster GPU runs:
  - Ensure `USE_AMP = True` and CUDA is available.
  - Confirm CuDNN benchmarking is enabled by the script for repeated runs.
- If predictions look visually incorrect:
  - Check the normalization pipeline in `infer.py` matches preprocessing used during training (value clipping, resize, normalization to [-1,1]).
  - Validate that input data units (e.g., dBZ) match training data units.

---

## Extending or adapting the repo

- To support batch inference on multiple scenes:
  - Wrap `load_18_stack()` and the inference loop in a loop over multiple input directories.
  - Or create a small coordinator script that finds multiple 18-frame folders and processes them sequentially.
- To save GeoTIFFs preserving georeference:
  - Replace the `PIL.Image` saving with `rasterio` and copy geospatial metadata from input sources (the current script writes raw image files without geospatial metadata).
- To change the number of predicted frames:
  - `PRED_LENGTH` is imported from `app9.py`. If you want a different prediction horizon, update the model/training to support that horizon and adjust `PRED_LENGTH` accordingly.

---

## License

See the `LICENSE` file in the repository root for license terms.

---

If you want, I can:
- adapt `extract_tiff.py` to a specific radar input format (NetCDF, HDF5, PNG, etc.) if you share a sample filename or sample files;
- update `infer.py` to preserve georeferencing in output TIFFs using rasterio;
- produce a parameterized CLI wrapper (argparse) for `infer.py` so you can pass CKPT, input and output paths from the command line.

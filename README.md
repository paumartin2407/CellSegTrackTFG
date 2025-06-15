# CellSegTrack – Cell Segmentation & Tracking Pipeline (TFG)

This repository contains the code and resources for the Final Degree Project (TFG): **CellSegTrack**—a modular pipeline for:

1. **Training a custom Cellpose model** on cell microscopy images.  
2. **Performing segmentation and tracking** on time‑lapse cell videos.

---

## 🧠 Features

### 1. Train Cellpose Model

Train a custom Cellpose model using your own dataset:

```bash
python train_model_main.py <model_name>
```

- Inputs images in `cellpose_model_training/pretrain`
- Applies data augmentation (`data_augmentation.py`)
- Trains using `cellpose_training.py`
- Outputs the model in the same directory  
- A pretrained model is already provided in `cellpose_model_training/`

---

### 2. Segmentation & Tracking Pipeline

Run preprocessing, segmentation, and/or tracking on `.tif` videos:

```bash
python main.py [--pre] [--seg] [--track] [--all] \
    [--green-index N] [--red-index M] \
    [--search-radius R] [--frames-window W] \
    <video_name>
```

- **Preprocessing** (`--pre` / `--all`):  
  Extract frames and separate channels (script: `preprocessing/tif_slicer_green.py`).

- **Segmentation** (`--seg` / `--all`):  
  Apply the Cellpose model to segment cells (script: `segmentation/CellposeTrainedModel.py`).  
  Generates overlay plots with `plotting/plots.py`.

- **Tracking** (`--track` / `--all`):  
  Perform backward tracking on segmented masks (script: `tracking/track_assembler.py`).  
  Plots resulting tracks and lineage descriptors via `plotting/plots.py`.

#### `main.py` Highlights

- Flags for each step and chaining options.
- Default parameters:  
  `green-index = 1`, `red-index = 0`, `search-radius = 35 px`, `frames-window = 3 frames`.
- Auto‑detects total frames for tracking termination.

Example usages:

```bash
# Full pipeline:
python main.py my_video --all

# Only segmentation & tracking:
python main.py my_video --seg --track

# Tracking only, custom params:
python main.py my_video \
    --track --search-radius 50 --frames-window 5
```

---

## 🗑 Cleanup Tool

Remove all outputs associated with a specific video:

```bash
python delete.py <video_name>
```

---

## 📂 Repository Structure

```
.
├── cellpose_model_training/
│   ├── data_augmentation.py
│   ├── cellpose_training.py
│   └── <trained_model_directory>/
│
├── preprocessing/
│   └── tif_slicer_green.py
│
├── segmentation/
│   └── CellposeTrainedModel.py
│
├── tracking/
│   └── track_assembler.py
│
├── plotting/
│   └── plots.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── masks/
│   └── results/
│
├── train_model_main.py
├── main.py
├── delete.py
└── README.md
```

---

## 🔧 Installation & Dependencies

Install required Python packages (tested on Python ≥ 3.8):

```bash
pip install -r requirements.txt
```

Key dependencies:

- `cellpose`
- `numpy`, `scipy`, `scikit-image`
- `matplotlib`, `argparse`

---

## 🎯 Usage Examples

```bash
# Train a model
python train_model_main.py my_cell_model

# Run the full pipeline
python main.py sample_video --all

# Segmentation only
python main.py sample_video --seg

# Custom tracking
python main.py sample_video \
    --track --search-radius 50 --frames-window 5

# Clean data for a video
python delete.py sample_video
```

---

## 🙋‍♂️ Author

**Pablo Martín Berná**  
Dual Degree in Mathematics & Computer Engineering, Applied Math Department, Universidad de Sevilla

---
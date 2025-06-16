# CellSegTrack – Cell Segmentation & Tracking Pipeline (TFG)

This repository hosts the code and resources for the Final Degree Project (**TFG**) entitled **CellSegTrack**—a modular pipeline that supports:

1. **Training a custom Cellpose model** on microscopy images.  
2. **Performing segmentation and tracking** on time‑lapse cell videos.

---

## 🧠 Features

### 1. Train a Cellpose Model

Train your own Cellpose model with a single command:

```bash
python train_model_main.py <model_name>
```

* **Input:** images placed in `cellpose_model_training/pretrain`  
* **Data augmentation:** handled by `data_augmentation.py`  
* **Training:** executed by `cellpose_training.py`  
* **Output:** the trained model is saved inside the same folder  
* A **pretrained model** trained specifically on our dataset and used for the tracking and video analyses in this project is included for convenience; however, we recommend retraining your own model or using one of the official Cellpose models if your data differ.

---

### 2. Segmentation & Tracking Pipeline

Run preprocessing, segmentation and/or tracking on a `.tif` video:

```bash
python main.py [--pre] [--seg] [--track] [--all] \
    [--green-index N] [--red-index M] \
    [--search-radius R] [--frames-window W] \
    <video_name>
```

---

## ⚙️ Execution Parameters

`main.py` exposes two sets of options:

1. **Pipeline parameters** – decide which stage(s) to run.  
2. **Algorithm parameters** – fine‑tune how the processing itself works.

### 1) Pipeline parameters
| Flag | Purpose | Script(s) involved |
|------|---------|--------------------|
| `--pre` | **Pre‑processing** – splits the TIFF into individual frames and separates red/green channels. | `preprocessing/tif_slicer_green.py` |
| `--seg` | **Segmentation** – applies the trained Cellpose model to the green frames and produces mask overlays. | `segmentation/CellposeTrainedModel.py`, `plotting/plots.py` |
| `--track` | **Tracking** – reconstructs cell trajectories backward in time and plots the results. | `tracking/track_assembler.py`, `plotting/plots.py` |
| `--all` | Runs *pre‑processing → segmentation → tracking* in sequence. | all of the above |

### 2) Algorithm parameters
| Flag | Default | Description |
|------|---------|-------------|
| `--green-index <int>` | `1` | Zero‑based index for the green channel in the TIFF (used during pre‑processing). |
| `--red-index <int>`   | `0` | Zero‑based index for the red channel in the TIFF (used during pre‑processing). |
| `--search-radius <int>` | `35` px | Pixel radius used to match masks between consecutive frames during tracking. |
| `--frames-window <int>` | `3` frames | Temporal window (how many frames back) inspected when searching for correspondences. |

---

### Default values recap

```text
green-index   = 1
red-index     = 0
search-radius = 35  # pixels
frames-window = 3   # frames
```

---

### Example commands

```bash
# Full pipeline
python main.py HeLa_01 --all

# Segmentation + Tracking only
python main.py HeLa_01 --seg --track

# Tracking only with custom parameters
python main.py HeLa_01 \
    --track --search-radius 50 --frames-window 5
```

---

### 🗑 Clean‑up tool

Remove every output generated for a specific video:

```bash
python delete.py <video_name>
```

---

## 📂 Repository Structure

```
.
+-- cellpose_model_training/
|   +-- data_augmentation.py
|   +-- cellpose_training.py
|   +-- <trained_model>/
|
+-- preprocessing/
|   +-- tif_slicer_green.py
|
+-- segmentation/
|   +-- CellposeTrainedModel.py
|
+-- tracking/
|   +-- cell_track.py
|   +-- mincostflow_graph_builder.py
|   +-- mincostflow_solver.py
|   +-- track_assembler.py
|
+-- plotting/
|   +-- plots.py
|
+-- data/
|   +-- raw/          # original videos (.tif)
|   +-- processed/    # extracted frames & channels
|   +-- masks/        # segmentation masks
|   +-- results/      # tracks & figures
|
+-- train_model_main.py
+-- main.py
+-- delete.py
\-- README.md
```

---

## 🔧 Installation

Tested with **Python ≥ 3.8**. Install all dependencies with:

```bash
pip install -r requirements.txt
```

Key libraries include: `cellpose`, `numpy`, `scipy`, `scikit-image`, `matplotlib`, `argparse`.

---

## 🙋‍♂️ Author

**Pablo Martín Berná**  
Dual Degree in Mathematics & Computer Engineering, Applied Math Department, Universidad de Sevilla

---

from __future__ import annotations
import os
import cv2
from skimage.io import imread, imsave
from cellpose import plot
from glob import glob
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
from collections import defaultdict


def natural_keys(text):
    """
    Helper function to sort file names numerically.
    """
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def plot_mask_overlays(video_name):
    """
    Plot and save mask overlays for a given video.

    Args:
        video_name (str): Name of the video without extension (e.g., 'exp0').

    Loads frames from:
        data/processed/<video_name>/
    Loads masks from:
        data/processed/<video_name>_masks/
    Saves overlays into:
        data/processed/<video_name>_masks_plots/
    """

    frames_folder = os.path.join("data", "processed", video_name, "green_channel")
    masks_folder = os.path.join("data", "masks", video_name)
    overlays_folder = os.path.join("data", "plots", video_name, "masks")

    os.makedirs(overlays_folder, exist_ok=True)

    # Get sorted list of frame and mask files
    frame_files = sorted(glob(os.path.join(frames_folder, "frame_*.tif")), key=natural_keys)
    mask_files = sorted(glob(os.path.join(masks_folder, "frame_*_masks.tif")), key=natural_keys)

    if len(frame_files) != len(mask_files):
        print("Warning: Number of frames and masks do not match exactly.")

    j = 0

    for frame_path, mask_path in zip(frame_files, mask_files):
        frame = imread(frame_path)
        mask = imread(mask_path)

        #if frame.dtype != np.uint8:
        #    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        overlay = plot.mask_overlay(frame, mask)

        # Save the overlay
        frame_base = os.path.splitext(os.path.basename(frame_path))[0]
        overlay_path = os.path.join(overlays_folder, f"{frame_base}_masks.png")
        imsave(overlay_path, overlay, check_contrast=False)

        j += 1
    
    print(f"Plotted {j} mask files for {video_name}.")


def load_tracks(path):
    """Return a dict {label: {start, end, parent}} from a track file."""
    tracks = {}
    for line_no, line in enumerate(Path(path).open(), 1):
        if not line.strip():
            continue
        try:
            label, begin, end, parent = map(int, line.split())
        except ValueError as err:
            raise ValueError(
                f"Line {line_no}: expected four space‑separated integers"
            ) from err
        tracks[label] = dict(start=begin, end=end, parent=parent)
    if not tracks:
        raise ValueError("The file is empty or malformed.")
    return tracks


def _compute_children(tracks):
    """Return a dict parent_label -> [child_labels]."""
    children = defaultdict(list)
    for lbl, info in tracks.items():
        p = info["parent"]
        if p:
            children[p].append(lbl)
    return children


def tidy_columns(tracks, spacing_between_roots = 1):
    """Assign an X coordinate to every track so that none overlap.

    * Continuity (one child)   → child keeps the parent's X.
    * Mitosis   (two or more) → children spread left/right, parent is centred.
    * Independent roots are spaced by ``spacing_between_roots`` columns.
    """
    children = _compute_children(tracks)
    x_pos = {}
    next_leaf_x = 0

    def dfs(node):
        nonlocal next_leaf_x
        kids = children.get(node, [])
        if not kids:                       # leaf
            x_pos[node] = next_leaf_x
            next_leaf_x += 1
        elif len(kids) == 1:               # continuity
            dfs(kids[0])
            x_pos[node] = x_pos[kids[0]]
        else:                              # mitosis ( ≥ 2 children )
            for k in sorted(kids):
                dfs(k)
            left  = min(x_pos[k] for k in kids)
            right = max(x_pos[k] for k in kids)
            x_pos[node] = (left + right) / 2.0

    roots = [l for l, info in tracks.items() if info["parent"] == 0]
    for r in sorted(roots, key=lambda l: tracks[l]["start"]):
        dfs(r)
        next_leaf_x += spacing_between_roots

    return x_pos


def plot_tracks(video_name, major_step=5, legend_text=None):
    """
    Draw the tree plot and save it to disk (no GUI window).

    Parameters
    ----------
    video_name : str
        Folder / file stem for this video.
    major_step : int, optional
        Interval for the bold y-ticks.  Default is 5.
    """

    # ------------------------------------------------------------------- paths
    plots_folder = os.path.join("data", "plots", video_name)
    os.makedirs(plots_folder, exist_ok=True)

    tracks_path = os.path.join(
        "data", "results", video_name , f"tracks_{video_name}.txt"
    )

    # ------------------------------------------------------------------- data
    tracks  = load_tracks(tracks_path)
    columns = tidy_columns(tracks)

    vert_segments, dash_segments = [], []
    node_x, node_y = [], []

    for lbl, tr in tracks.items():
        x = columns[lbl]
        vert_segments.append([(x, tr["start"]), (x, tr["end"])])
        node_x.extend([x] * (tr["end"] - tr["start"] + 1))
        node_y.extend(range(tr["start"], tr["end"] + 1))

        if tr["parent"]:
            p = tr["parent"]
            dash_segments.append([(columns[p], tracks[p]["end"]), (x, tr["start"])])
    
    # ----------------------------------------------------------------- figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.add_collection(LineCollection(vert_segments, colors="k", lw=1))
    ax.add_collection(LineCollection(dash_segments, colors="k", lw=0.6,
                                     linestyles="dashed"))
    ax.scatter(node_x, node_y, s=4, color="k", rasterized=True, zorder=3)

    t_min, t_max = min(node_y), max(node_y)
    ax.set_yticks(range(t_min, t_max + 1), minor=True)
    ax.set_yticks(range(t_min, t_max + 1, major_step))
    ax.grid(which="both", axis="y", color="gray", lw=0.2)
    ax.set_ylim(t_max + 0.5, t_min - 0.5)
    ax.set_xticks([])
    ax.set_ylabel("Time (frames)")
    ax.set_title("Track trees (non-overlapping X)")
    if legend_text:
        fig.text(0.05, -0.05, legend_text, ha='left', va='top', fontsize='medium', wrap=True)
    plt.tight_layout()

    # ----------------------------------------------------------- save & close
    out_path = os.path.join(
        plots_folder, f"tracks_tree.png"
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)          # ensures no GUI window opens and frees memory
    print(f"Tracks Tree plotted for {video_name}")


def plot_saved_lineage_descriptors(video_name, final_frame, descriptor, lineages,
                                   y_max=1000):
    """
    Read the lineage descriptor sequences produced by
        lineages_<descriptor>_intensities.txt
    and save one plot per lineage to

        data/plots/<video_name>/intensities_<descriptor>/lineage_<leaf>.png

    • X-axis: sample index (0 … N-1) because absolute frame numbers were not
      stored in the file.  The original *final_frame* is kept only to set x-limits
      for visual comparability.
    • Y-axis: descriptor value (clip shown range with *y_max* if desired).
    """

    # -------- folders -------------------------------------------------- #
    res_dir   = os.path.join("data", "results", video_name)
    plots_dir = os.path.join("data", "plots", video_name,
                             f"intensities_{descriptor}")
    os.makedirs(plots_dir, exist_ok=True)

    in_path = os.path.join(
        res_dir, f"lineages_{descriptor}_intensities_{video_name}.txt"
    )

    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Missing file: {in_path}")

    n_plotted = 0
    with open(in_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            # ---- parse "<leaf>: v0 v1 v2 …" --------------------------- #
            leaf_str, values_str = line.split(":", 1)
            leaf_mask_value = leaf_str.strip()
            vals = np.array(
                [float(v) if v.lower() != "nan" else np.nan
                 for v in values_str.strip().split()]
            )

            if len(vals) < 150:            # emulate the old min-length filter
                continue

            if int(leaf_str) not in lineages:
                continue

            # ---- make the plot --------------------------------------- #
            x = np.arange(len(vals))
            plt.figure(figsize=(10, 5))
            plt.plot(x, vals, linewidth=1)
            plt.xlabel("Sample index")
            plt.ylabel(f"{descriptor} red-channel intensity")
            plt.title(f"Lineage leaf mask={leaf_mask_value}  ({video_name})")
            plt.xlim(0, final_frame)
            plt.ylim(0, y_max)
            plt.tight_layout()

            out_name = f"lineage_{leaf_mask_value}.png"
            plt.savefig(os.path.join(plots_dir, out_name), bbox_inches="tight")
            plt.close()
            n_plotted += 1

    print(f"Plotted {n_plotted} lineage traces to {plots_dir}")
    

def plot_lineage_descriptors(video_name, final_frame, descriptor):
    """
    Plot mean red-channel intensity for every leaf lineage and save
    each figure in data/plots/<video_name>/intensities/.
    """
    results_dir      = os.path.join("data", "results", video_name)
    plots_folder     = os.path.join("data", "plots")
    intensities_folder = os.path.join(plots_folder, video_name, f"intensities_{descriptor}")
    os.makedirs(intensities_folder, exist_ok=True)

    # 1) Load metadata and intensity values
    tracks_df = pd.read_csv(
        os.path.join(results_dir, f"tracks_{video_name}.txt"),
        sep=r"\s+",
        header=None,
        names=["label", "start", "end", "parent"],
        dtype={"label": int, "start": int, "end": str, "parent": int},
    )
    desc_df = pd.read_csv(
        os.path.join(results_dir, f"intensities_{descriptor}_{video_name}.txt"),
        sep=r"\s+",
        header=None,
        names=["label", "frame", "intensity"],
        dtype={"label": int, "frame": int, "intensity": float},
    )

    masks_df = pd.read_csv(
        os.path.join(results_dir, f"masks_{video_name}.txt"),
        sep=r"\s+",
        header=None,
        names=["label", "frame", "mask_value"],
        dtype={"label": int, "frame": int, "mask_value": int},
    )
    # Dictionary: label → mask_value at the last frame where that label appears
    last_mask_of = (
        masks_df.sort_values("frame")
                .groupby("label")["mask_value"]
                .last()
                .to_dict()
    )

    # Quick look-ups
    children_of = tracks_df.groupby("parent")["label"].apply(list).to_dict()
    parent_of   = tracks_df.set_index("label")["parent"].to_dict()

    all_labels  = set(tracks_df["label"])
    leaves      = sorted(all_labels - (set(children_of) & all_labels))

    def path_root_to_leaf(leaf):
        path = [leaf]
        while parent_of[path[-1]] != 0:   # 0 ⇒ no parent (root)
            path.append(parent_of[path[-1]])
        return list(reversed(path))       # [root, …, leaf]

    j = 0

    # 2) Generate and save one plot per leaf lineage
    for leaf in leaves:
        chain = path_root_to_leaf(leaf)
        sub   = desc_df[desc_df["label"].isin(chain)].sort_values("frame")

        if len(sub) <= 150:                # skip very short traces
            continue


        # ------------------------------------------------------------------ #
        # Interpolate zeros → treat as NaN, then linear-interpolate           #
        # ------------------------------------------------------------------ #
        series = (
            sub.set_index("frame")["intensity"]
            .replace(0, np.nan)                 # 0 ⇒ missing
            .interpolate(method="linear")       # internal gaps
            .bfill()                            # leading NaNs
            .ffill()                            # trailing NaNs
        )

        plt.figure(figsize=(10, 5))
        plt.plot(series.index, series.values)
        last_mask_value = last_mask_of.get(leaf, "NA")
        plt.title(f"Lineage root={chain[0]} → leaf={last_mask_value}   ({video_name})")
        plt.xlabel("Frame")
        plt.ylabel(f"{descriptor} red-channel intensity")
        plt.xlim(0, final_frame)
        plt.ylim(0, 1000)

        plt.tight_layout()

        out_file = f"lineage_{chain[0]}_{leaf}.png"
        out_path = os.path.join(intensities_folder, out_file)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()                       # prevents any display

        j += 1
    
    print(f"Plotted {j} lineage intensities for {video_name}")




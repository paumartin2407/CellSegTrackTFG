import os
import warnings, re

# ─────────────────────────────────────────────────────────────────────────────
# Silence PyTorch’s “weights_only=False” FutureWarning
# ---------------------------------------------------------------------------
# Context:
#   • Cellpose (≤ v3.0.10 and even current 4.x snapshots) loads its model with
#       torch.load(path)                      # weights_only=False (default)
#   • Since PyTorch 2.2, that call emits
#       FutureWarning: You are using `torch.load` with `weights_only=False` …
#
# The warning is harmless for trusted checkpoints but floods the console every
# run.  Until Cellpose switches to `weights_only=True`, we suppress *just* this
# one line while leaving every other FutureWarning visible.
#
# How it works:
#   • `append=False` inserts the rule at position 0 so it outranks defaults.
#   • `message=re.escape(…)` matches the warning text literally (no regex
#     surprises from dots or backticks).
#   • No `module=` filter → covers the line no matter where PyTorch raises it.
#
# Remove this block once Cellpose merges the fix (check their release notes).
# ─────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings(
    "ignore",
    message=re.escape("You are using `torch.load` with `weights_only=False`"),
    category=FutureWarning,
    append=False      
)

from preprocessing.tif_slicer_green import preprocess_video
from segmentation.CellposeTrainedModel import compute_masks
from tracking.track_assembler import init_tracks_links, tracking_recursive, save_tracks_and_positions
from plotting.plots import plot_mask_overlays, plot_tracks, plot_lineage_descriptors
import argparse

def run_preprocessing(video_name, green_idx, red_idx):
    preprocess_video(video_name, green_index=green_idx, red_index=red_idx)

def run_segmentation(video_name, model_path="cellpose_model_training/oCyto400x400"):
    compute_masks(video_name=video_name, model_path=model_path)
    plot_mask_overlays(video_name)


def run_tracking(video_name, end_frame, search_radius=35, frames_window=3):
    processed_base_path = "data/processed"
    masks_base_path = "data/masks"
    results_base_path = "data/results"

    frames_folder = os.path.join(processed_base_path, video_name, 'green_channel')
    masks_folder = os.path.join(masks_base_path, video_name)

    start_frame = 1

    tracks, links = init_tracks_links(masks_folder, start_frame)
    tracks = tracking_recursive(masks_folder, frames_folder, start_frame, end_frame, tracks, links, search_radius, frames_window)
    save_tracks_and_positions(tracks, video_name, results_base_path)

    plot_tracks(video_name, legend_text=f"Video: {video_name}")


def main():

    parser = argparse.ArgumentParser(description="Modular Cell-Tracking Pipeline")
    parser.add_argument("video_name", help="Base name of the .tif in data/raw/")

    # Channel indices only affect preprocessing
    parser.add_argument(
        "--green-index", type=int, default=1,
        help="Green-channel index in the TIFF (default: 1)",
    )
    parser.add_argument(
        "--red-index", type=int, default=0,
        help="Red-channel index in the TIFF (default: 0)",
    )

    parser.add_argument(
        "--search-radius",
        type=int,
        default=35,
        help="Search radius used during tracking (default: 35)",
    )

    parser.add_argument(
        "--frames-window",
        type=int,
        default=3,
        help="Frames window used during tracking (default: 3)",
    )

    parser.add_argument("--pre", action="store_true", help="Run preprocessing")
    parser.add_argument("--seg", action="store_true", help="Run segmentation")
    parser.add_argument("--track", action="store_true", help="Run tracking")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline")

    args = parser.parse_args()
    video_name = args.video_name
    
    if args.all or args.pre:
        print(f">> Preprocessing (green={args.green_index}, red={args.red_index}) …")
        run_preprocessing(video_name, args.green_index, args.red_index)
    
    frames_folder = os.path.join("data/processed", video_name, 'green_channel')
    end_frame = len([f for f in os.listdir(frames_folder) if f.endswith('.tif')])

    if args.all or args.seg:
        print(">> Segmenting...")
        run_segmentation(video_name)

    if args.all or args.track:
        print(f">> Tracking (radius = {args.search_radius}, frames window = {args.frames_window}) …")
        run_tracking(video_name, end_frame, args.search_radius, args.frames_window)

    print("Done.")

if __name__ == "__main__":
    main()
import os
import numpy as np
import cv2
from .mincostflow_graph_builder import compute_centroids, build_active_centroids, compute_edges_costs_capacities
from .mincostflow_solver import solve_cell_tracking_flow
from .cell_track import CellTrack

def init_tracks_links(masks_path, start_frame):
    """
    Initialize CellTrack objects and links from the first mask.

    Args:
        masks_path (str): Path to folder with masks.
        start_frame (int): Frame index to start tracking.

    Returns:
        list: List of initialized CellTrack objects.
        dict: Dictionary mapping label -> track label.
    """
    tracks = []
    links = {}

    # Load initial mask
    mask_start_path = os.path.join(masks_path, f"frame_{start_frame}_masks.tif")
    mask_start = cv2.imread(mask_start_path, cv2.IMREAD_UNCHANGED)

    unique_labels = np.unique(mask_start)
    unique_labels = unique_labels[unique_labels > 0]

    centroids = compute_centroids(mask_start)

    # Initialize tracks
    for label in unique_labels:
        track = CellTrack(label, start_frame)
        track.positions.append(centroids[label])
        track.masks.append(label)
        tracks.append(track)
        links[label] = label

    return tracks, links

def update_tracks_links(tracks, links, flow, current_frame, centroids_t1, max_gap_frames):
    """
    Update tracks and links based on flow solution for current frame transition.

    Args:
        tracks (list): Current list of CellTrack objects.
        links (dict): Current links dictionary.
        flow (dict): Optimal flow solution (cvxpy variables).
        current_frame (int): Current frame index.
        centroids_t1 (dict): Centroids at time t+1.

    Returns:
        list: Updated tracks.
        dict: Updated links.
    """
    links_updated = {}

    for (u, v), f in flow.items():
        if f.value > 0:

            if u.startswith("L") and v.startswith("R"):
                # Regular match: cell continues
                i = int(u[1:])
                j = int(v[1:])
                links_updated[j] = links.get(i, i)
                for track in tracks:
                    if track.label == links.get(i, i):
                        track.positions.append(centroids_t1[j])
                        track.masks.append(j)
                        track.missing_frames = 0 

            elif u.startswith("L") and v == "D":
                # Disappearance
                i = int(u[1:])
                for track in tracks:
                    if track.label == links.get(i, i):
                        track.missing_frames += 1
                        track.positions.append((0, 0))
                        track.masks.append(0)
                        if track.missing_frames >= max_gap_frames:
                            track.end_frame = current_frame - max_gap_frames

            elif u.startswith("L") and v.startswith("S_"):
                # Division (split)
                parts = v.split("_")
                i = int(parts[1])
                j = int(parts[2])
                k = int(parts[3])

                # Terminate parent track
                for track in tracks:
                    if track.label == links.get(i, i):
                        track.end_frame = current_frame

                # Create daughter tracks
                for child_j in [j, k]:
                    new_track = CellTrack(len(tracks) + 1, current_frame + 1, parent_label=links.get(i, i))
                    new_track.positions.append(centroids_t1[child_j])
                    new_track.masks.append(child_j)
                    tracks.append(new_track)
                    links_updated[child_j] = new_track.label

            elif u == "A" and v.startswith("R"):
                # New appearance
                j = int(v[1:])
                new_track = CellTrack(len(tracks) + 1, current_frame + 1)
                new_track.positions.append(centroids_t1[j])
                new_track.masks.append(j)
                tracks.append(new_track)
                links_updated[j] = new_track.label

    return tracks, links_updated

def tracking_recursive(masks_path, frames_path, current_frame, end_frame, tracks, links, search_radius, max_gap_frames, centroids_t=None):
    """
    Recursive tracking across frames.

    Args:
        masks_path (str): Path to masks.
        frames_path (str): Path to frames.
        current_frame (int): Current frame index.
        end_frame (int): Final frame index.
        tracks (list): Current CellTracks.
        links (dict): Current links.
        max_gap_frames (int): Temporal gap for track termination.
        centroids_t (dict, optional): Centroids at time t.

    Returns:
        list: Final list of tracks.
    """
    if current_frame == end_frame:
        for track in tracks:
            if track.end_frame is None:
                track.end_frame = end_frame
        return tracks
    
    else:
        # Load images and masks
        mask_t = cv2.imread(os.path.join(masks_path, f"frame_{current_frame}_masks.tif"), cv2.IMREAD_UNCHANGED)
        mask_t1 = cv2.imread(os.path.join(masks_path, f"frame_{current_frame + 1}_masks.tif"), cv2.IMREAD_UNCHANGED)

        frame_t = cv2.imread(os.path.join(frames_path, f"frame_{current_frame}.tif"), cv2.IMREAD_UNCHANGED)
        frame_t1 = cv2.imread(os.path.join(frames_path, f"frame_{current_frame + 1}.tif"), cv2.IMREAD_UNCHANGED)

        # Normalize
        frame_t = cv2.normalize(frame_t, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame_t1 = cv2.normalize(frame_t1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        centroids_t1 = compute_centroids(mask_t1)
        if centroids_t is None:
            centroids_t = compute_centroids(mask_t)
        mask_t = cv2.normalize(mask_t, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        active_orig, virtual_links = build_active_centroids(tracks, centroids_t, links, max_gap_frames)

        costs, capacities = compute_edges_costs_capacities(active_orig, centroids_t1, search_radius)
        flow = solve_cell_tracking_flow(len(active_orig), len(centroids_t1), capacities, costs)

        tracks, links = update_tracks_links(tracks, virtual_links, flow, current_frame, centroids_t1, max_gap_frames)

        return tracking_recursive(masks_path, frames_path, current_frame + 1, end_frame, tracks, links, search_radius, max_gap_frames, centroids_t1)

def save_tracks_and_positions(tracks, video_name, results_base_path="data/results"):
    """
    Dump track metadata, (x, y) positions, and mask values to plain-text files.

    File layout created
    -------------------
    tracks_<video>.txt    -> one line per track:   label start_frame end_frame parent_label
    positions_<video>.txt -> one line per frame:   label current_frame x y
    masks_<video>.txt     -> one line per frame:   label current_frame mask_value
    """

    # Ensure the output directory exists
    output_dir = os.path.join(results_base_path, video_name)
    os.makedirs(output_dir, exist_ok=True)

    # Full paths of the three output files
    tracks_file    = os.path.join(output_dir, f"tracks_{video_name}.txt")
    positions_file = os.path.join(output_dir, f"positions_{video_name}.txt")
    masks_file     = os.path.join(output_dir, f"masks_{video_name}.txt")

    # Open all files in a single context manager
    with open(tracks_file, "w") as tf, \
         open(positions_file, "w") as pf, \
         open(masks_file, "w") as mf:

        # Iterate over every CellTrack
        for track in tracks:
            # --------- Track-level information ------------------------ #
            # Use '-' when end_frame has not been assigned yet
            end_frame = track.end_frame if track.end_frame is not None else "-"
            tf.write(f"{track.label} {track.start_frame} {end_frame} {track.parent_label}\n")

            # --------- Frame-by-frame information --------------------- #
            # `positions` and `masks` are parallel lists: element i
            # belongs to frame  start_frame + i
            for idx, (x, y) in enumerate(track.positions):
                current_frame = track.start_frame + idx
                # Write centroids
                pf.write(f"{track.label} {current_frame} {x} {y}\n")

            for idx, mask_value in enumerate(track.masks):
                current_frame = track.start_frame + idx
                # Write mask values (integer label, array hash, etc.)
                mf.write(f"{track.label} {current_frame} {mask_value}\n")

    print(f"Saved tracks, positions and masks for {video_name}")


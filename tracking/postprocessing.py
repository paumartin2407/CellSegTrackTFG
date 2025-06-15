import os
import numpy as np
import pandas as pd
import os
from scipy.optimize import linear_sum_assignment 

def clean_tracks(video_name, results_base_path="data/results", min_duration=3):
    """
    Clean tracking results by:
    - Removing short tracks.
    - Renumbering track IDs.
    - Updating parent references.
    - Cleaning positions and masks files accordingly.

    Args:
        video_name (str): Name of the video (e.g., 'exp9').
        results_base_path (str): Base path to results folder.
        min_duration (int): Minimum track duration to keep.
    """
    # Define file paths
    tracks_path = os.path.join(results_base_path, video_name, f"tracks_{video_name}.txt")
    positions_path = os.path.join(results_base_path, video_name, f"positions_{video_name}.txt")
    masks_path = os.path.join(results_base_path, video_name, f"masks_{video_name}.txt")

    # Load tracks
    df = pd.read_csv(tracks_path, sep=r"\s+", names=["id", "start", "end", "parent"])

    # Compute track duration and filter by minimum duration
    df["duration"] = df["end"] - df["start"]
    df = df[df["duration"] >= min_duration].reset_index(drop=True)

    # Renumber IDs from 1 to N
    df["id_new"] = df.index + 1
    id_map = dict(zip(df["id"], df["id_new"]))

    # Update parent references
    df["parent_new"] = df["parent"].map(id_map).fillna(0).astype(int)

    # Save cleaned tracks
    cleaned_tracks_path = os.path.join(results_base_path, video_name, f"clean_tracks_{video_name}.txt")
    df_clean = df[["id_new", "start", "end", "parent_new"]]
    df_clean.columns = ["id", "start", "end", "parent"]
    df_clean.to_csv(cleaned_tracks_path, sep=" ", index=False, header=False)

    # Load and clean positions
    pos = pd.read_csv(positions_path, sep=r"\s+", names=["track", "frame", "x", "y"])
    pos = pos[pos["track"].isin(id_map)].copy()
    pos["track"] = pos["track"].map(id_map)
    info = df[["id_new", "start", "end"]].rename(columns={"id_new": "track"})
    pos = pos.merge(info, on="track", how="inner")
    pos = pos[(pos["frame"] >= pos["start"]) & (pos["frame"] <= pos["end"])]
    cleaned_positions_path = os.path.join(results_base_path, video_name, f"clean_positions_{video_name}.txt")
    pos = pos[["track", "frame", "x", "y"]].sort_values(["track", "frame"]).reset_index(drop=True)
    pos.to_csv(cleaned_positions_path, sep=" ", index=False, header=False)

    # Load and clean masks
    masks = pd.read_csv(masks_path, sep=r"\s+", names=["track", "frame", "value"])
    masks = masks[masks["track"].isin(id_map)].copy()
    masks["track"] = masks["track"].map(id_map)
    masks = masks.merge(info, on="track", how="inner")
    masks = masks[(masks["frame"] >= masks["start"]) & (masks["frame"] <= masks["end"])]
    cleaned_masks_path = os.path.join(results_base_path, video_name, f"clean_masks_{video_name}.txt")
    masks = masks[["track", "frame", "value"]].sort_values(["track", "frame"]).reset_index(drop=True)
    masks.to_csv(cleaned_masks_path, sep=" ", index=False, header=False)

    print(f"Tracks, positions, and masks cleaned successfully for '{video_name}'.")

def _build_cost_matrix(parents, children, edges, sentinel=1e9):
    """Return a cost matrix where invalid pairs get a large *sentinel* cost.

    *sentinel* es un valor alto (por ej. 1e9) que hace que el algoritmo
    húngaro nunca seleccione esas parejas, salvo que no exista alternativa.
    """
    cost = np.full((len(parents), len(children)), sentinel, dtype=float)
    for pid, cid, dist in edges:
        r = parents.index(pid)
        c = children.index(cid)
        cost[r, c] = dist
    return cost


def merge_tracks_optimal(
    video_name,
    results_base_path="data/results",
    frame_window=4,
    max_children=2,
    distance_threshold=None,):

    """Funde fragmentos minimizando la *suma global* de distancias.

    - Cada padre puede adoptar hasta ``max_children`` hijos.
    - Sólo se consideran hijos cuyo *start* ocurre dentro de ``frame_window``
      frames tras el *end* del padre.
    - Se descartan pares cuya distancia supere ``distance_threshold`` (si se
      proporciona).
    """

    # 1. Cargar datos -----------------------------------------------------
    tracks_path = os.path.join(results_base_path, video_name, f"clean_tracks_{video_name}.txt")
    positions_path = os.path.join(results_base_path, video_name, f"clean_positions_{video_name}.txt")

    tracks = pd.read_csv(tracks_path, sep=r"\s+", names=["id", "start", "end", "parent"])
    pos = pd.read_csv(positions_path, sep=r"\s+", names=["track", "frame", "x", "y"])

    last_pos = pos.sort_values("frame").groupby("track").tail(1).set_index("track")
    first_pos = pos.sort_values("frame").groupby("track").head(1).set_index("track")

    # Padress con mitosis reales (≥1 hijos originales) -------------------
    bio_children = tracks["parent"].value_counts().to_dict()

    # 2. Generar aristas válidas (padre, hijo, distancia) -----------------
    edges = []
    parent_capacity = {}
    candidate_children = set()

    for _, parent in tracks.iterrows():
        if bio_children.get(parent.id, 0) > 0:
            continue  # saltar padres con hijos biológicos
        if parent.id not in last_pos.index:
            continue

        parent_capacity[parent.id] = max_children
        x_p, y_p = last_pos.loc[parent.id, ["x", "y"]]
        pend = parent.end

        mask = (tracks.start >= pend) & (tracks.start <= pend + frame_window) & (tracks.id != parent.id)
        for _, child in tracks[mask].iterrows():
            if child.id not in first_pos.index:
                continue
            x_c, y_c = first_pos.loc[child.id, ["x", "y"]]
            dist = float(np.hypot(x_c - x_p, y_c - y_p))
            if distance_threshold is None or dist <= distance_threshold:
                edges.append((parent.id, child.id, dist))
                candidate_children.add(child.id)

    if not edges:
        print("No merge candidates satisfy given constraints.")
        return

    parents = list(parent_capacity.keys())
    children = sorted(candidate_children)

    # 3. Resolver con algoritmo húngaro ----------------------------------
    sentinel = 1e8  # coste alto para pares inválidos
    cost_matrix = _build_cost_matrix(parents, children, edges, sentinel=sentinel)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 4. Aplicar fusiones respetando capacidad ----------------------------
    merges_by_parent = {pid: [] for pid in parents}
    total_dist = 0.0

    for r, c in zip(row_ind, col_ind):
        pid = parents[r]
        cid = children[c]
        if cost_matrix[r, c] >= sentinel:
            continue  # par inválido
        if len(merges_by_parent[pid]) < parent_capacity[pid]:
            merges_by_parent[pid].append(cid)
            total_dist += cost_matrix[r, c]

    tracks.loc[tracks.id.isin([cid for v in merges_by_parent.values() for cid in v]), "parent"] = \
        tracks.id.map({cid: pid for pid, cids in merges_by_parent.items() for cid in cids})

    # 5. Guardar resultados ----------------------------------------------
    out_path = os.path.join(results_base_path, video_name, f"merged_tracks_{video_name}.txt")
    tracks.to_csv(out_path, sep=" ", index=False, header=False)

    merged_count = sum(len(v) for v in merges_by_parent.values())
    print(f"Optimal merge complete for '{video_name}'. {merged_count} children merged. Total distance = {total_dist:.2f}.")

    
def merge_tracks(video_name, results_base_path="data/results", frame_window=4, distance_threshold=45):
    """
    Merge up to two fragmented tracks per parent based on closest spatio-temporal proximity.

    Args:
        video_name (str): Name of the video (e.g., 'exp9').
        results_base_path (str): Base path to results folder.
        frame_window (int): Max number of frames between end and start.
        distance_threshold (float): Max spatial distance to merge tracks.
    """
    # Define file paths
    cleaned_tracks_path = os.path.join(results_base_path, video_name, f"clean_tracks_{video_name}.txt")
    cleaned_positions_path = os.path.join(results_base_path, video_name, f"clean_positions_{video_name}.txt")

    # Load tracks and positions
    tracks = pd.read_csv(cleaned_tracks_path, sep=r"\s+", names=["id", "start", "end", "parent"])
    positions = pd.read_csv(cleaned_positions_path, sep=r"\s+", names=["track", "frame", "x", "y"])

    # Prepare first and last known positions
    last_pos = positions.sort_values("frame").groupby("track").tail(1).set_index("track")
    first_pos = positions.sort_values("frame").groupby("track").head(1).set_index("track")

    # Map track id to number of children (mitosis detection)
    parent_count = tracks["parent"].value_counts().to_dict()

    for parent_row in tracks.itertuples():
        if parent_count.get(parent_row.id, 0) > 0:
            continue  # skip if already has children (mitosis)

        parent_end = parent_row.end
        if parent_row.id not in last_pos.index:
            continue

        x0, y0 = last_pos.loc[parent_row.id, ["x", "y"]]

        # Find candidate children
        candidates = tracks[(tracks.start > parent_end) &
                            (tracks.start <= parent_end + frame_window)]

        distances = []
        for row in candidates.itertuples():
            if row.id not in first_pos.index:
                continue
            x1, y1 = first_pos.loc[row.id, ["x", "y"]]
            dist = np.hypot(x1 - x0, y1 - y0)
            if dist < distance_threshold:
                distances.append((dist, row.id))

        # Sort all candidates by distance (closest first) and assign up to two
        distances.sort()
        merged = 0
        for _, child_id in distances:
            if merged >= 2:
                break
            tracks.loc[tracks.id == child_id, "parent"] = parent_row.id
            parent_count[parent_row.id] = parent_count.get(parent_row.id, 0) + 1
            merged += 1

    # Save result
    merged_tracks_path = os.path.join(results_base_path, video_name, f"merged_tracks_{video_name}old.txt")
    tracks.to_csv(merged_tracks_path, sep=" ", index=False, header=False)
    print(f"Tracks merged successfully for '{video_name}' considering closest pairs only.")
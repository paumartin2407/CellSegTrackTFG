import numpy as np
from scipy.spatial import KDTree
from itertools import combinations

def compute_centroids(mask):
    """
    Compute centroids for each labeled cell in the mask.

    Args:
        mask (np.ndarray): Segmentation mask where each cell has a unique label (>0).

    Returns:
        dict: Dictionary mapping each label to its centroid as (cx, cy).
    """
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background (label 0)
    centroids = {}

    for label in unique_labels:
        y_coords, x_coords = np.where(mask == label)
        if len(x_coords) > 0 and len(y_coords) > 0:
            # Centroid is the mean of x and y coordinates
            cx, cy = int(np.mean(x_coords)), int(np.mean(y_coords))
            centroids[label] = (cx, cy)
    return centroids

def build_active_centroids(tracks, det_centroids, links, max_gap_frames):
    """
    Constructs the set of active centroids to be used as the Left (L) nodes of the graph.

    This includes:
    - Real detections from the current frame (det_centroids)
    - Predicted positions for tracks that have recently disappeared (gap ≤ max_gap_frames)

    Args:
        tracks (list): List of CellTrack objects.
        det_centroids (dict): Dictionary of real detections in the current frame {label: (x, y)}.
        links (dict): Current mapping of new labels to existing track labels.
        max_gap_frames (int): Maximum number of consecutive missing frames to tolerate for a track 
                              to still be considered "active" (used for linking predictions).

    Returns:
        active (dict): Dictionary of active centroids {label: (x, y)}, including predictions.
        links (dict): Updated label-to-track mapping, including new predicted labels.
    """
    active = dict(det_centroids)  # Start with real detections
    new_label = len(links) + 1    # Start labeling predicted points after the last detection label

    for tr in tracks:
        # Consider tracks that are not ended and have been missing for ≤ max_gap_frames
        if 0 < tr.missing_frames < max_gap_frames and tr.end_frame is None:
            # Use the last known position before it disappeared
            last_centroid = tr.positions[-(tr.missing_frames + 1)]
            active[new_label] = last_centroid
            links[new_label] = tr.label  # Link new label to the original track label
            new_label += 1

    return active, links


def compute_edges_costs_capacities(centroids_t, centroids_t1, search_radius=35, image_size=(1024, 1024)):
    """
    Compute the costs and capacities of the edges in a min-cost flow graph 
    for cell tracking between two consecutive frames.

    Args:
        centroids_t (dict): Centroids at time t (label -> (x, y)).
        estimated_centroids (dict): Estimated centroids at t+1 (predicted from optical flow).
        centroids_t1 (dict): Observed centroids at time t+1.
        search_radius (float): Radius to consider neighboring detections (default: 35).
        image_size (tuple): Size of the images (width, height), default (1024, 1024).

    Returns:
        tuple: (costs dict, capacities dict) where keys are edges (node1, node2).
    """

    costs = {}
    capacities = {}

    # Extract centroid coordinates
    centroids_t_vals = list(centroids_t.values())
    centroids_t1_vals = list(centroids_t1.values())

    # Build KD-Trees for nearest neighbor queries
    centroids_t_Tree = KDTree(centroids_t_vals)
    centroids_t1_Tree = KDTree(centroids_t1_vals)

    centroids_t_keys = centroids_t.keys()
    centroids_t1_keys = centroids_t1.keys()

    img_width, img_height = 1024, 1024

    # Add source and sink connections for nodes at t and t+1
    for i in centroids_t_keys:
        costs[("T+", f"L{i}")] = 0
        capacities[("T+", f"L{i}")] = 1

    for j in centroids_t1_keys:
        costs[(f"R{j}", "T-")] = 0
        capacities[(f"R{j}", "T-")] = 1

    # Add auxiliary arcs
    costs[("T+", "A")] = 0
    capacities[("T+", "A")] = len(centroids_t1)

    costs[("D", "T-")] = 0
    capacities[("D", "T-")] = len(centroids_t)

    costs[("A", "D")] = 0
    capacities[("A", "D")] = len(centroids_t)

    # Find neighbors of each predicted centroid in frame t+1
    neighbors = centroids_t1_Tree.query_ball_point(centroids_t_vals, r=search_radius)

    for i, neighbor_indices in enumerate(neighbors):
        node_i = f"L{i+1}"  # Node from frame t
        p_it = np.array(centroids_t_vals[i], dtype=np.float64)

        if len(neighbor_indices) == 0:
            # No neighbors, add direct death edge
            costs[(node_i, "D")] = 0
        else:
            # Compute cost for death edge
            distances, _ = centroids_t1_Tree.query(p_it, k=3)
            w_D_CN = (distances[1] + distances[2]) / 2.0
            w_D = min(w_D_CN, p_it[0], img_width - p_it[0], p_it[1], img_height - p_it[1])
            costs[(node_i, "D")] = w_D

            # Add edges to nearby detections
            for j in neighbor_indices:
                node_j = f"R{j+1}"  # Node from frame t+1
                p_jt1 = np.array(centroids_t1_vals[j], dtype=np.float64)
                distance = np.linalg.norm(p_it - p_jt1)
                costs[(node_i, node_j)] = distance
                capacities[(node_i, node_j)] = 1

            # Create intermediate nodes for possible divisions
            for j, k in combinations(neighbor_indices, 2):
                node_j = f"R{j+1}"
                node_k = f"R{k+1}"
                s_node = f"S_{i+1}_{j+1}_{k+1}"

                # Compute division cost
                p_jt1 = np.array(centroids_t1_vals[j], dtype=np.float64)
                p_kt1 = np.array(centroids_t1_vals[k], dtype=np.float64)
                c1 = np.linalg.norm(p_it - 0.5 * (p_jt1 + p_kt1))  # Avg distance
                c2 = abs(np.linalg.norm(p_it - p_jt1) - np.linalg.norm(p_it - p_kt1))  # Symmetry term
                w = (c1 + c2) ** (3/4)

                # Connect auxiliary nodes
                costs[("A", s_node)] = 0
                capacities[("A", s_node)] = 1

                costs[(node_i, s_node)] = w
                capacities[(node_i, s_node)] = 1

                costs[(s_node, node_j)] = 0
                capacities[(s_node, node_j)] = 1

                costs[(s_node, node_k)] = 0
                capacities[(s_node, node_k)] = 1

    # Ensure all L nodes have a death edge
    for i in centroids_t_keys:
        capacities[(f"L{i}", "D")] = 1
        if (f"L{i}", "D") not in costs:
            costs[(f"L{i}", "D")] = 0

    # Compute cost for appearances (A -> R nodes)
    neighbors_indices_set = set(num + 1 for sublist in neighbors for num in sublist)
    for l in centroids_t1_keys:
        capacities[("A", f"R{l}")] = 1

        if l in neighbors_indices_set:
            p_lt1 = np.array(centroids_t1_vals[l-1], dtype=np.float64)
            distances, _ = centroids_t_Tree.query(p_lt1, k=3)
            w_A_CN = (distances[1] + distances[2]) / 2.0
            w_A = min(w_A_CN, p_lt1[0], img_width - p_lt1[0], p_lt1[1], img_height - p_lt1[1])
            costs[("A", f"R{l}")] = w_A
        else:
            costs[("A", f"R{l}")] = 0

    return costs, capacities
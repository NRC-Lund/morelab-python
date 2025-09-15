import qtm
import numpy as np
import os
from typing import Tuple

# Set up QTM Python API
#this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#if this_dir not in sys.path:
#    sys.path.append(this_dir)
#import qtm
from helpers.traj import get_unlabeled_marker_ids, get_labeled_marker_ids, get_marker_positions


def gui_generate_reference_distribution():
    # Check QTM version
    qtm_version = qtm.get_version()
    if qtm_version['major']<2025:
        print("Requires QTM 2025.1 or later.")
        return

    # Get labeled trajectories
    ids = get_labeled_marker_ids()

    # Remove prefixes from labels
    labels = []
    for id in ids:
        # Remove prefix
        label = qtm.data.object.trajectory.get_label(id)
        before, sep, after = label.partition("_")
        labels.append(after if sep else label) # Safe also if no prefix
    labels = np.array(labels, dtype=object)    # Make it indexable

    # Get positions
    pos = get_positions(ids)
    #print(f"Combined position data shape: {pos.shape}")

    # Calculate reference distributions
    print(f"Calculating reference distribution...")
    P, ixs, edges = calculate_reference_distributions(pos)
    P_labels = labels[ixs]  # Labels for each pair

    # Select file to save to
    fname_qtm = qtm.file.get_path()
    fname_npz = os.path.splitext(fname_qtm)[0] + ".npz"
    fname_npz = qtm.gui.dialog.show_save_file_dialog("Save reference distribution", 
                                                     ["NumPy files (*.npz)"], 
                                                     os.path.basename(fname_npz), 
                                                     os.path.dirname(fname_npz))
    if not fname_npz:
        print("No file selected, aborting")
        return
    
    # Save to npz file
    np.savez(fname_npz, P=P, P_labels=P_labels, edges=edges, labels=labels)
    print(f"Saved to {fname_npz}.")

def gui_auto_label():
    fname_npz = "Y:\Analysis\eScienceMoves\AutoLabel\QTM\P013\S4.npz"
    npz = np.load(fname_npz, allow_pickle=True)
    P_ref = npz['P']
    P_labels_ref = npz['P_labels']
    edges = npz['edges']
    labels_ref = npz['labels']
    print(f"Loaded reference distribution from {fname_npz}.")

    # Ungroup trajectories to single parts and delete gap-filled parts
    ungroup_trajectories()

    # Sort on trajectory length (descending)
    ids_unlabeled = get_unlabeled_marker_ids()
    counts = np.array([
        qtm.data.series._3d.get_sample_count(id)
        for id in ids_unlabeled
    ], dtype=int)
    order = np.argsort(-counts)              # indices sorted by descending count
    sorted_ids = np.array(ids_unlabeled)[order]
    sorted_counts = counts[order]
    if sorted_counts[0]<100:
        print('Only short trajectories.')
        return


def calculate_distribution(co1: np.ndarray, co2: np.ndarray, edges: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(co1 - co2, ord=2, axis=0)
    distances = distances[~np.isnan(distances)]
    num_bins = len(edges) - 1
    if distances.size == 0 or num_bins <= 0:
        return np.zeros((num_bins,), dtype=float)
    counts, _ = np.histogram(distances, bins=edges)
    probabilities = counts.astype(float) / float(distances.size)
    return probabilities


def distribution_similarity_score(P1: np.ndarray, P2: np.ndarray) -> float:
    if P1.shape != P2.shape:
        raise ValueError("P1 and P2 must have the same shape")
    return float(np.sum(P1 * P2))


def calculate_reference_distributions(
    co: np.ndarray,
    resolution: float = 10.0,
    max_distance: float = 3000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if co.ndim != 3 or co.shape[1] != 3:
        raise ValueError("Expected co with shape (num_trajectories, 3, num_frames)")
    num_trajectories = co.shape[0]
    edges = np.arange(0.0, max_distance + resolution, resolution, dtype=float)
    num_bins = len(edges) - 1
    num_pairs = (num_trajectories * num_trajectories - num_trajectories) // 2
    P = np.full((num_pairs, num_bins), np.nan, dtype=float)
    ixs = np.full((num_pairs, 2), -1, dtype=int)
    pair_index = 0
    for i1 in range(num_trajectories - 1):
        co1 = np.squeeze(co[i1, :, :])
        for i2 in range(i1 + 1, num_trajectories):
            co2 = np.squeeze(co[i2, :, :])
            P[pair_index, :] = calculate_distribution(co1, co2, edges)
            ixs[pair_index, :] = (i1, i2)
            pair_index += 1
    return P, ixs, edges


def get_positions(ids):
    # First pass: figure out num_frames from the first trajectory
    first_series = qtm.data.series._3d.get_samples(ids[0])
    num_frames = len(first_series)
    num_traj = len(ids)

    # Allocate array
    pos = np.full((num_traj, 3, num_frames), np.nan, dtype=float)

    # Second pass: fill data array
    for ti, id in enumerate(ids):
        series = qtm.data.series._3d.get_samples(id)
        # Fill in positions frame by frame
        for fj, f in enumerate(series):
            if f is not None and f.get("position") is not None:
                pos[ti, :, fj] = f["position"]

    return pos


def ungroup_trajectories():
## Split all unidentified trajectories into single parts and delete gap-filled parts
    id_unlabeled = get_unlabeled_marker_ids()
    filled_count = 0
    for marker_id in id_unlabeled:
        part_count = qtm.data.object.trajectory.get_part_count(marker_id)
        while part_count > 0:
            # Get the first part of the trajectory
            part = qtm.data.object.trajectory.get_part(marker_id, 0)
            if part['type'] == "filled":
                qtm.data.object.trajectory.delete_parts(marker_id, [0])  # delete filled part
                filled_count += 1
            else:
                # Create a new trajectory for the part
                new_traj = qtm.data.object.trajectory.add_trajectory()
                # Move the part to the new trajectory
                qtm.data.object.trajectory.move_parts(marker_id, new_traj, [0])
            part_count -= 1
    if filled_count > 0:
        print(f"Deleted {filled_count} gap-filled parts.")
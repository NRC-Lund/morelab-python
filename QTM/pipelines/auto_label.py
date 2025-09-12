import qtm
import numpy as np
import os
from typing import Tuple

def gui_generate_reference_distribution():
    #print(qtm.get_version())
    fname_qtm = qtm.file.get_path()
    fname_npz = os.path.splitext(fname_qtm)[0] + ".npz"
    fname_npz = qtm.gui.dialog.show_save_file_dialog("Save reference distribution", 
                                                     ["NumPy files (*.npz)"], 
                                                     os.path.basename(fname_npz), 
                                                     os.path.dirname(fname_npz))
    if not fname_npz:
        print("No file selected, aborting")
        return    
    generate_reference_distribution(fname_npz)

def generate_reference_distribution(fname_npz):
    print(f"Generating reference distribution and saving to {fname_npz}...")

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


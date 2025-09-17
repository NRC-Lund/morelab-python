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
    labels = get_labels_without_prefix(ids)
    pos = get_positions(ids)

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
    debug_flag = False

    # Check QTM version
    qtm_version = qtm.get_version()
    if qtm_version['major']<2025:
        print("Requires QTM 2025.1 or later.")
        return
    
    # Select reference distribution file
    if debug_flag:
        fname_npz = ["Y:\Analysis\eScienceMoves\AutoLabel\QTM\P013\S0 cogn.npz"]
    else:    
        fname_npz = qtm.gui.dialog.show_open_file_dialog(
            "Load reference distribution", 
            ["NumPy files (*.npz)"])
        if not fname_npz:
            print("No file selected, aborting")
            return

    # Load reference distribution
    npz = np.load(fname_npz[0], allow_pickle=True)
    P_ref = npz['P']
    P_labels_ref = npz['P_labels']
    edges = npz['edges']
    labels_ref = npz['labels']
    print(f"Loaded reference distribution from {fname_npz[0]}.")
    qtm.gui.message.add_message(
        f"Loaded distribution {fname_npz[0]}", 
        f"Loaded reference distribution from {fname_npz[0]}.", 
        "info")

    # Relabel the labeled trajectories
    print("Relabeling labeled trajectories...")
    relabel_labeled_trajectories(
        P_ref=P_ref,
        P_labels_ref=P_labels_ref,
        edges=edges,
        labels_ref=labels_ref,
    )

    # Label unlabeled trajectories
    print("Labeling unlabeled trajectories...")
    label_all_unlabeled_trajectories(
        P_ref=P_ref,
        P_labels_ref=P_labels_ref,
        edges=edges,
        labels_ref=labels_ref,
        min_len=20,
    )


def gui_auto_label_selected_trajectories():
    debug_flag = False

    # Check QTM version
    qtm_version = qtm.get_version()
    if qtm_version['major']<2025:
        print("Requires QTM 2025.1 or later.")
        return
    
    # Select reference distribution file
    if debug_flag:
        fname_npz = ["Y:\Analysis\eScienceMoves\AutoLabel\QTM\P013\S0 cogn.npz"]
    else:    
        fname_npz = qtm.gui.dialog.show_open_file_dialog(
            "Load reference distribution", 
            ["NumPy files (*.npz)"])
        if not fname_npz:
            print("No file selected, aborting")
            return

    # Load reference distribution
    npz = np.load(fname_npz[0], allow_pickle=True)
    P_ref = npz['P']
    P_labels_ref = npz['P_labels']
    edges = npz['edges']
    labels_ref = npz['labels']
    print(f"Loaded reference distribution from {fname_npz[0]}.")

    # Get selected trajectories
    ids_selected = qtm.gui.selection.get_selections("trajectory")
    ids_selected = [item['id'] for item in ids_selected if 'id' in item]
    if len(ids_selected) == 0:
        print("No trajectories selected, aborting")
        return
    
    # Check if trajectories are labeled or unlabeled
    ids_selected_labeled = [tid for tid in ids_selected if tid in get_labeled_marker_ids()]
    ids_selected_unlabeled = [tid for tid in ids_selected if tid in get_unlabeled_marker_ids()]

    # Relabel the labeled trajectories
    if len(ids_selected_labeled) > 0:
        print("Relabeling labeled trajectories...")
        relabel_labeled_trajectories(
            P_ref=P_ref,
            P_labels_ref=P_labels_ref,
            edges=edges,
            labels_ref=labels_ref,
            ids_labeled=ids_selected_labeled,
        )

    # Label unlabeled trajectories
    if len(ids_selected_unlabeled) > 0:
        for ids in ids_selected_unlabeled:
            label_unlabeled_trajectory(
                id_sel=ids,
                P_ref=P_ref,
                P_labels_ref=P_labels_ref,
                edges=edges,
                labels_ref=labels_ref,
            )


def gui_remove_spikes():
    print("Removing spikes and filling gaps...")
    ids = get_labeled_marker_ids()
    
    for idx, id in enumerate(ids):
        label = qtm.data.object.trajectory.get_label(id)
        pos = np.squeeze(get_positions([id]))
        spikes = detect_spikes(pos, k=20.0, include_neighbor=True)
        print(f"Detected {np.sum(spikes)} spikes in {label}.")
        #print(np.flatnonzero(spikes))


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


def get_positions(ids, series_range=None):
    # Get number of frames for the whole recording if not given
    if series_range is None:
        series_range = qtm.gui.timeline.get_measured_range()

    # Allocate array
    num_frames = series_range["end"]-series_range["start"]+1
    num_traj = len(ids)
    pos = np.full((num_traj, 3, num_frames), np.nan, dtype=float)

    # Fill array
    for ti, id in enumerate(ids):
        series = qtm.data.series._3d.get_samples(id, series_range)
        # Fill in positions frame by frame
        for fj, f in enumerate(series):
            if f is not None and f.get("position") is not None:
                pos[ti, :, fj] = f["position"]

    return pos # shape: (num_traj, 3, num_frames)


def detect_spikes(traj: np.ndarray, k: float = 6.0, include_neighbor: bool = False) -> np.ndarray:


    N = traj.shape[1]
    if N < 3:
        # need at least 3 frames for a second-order diff
        return np.zeros(N, dtype=bool)

    # Δposition vectors between consecutive frames: shape (3, N-1)
    dp = np.diff(traj, axis=1)

    # speed per step: shape (N-1,)
    speed = np.linalg.norm(dp, axis=0)

    # Δspeed (equiv. conv with [-1, 1]): shape (N-2,)
    ds = np.diff(speed)

    # Robust thresholding on ds
    med = np.nanmedian(ds)
    mad = np.nanmedian(np.abs(ds - med))
    if not np.isfinite(mad) or mad == 0:
        mad = 1e-12
    z = np.abs(ds - med) / (1.4826 * mad)  # ~|z-score| using MAD

    kernel = np.ones(4, dtype=float)  # default boxcar length 7
    k_used = kernel / kernel.sum()
    z_eff = np.convolve(z, k_used, mode="same")
    hits = z_eff > k



#    hits = z > k                           # boolean (N-2,)

    # Map Δspeed index i to frame i+1 (centered between steps i and i+1)
    spikes = np.zeros(N, dtype=bool)
    spikes[1:-1] = hits

    if include_neighbor and hits.any():
        # also mark neighbors around the centered frame
        spikes[:-1] |= np.r_[False, hits]      # previous frame
        spikes[1:]  |= np.r_[hits,  False]     # next frame

    # Steps that involved NaNs produce NaN in speed/ds and won't trigger hits.
    return spikes



def ungroup_unlabeled_trajectories():
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
        print(f"Deleted {filled_count} unlabeled, gap-filled parts.")


def delete_gapfilled_parts(ids=None):
    if ids is None:
        ids = qtm.data.series._3d.get_series_ids()

    # Delete all gap-filled parts
    for id in ids:
        label = qtm.data.object.trajectory.get_label(id)
        gap_filled_parts = []
        parts = qtm.data.object.trajectory.get_parts(id)
        for part, item in enumerate(parts):
            if item["type"] == "filled":
                gap_filled_parts.append(part)
        for part in reversed(gap_filled_parts):
            qtm.data.object.trajectory.delete_parts(id, [part])
            #print(f"part {part} was deleted from {label} due to being gap-filled")
        if qtm.data.object.trajectory.get_part_count(id) == 0:
            qtm.data.object.trajectory.delete_trajectory(id)
            #print(f"trajectory {id} deleted as all parts were removed from it")


def get_prefix():
    ids = get_labeled_marker_ids()
    label = qtm.data.object.trajectory.get_label(ids[0])
    before, sep, after = label.partition("_")
    return before if sep else ""

def get_labels_without_prefix(ids):
    # Remove prefixes from labels
    labels = []
    for id in ids:
        # Remove prefix
        label = qtm.data.object.trajectory.get_label(id)
        before, sep, after = label.partition("_")
        labels.append(after if sep else label) # Safe also if no prefix
    labels = np.array(labels, dtype=object)    # Make it indexable
    return labels

def guess_label(
    labels_ref: np.ndarray,         # candidate labels to try, shape: (K,)
    edges: np.ndarray,              # histogram edges, shape: (num_bins+1,)
    P1: np.ndarray,                  # reference distributions, shape: (num_pairs, num_bins)
    P1_labels: np.ndarray,           # pair labels for P1, shape: (num_pairs, 2), dtype str/object
    id_sel: int,                    # selected trajectory index (e.g., from find_longest_trajectory)
    series_range: dict = None       # optional range to use instead of full range
):
    
    # Get range of selected trajectory
    if series_range is None:
        series_range = qtm.data.series._3d.get_sample_range(id_sel)
    num_frames = series_range["end"]-series_range["start"]+1

    # Get positions of selected trajectory
    pos_sel = np.full((3, num_frames), np.nan, dtype=float)
    series = qtm.data.series._3d.get_samples(id_sel, series_range)
    # Fill in positions frame by frame
    for fj, f in enumerate(series):
        if f is not None and f.get("position") is not None:
            pos_sel[:, fj] = f["position"]
    
    # Get ids for all reference labels
    prefix = get_prefix()

    # Add prefix to reference labels
    ids_ref = [
        qtm.data.object.trajectory.find_trajectory(prefix + "_" + label)
        for label in labels_ref
    ]

    # Get positions for all reference trajectories
    pos_ref = get_positions(ids_ref, series_range)

    # Select candidates
    candidate_labels = labels_ref # Now we try all possible markers, but could be restricted to only missing ones

    # Calculate scores for all candidates
    score_m = np.full((len(candidate_labels),), np.nan, dtype=float)
    for iGuess, cand in enumerate(candidate_labels):
        # Select trajectories to compare to (all except the candidate itself)
        b_compare = ~np.isin(labels_ref, cand)
        n_compare = np.sum(b_compare)
        pos_compare = pos_ref[b_compare, :, :]               # (n_compare, 3, num_frames)
        labels_compare = np.asarray(labels_ref[b_compare])   # (n_compare,)

        # Calculate similarity scores for all marker pairs
        scores = np.full((n_compare,), np.nan, dtype=float)
        for iP in range(n_compare):
            # Calculate distance distribution
            P2 = calculate_distribution(pos_sel, np.squeeze(pos_compare[iP, :, :]), edges)

            # Find reference pair (row where both labels appear in the row, order-insensitive)
            mask = ((P1_labels[:, 0] == cand) | (P1_labels[:, 1] == cand)) & \
                   ((P1_labels[:, 0] == labels_compare[iP]) | (P1_labels[:, 1] == labels_compare[iP]))
            ix_ref = np.flatnonzero(mask)
            assert ix_ref.size == 1, "Could not find unique reference pair."

            # Calculate similarity score
            scores[iP] = distribution_similarity_score(P2, P1[ix_ref[0], :])

        # Average score for this candidate
        score_m[iGuess] = np.nanmean(scores)

    # Order candidates by score
    sort_ix = np.argsort(-score_m)  # Indices into candidate_labels, sorted by descending score
    ordered_labels = candidate_labels[sort_ix]
    ordered_scores = score_m[sort_ix]
    
    # Calculate contrast (best/second-best)
    best_score = score_m[sort_ix[0]] if score_m.size > 0 else np.nan
    if score_m.size > 1 and np.isfinite(score_m[sort_ix[1]]):
        contrast = best_score / score_m[sort_ix[1]]
    else:
        contrast = np.nan

    return ordered_labels, ordered_scores, contrast


def label_all_unlabeled_trajectories(
    P_ref: np.ndarray,
    P_labels_ref: np.ndarray,
    edges: np.ndarray,
    labels_ref: np.ndarray,
    min_len: int = 20,
    max_outer_iters: int = 1000,
    min_score: float = 0.001,
):

    # Ungroup all unlabeled trajectories to single parts and delete gap-filled parts
    ungroup_unlabeled_trajectories()

    # Iteratively label the longest unlabeled trajectory
    outer_iters = 0
    iters_n_unlabeled = []
    while True:
        outer_iters += 1
        if outer_iters > max_outer_iters:
            print("Stopping: reached max_outer_iters (safety).")
            break

        # Get all unlabeled trajectories
        #ids_unlabeled = get_unlabeled_marker_ids()
        ids_unlabeled = [
            tid for tid in get_unlabeled_marker_ids()
            if not qtm.data.object.trajectory.get_is_discarded(tid)
        ]

        
        n_unlabeled = len(ids_unlabeled)
        iters_n_unlabeled.append(n_unlabeled)
        if n_unlabeled == 0:
            print("No unlabeled trajectories left.")
            break
        print(f"{n_unlabeled} unlabeled trajectories left.")

        # Check for stagnation
        if len(iters_n_unlabeled) > 10 and iters_n_unlabeled[-9]<n_unlabeled+5:
            print("Warning: unlabeled count did not decrease enough in the last 10 passes, stopping.")
            break

        # Sort by length (desc)
        counts = np.array(
            [qtm.data.series._3d.get_sample_count(tid) for tid in ids_unlabeled],
            dtype=int,
        )
        order = np.argsort(-counts)
        sorted_ids = np.array(ids_unlabeled)[order]
        sorted_counts = counts[order]

        if sorted_counts[0] < min_len:
            print(f"Only short trajectories left (<{min_len}). Stopping.")
            break

        # Pick the longest unlabeled trajectory
        id_sel = int(sorted_ids[0])

        label_unlabeled_trajectory(
            id_sel=id_sel,
            P_ref=P_ref,
            P_labels_ref=P_labels_ref,
            edges=edges,
            labels_ref=labels_ref,
            min_score=min_score,
        )


def label_unlabeled_trajectory(
    id_sel: int,
    P_ref: np.ndarray,
    P_labels_ref: np.ndarray,
    edges: np.ndarray,
    labels_ref: np.ndarray,
    min_score: float = 0.001,
):

    # Guess label
    labels_guess, scores, contrast = guess_label(
        id_sel=id_sel, 
        P1=P_ref, 
        P1_labels=P_labels_ref, 
        edges=edges, 
        labels_ref=labels_ref
    )
    if scores[0] < min_score:
        print(f"Best guess score {scores[0]:.4f} below min_score {min_score:.4f}. Discarding.")
        qtm.data.object.trajectory.set_is_discarded(id_sel, True)
        return
        
    for label_guess, score in zip(labels_guess, scores):
        print(f"Guess: {label_guess} (score {score:.4f})")

        # Find the target trajectory
        id_guess = qtm.data.object.trajectory.find_trajectory(get_prefix() + "_" + label_guess)
        if id_guess is None:
            raise RuntimeError(f"Could not find trajectory for guessed label {label_guess}.")

        # Unlabel/move overlapping parts out of the guessed trajectory
        resolved = resolve_overlaps_into_target(
            id_sel=id_sel,
            id_target=id_guess,
            P_ref=P_ref,
            P_labels_ref=P_labels_ref,
            edges=edges,
            labels_ref=labels_ref,
        )
        if not resolved:
            print("Existing part fits better, trying next guess...")
            continue

        # Finally, move the selected unlabeled trajectory into the guessed one
        qtm.data.object.trajectory.move_parts(id_sel, id_guess)
        print(f"Moved candidate part to {label_guess}.\n")
        break
    else:
        raise RuntimeError("No suitable guess found, stopping.")

    return


def resolve_overlaps_into_target(
    id_sel: int,
    id_target: int,
    max_overlap_iters: int = 1000,
    P_ref: np.ndarray = None,
    P_labels_ref: np.ndarray = None,
    edges: np.ndarray = None,
    labels_ref: np.ndarray = None,
) -> int:
    """
    For the selected trajectory `id_sel`, remove any overlapping parts that currently
    exist in the target labeled trajectory `id_target` by moving those overlapping
    parts out to newly created trajectories.
    """
    success = False

    # Range of the candidate part we want to insert
    series_range = qtm.data.series._3d.get_sample_range(id_sel)
    if series_range is None:
        print("Selected trajectory has no range; skipping overlap resolution.")
        return success

    overlap_iters = 0
    while True:
        overlap_iters += 1
        if overlap_iters > max_overlap_iters:
            print("Stopping overlap resolution: reached max_overlap_iters (safety).")
            success = False
            break

        parts = qtm.data.series._3d.get_sample_ranges(id_target) or []
        print(f"Checking {len(parts)} existing parts for overlaps...")

        # Find first overlapping part index
        first_idx = next(
            (
                i for i, part in enumerate(parts)
                if not (part['end'] < series_range['start'] or part['start'] > series_range['end'])
            ),
            None
        )

        if first_idx is None:
            print("No overlaps.")
            success = True
            break

        print(f"Existing overlapping part: {parts[first_idx]['start']}-{parts[first_idx]['end']}")
        print(f"Candidate part: {series_range['start']}-{series_range['end']}")

        # Check which part fits better: the existing one or the candidate
        labels_guess1, scores1, contrast1 = guess_label(
            id_sel=id_sel, 
            P1=P_ref, 
            P1_labels=P_labels_ref, 
            edges=edges, 
            labels_ref=labels_ref
        )
        labels_guess2, scores2, contrast2 = guess_label(
            id_sel=id_target,
            series_range=parts[first_idx], 
            P1=P_ref, 
            P1_labels=P_labels_ref, 
            edges=edges, 
            labels_ref=labels_ref
        )
        print(f"Candidate part best guess: {labels_guess1[0]} (score {scores1[0]:.4f}, contrast {contrast1:.4f})")
        print(f"Existing part best guess: {labels_guess2[0]} (score {scores2[0]:.4f}, contrast {contrast2:.4f})")

        if scores1[0] > scores2[0]:
            # Create a new trajectory and move the conflicting part out of the target
            new_traj = qtm.data.object.trajectory.add_trajectory()
            qtm.data.object.trajectory.move_parts(id_target, new_traj, [first_idx])
            print("Unlabeled the overlapping part.")
        else:
            print("Keeping the existing part, skipping overlap resolution.")
            for label, score in zip(labels_guess1, scores1):
                print(f" {label}: {score:.4f}")
            success = False
            break

    return success


def relabel_labeled_trajectories(
    P_ref: np.ndarray,
    P_labels_ref: np.ndarray,
    edges: np.ndarray,
    labels_ref: np.ndarray,
    ids_labeled: list = None,
    min_len: int = 20,
    max_outer_iters: int = 1000,
    min_score: float = 0.001,
):
    if ids_labeled is None:
        ids_labeled = get_labeled_marker_ids()

    # Delete gap-filled parts first
    delete_gapfilled_parts(ids_labeled)

    for idx_labeled, id_labeled in enumerate(ids_labeled):
        print(f"Checking trajectory {idx_labeled+1}/{len(ids_labeled)}...")

        # Get label with and without prefix
        label = qtm.data.object.trajectory.get_label(id_labeled)
        label_noprefix = label.split("_",1)[1]  # Safe also if no prefix

        # Check all parts of the trajectory
        parts = qtm.data.series._3d.get_sample_ranges(id_labeled)
        if label:
            print(f" {label} has {len(parts)} parts.")

        idx_mismatch = []
        for idx_part, part in enumerate(parts):
            print(f"  Part {idx_part}: {part['start']}-{part['end']}")
            # Guess label
            labels_guess, scores, contrast = guess_label(
                id_sel=id_labeled,
                series_range=parts[idx_part],
                P1=P_ref, 
                P1_labels=P_labels_ref, 
                edges=edges, 
                labels_ref=labels_ref
            )
            print(f"   Score={scores[0]:.4f}, contrast={contrast:.4f}")
            # Compare the best guess with the current label
            if labels_guess[0] != label_noprefix:
                print(f"   Current label '{label_noprefix}' does not match the best guess '{labels_guess[0]}'.")
                idx_mismatch.append(idx_part)

        if len(idx_mismatch) > 0:
            new_traj = qtm.data.object.trajectory.add_trajectory()
            qtm.data.object.trajectory.move_parts(id_labeled, new_traj, idx_mismatch)
            print(f"  Unlabeled {len(idx_mismatch)} mismatching part(s).")
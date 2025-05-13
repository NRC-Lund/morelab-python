# #############################################################################
#                               MoRe-Lab QTM Scripting                       #
#               Python Motion Data Elaboration Toolbox for QTM               #
#
# This file is part of the MoRe-Lab QTM Scripting utilities.
# Copyright (C) 2025  Zachary Flahaut
#
# MoRe-Lab QTM Scripting is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MoRe-Lab QTM Scripting is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Author: Zachary Flahaut
# #############################################################################
import numpy as np
import qtm
from PyQt5 import QtWidgets

# ------------------------------------------------------------------------------
# USER-DEFINED LABELS and Reconstruction Mappings
# ------------------------------------------------------------------------------
# List all your trajectories
trajectory_labels = [
    # Head & chest
    "Q_HeadL", "Q_HeadR", "Q_HeadFront", "Q_Chest",
    # Thorax
    "Q_SpineThoracic2", "Q_SpineThoracic12",
    # Upper limbs
    "Q_LShoulderTop", "Q_LArm", "Q_LElbowOut", "Q_LElbowIn",
    "Q_LWristIn", "Q_LWristOut", "Q_LHand2",
    "Q_RShoulderTop", "Q_RArm", "Q_RElbowOut", "Q_RElbowIn",
    "Q_RWristIn", "Q_RWristOut", "Q_RHand2",
    # Pelvis / waist
    "Q_LSips", "Q_RSips", "Q_WaistLFront", "Q_WaistL", "Q_WaistBack",
    "Q_WaistR", "Q_WaistRFront", "Q_ChestLow",
    # Lower limbs (left)
    "Q_LThighHigh", "Q_LThighLow", "Q_LThighMedial",
    "Q_LShinFrontHigh", "Q_LShinFrontLow", "Q_LShinSide",
    "Q_LKneeIn", "Q_LKneeOut",
    "Q_LAnkleIn", "Q_LAnkleOut", "Q_LHeelBack",
    "Q_LForefoot1", "Q_LForefoot2", "Q_LForefoot5",
    # Lower limbs (right)
    "Q_RThighHigh", "Q_RThighLow", "Q_RThighMedial",
    "Q_RShinFrontHigh", "Q_RShinFrontLow",
    "Q_RShinSide",
    "Q_RKneeIn", "Q_RKneeOut",
    "Q_RAnkleIn", "Q_RAnkleOut", "Q_RHeelBack",
    "Q_RForefoot1", "Q_RForefoot2", "Q_RForefoot5"
]

# ------------------------------------------------------------------------------
# RECONSTRUCTION MAPPINGS: Ordered lists of source sets per target
# Each target has primary triple and fallbacks; each list is a list of marker-label lists.
# During reconstruction we pick the first source set with ≥3 available markers.
# If more than 3 are available, we include them all for a robust fit.
# ------------------------------------------------------------------------------
reconstruct_mapping = {
    "Q_HeadFront": [
        ["Q_HeadL", "Q_HeadR", "Q_SpineThoracic2"],
        ["Q_HeadL", "Q_HeadR", "Q_Chest"],
        ["Q_HeadR", "Q_Chest", "Q_SpineThoracic2"],
        ["Q_HeadL", "Q_Chest", "Q_SpineThoracic2"]
    ],
    "Q_RAnkleOut": [
        ["Q_RAnkleIn", "Q_RHeelBack", "Q_RForefoot2"],
        ["Q_RAnkleIn", "Q_RHeelBack", "Q_RForefoot1"],
        ["Q_RAnkleIn", "Q_RHeelBack", "Q_RForefoot5"],
        ["Q_RHeelBack", "Q_RForefoot2", "Q_RForefoot1"],
        ["Q_RHeelBack", "Q_RForefoot2", "Q_RForefoot5"],
        ["Q_RHeelBack", "Q_RForefoot1", "Q_RForefoot5"]

    ],
    "Q_RAnkleIn":  [
        ["Q_RAnkleOut", "Q_RHeelBack", "Q_RForefoot2"],
        ["Q_RAnkleOut", "Q_RHeelBack", "Q_RForefoot1"],
        ["Q_RAnkleOut", "Q_RHeelBack", "Q_RForefoot5"],
        ["Q_RHeelBack", "Q_RForefoot2", "Q_RForefoot1"],
        ["Q_RHeelBack", "Q_RForefoot2", "Q_RForefoot5"],
        ["Q_RHeelBack", "Q_RForefoot1", "Q_RForefoot5"]
    ],
    "Q_RKneeOut":  [
        ["Q_RKneeIn", "Q_RThighLow", "Q_RThighMedial"],
        ["Q_RKneeIn", "Q_RShinFrontHigh", "Q_RShinFrontLow"]
    ],
    "Q_RKneeIn":   [
        ["Q_RKneeOut", "Q_RThighLow", "Q_RThighMedial"],
        ["Q_RKneeOut", "Q_RShinFrontHigh", "Q_RShinFrontLow"]
    ],
    "Q_WaistRFront": [
        ["Q_RSips", "Q_WaistR", "Q_WaistLFront"]
    ],
    "Q_RElbowOut": [
        ["Q_RElbowIn","Q_RShoulderTop","Q_RArm"],
        ["Q_RElbowIn","Q_RWristIn","Q_RWristOut"]
    ],
    "Q_RElbowIn": [
        ["Q_RElbowOut","Q_RShoulderTop","Q_RArm"],
        ["Q_RElbowOut","Q_RWristIn","Q_RWristOut"]
    ],
    # Left side mirroring
    "Q_LAnkleOut": [
        ["Q_LAnkleIn", "Q_LHeelBack", "Q_LForefoot2"],
        ["Q_LAnkleIn", "Q_LHeelBack", "Q_LForefoot1"],
        ["Q_LAnkleIn", "Q_LHeelBack", "Q_LForefoot5"],
        ["Q_LHeelBack", "Q_LForefoot2", "Q_LForefoot1"],
        ["Q_LHeelBack", "Q_LForefoot2", "Q_LForefoot5"],
        ["Q_LHeelBack", "Q_LForefoot1", "Q_LForefoot5"]
    ],
    "Q_LAnkleIn":  [
        ["Q_LAnkleOut", "Q_LHeelBack", "Q_LForefoot2"],
        ["Q_LAnkleOut", "Q_LHeelBack", "Q_LForefoot1"],
        ["Q_LAnkleOut", "Q_LHeelBack", "Q_LForefoot5"],
        ["Q_LHeelBack", "Q_LForefoot2", "Q_LForefoot1"],
        ["Q_LHeelBack", "Q_LForefoot2", "Q_LForefoot5"],
        ["Q_LHeelBack", "Q_LForefoot1", "Q_LForefoot5"]
    ],
    "Q_LKneeOut":  [
        ["Q_LKneeIn", "Q_LThighLow", "Q_LThighMedial"],
        ["Q_LKneeIn", "Q_LShinFrontHigh", "Q_LShinFrontLow"]
    ],
    "Q_LKneeIn":   [
        ["Q_LKneeOut", "Q_LThighLow", "Q_LThighMedial"],
        ["Q_LKneeOut", "Q_LShinFrontHigh", "Q_LShinFrontLow"]
    ],
    "Q_WaistLFront": [
        ["Q_LSips", "Q_WaistL", "Q_WaistRFront"]
    ],
    "Q_LElbowOut": [
        ["Q_LElbowIn","Q_LShoulderTop","Q_LArm"],
        ["Q_LElbowIn","Q_LWristIn","Q_LWristOut"]
    ],
    "Q_LElbowIn": [
        ["Q_LElbowOut","Q_LShoulderTop","Q_LArm"],
        ["Q_LElbowOut","Q_LWristIn","Q_LWristOut"]
    ]
}

# ------------------------------------------------------------------------------
# STORAGE FOR STATIC POSITIONS
# This dictionary maps label → numpy array of [x, y, z] from the static trial.
# ------------------------------------------------------------------------------
_static_positions = {}

# ------------------------------------------------------------------------------
# FUNCTION: STATIC CALIBRATION
# Purpose:  Read each defined marker's 3D position at the static pose frame
# Steps:
#   1) Query QTM for the measured frame range
#   2) Pick the middle frame to avoid edge noise
#   3) For each label, get its trajectory ID and sample at that frame
#   4) If valid, store its [x,y,z] in _static_positions
#   5) Warn about any missing markers
# ------------------------------------------------------------------------------
def static_calibration():
    global _static_positions
    print("Running static calibration…")

    # Determine the middle frame of the measured range
    rng = qtm.gui.timeline.get_measured_range()
    frame = (rng["start"] + rng["end"]) // 2

    _static_positions.clear()
    missing = []
    for label in trajectory_labels:
        # Find trajectory in QTM by name
        tid = qtm.data.object.trajectory.find_trajectory(label)
        if tid is None:
            # Keep track of markers not found in static trial
            missing.append(label)
            continue
        sample = qtm.data.series._3d.get_sample(tid, frame)
        if not sample or sample.get("position") is None:
            missing.append(label)
            continue
        _static_positions[label] = np.array(sample["position"])

    if missing:
        print(f"WARNING: static data missing for {len(missing)} markers:\n  {missing}")
    print(f"Static calibration complete: captured {len(_static_positions)} positions.")

# ------------------------------------------------------------------------------
# FUNCTION: UMEYAMA ALGORITHM
#    Estimate optimal rotation R and translation t aligning A→B in a least‐squares sense
# Reference: S. Umeyama, "Least-Squares Estimation of Transformation Parameters
#            between Two Point Patterns," IEEE Transactions on Pattern Analysis
#            and Machine Intelligence, vol. 13, no. 4, pp. 376-380, 1991.
# Inputs:
#   A, B: Nx3 arrays of corresponding 3D points
# Outputs:
#   R: 3x3 rotation matrix
#   t: 3x1 translation vector
# Steps:
#   1) Compute centroids of A and B
#   2) Center points by subtracting centroids
#   3) Compute cross-covariance H = sum(A'ᵀ B') / N
#   4) SVD of H = U Σ Vᵀ, then R = V Uᵀ (fix sign if needed)
#   5) t = μ_B − R μ_A
# ------------------------------------------------------------------------------
def _umeyama(A, B):
    assert A.shape == B.shape, "Point sets must have same shape"
    # Compute centroids
    muA = A.mean(axis=0)
    muB = B.mean(axis=0)
    # Center
    AA = A - muA
    BB = B - muB
    # Cross-covariance
    H = AA.T.dot(BB) / A.shape[0]
    # SVD
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    # Ensure proper rotation
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T.dot(U.T)
    # Translation
    t = muB - R.dot(muA)
    return R, t

# ------------------------------------------------------------------------------
# FUNCTION: _select_targets
#   Display a GUI list of targets that are not 100% filled
# Steps:
#   1) For each target, count frames with valid data vs total frames
#   2) If present < total, add to "incomplete" list with fill count
#   3) Show a multi-select Qt dialog listing "label (present/total, %)%"
#   4) Return list of chosen labels (or empty if cancelled)
# ------------------------------------------------------------------------------
def _select_targets():
    rng = qtm.gui.timeline.get_measured_range()
    frames = list(range(rng["start"], rng["end"] + 1))
    incomplete = []

    # Compute present count per target
    for target in reconstruct_mapping:
        tid = qtm.data.object.trajectory.find_trajectory(target)
        present = 0
        if tid is not None:
            for f in frames:
                sample = qtm.data.series._3d.get_sample(tid, f)
                if sample and sample.get("position") is not None:
                    present += 1
        # if any frames missing, include
        if present < len(frames):
            incomplete.append((target, present, len(frames)))

    # Nothing to fix
    if not incomplete:
        QtWidgets.QMessageBox.information(None, "Reconstruction",
            "All targets have full data. Nothing to reconstruct.")
        return []

    # Show multi-selection list with fill %
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dlg = QtWidgets.QDialog()
    dlg.setWindowTitle("Select Trajectories to Recreate")
    layout = QtWidgets.QVBoxLayout(dlg)
    lw = QtWidgets.QListWidget()
    for tgt, pres, tot in incomplete:
        pct = pres/tot*100
        lw.addItem(f"{tgt} ({pres}/{tot}, {pct:.1f}% filled)")
    lw.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
    layout.addWidget(lw)
    btns = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    btns.accepted.connect(dlg.accept)
    btns.rejected.connect(dlg.reject)
    layout.addWidget(btns)
    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return []
    return [item.text().split()[0] for item in lw.selectedItems()]

# ------------------------------------------------------------------------------
# FUNCTION: DYNAMIC CALIBRATION
#    For each selected target, pick the best source set, reconstruct per frame
#    Handles full replace vs partial add if >25% existing fill
# Steps:
#   1) Ensure static calibration was run
#   2) Determine targets (via GUI or supplied list)
#   3) For each target:
#        a) Find first source set with ≥3 tracked markers
#        b) Compute static source matrix A (k×3)
#        c) For each frame:
#           - If >25% existing data and frame present, keep it
#           - Else, sample dynamic B (k×3), run Umeyama, apply to static target
#        d) Write all samples back into QTM
#   4) Print summary per target ("partial add" vs "full replace")
# ------------------------------------------------------------------------------
def dynamic_calibration(selected_targets=None):
    if not _static_positions:
        print("ERROR: Run static_calibration() first.")
        return

        # 2) Determine which targets to process (via GUI or explicit list)
    if selected_targets is None:
        targets = _select_targets()
    else:
        targets = [t for t in selected_targets if t in reconstruct_mapping]
    if not targets:
        print("No targets selected. Aborting.")
        return

        # 3) Prepare frame list
    rng = qtm.gui.timeline.get_measured_range()
    frames = list(range(rng["start"], rng["end"] + 1))
    total = len(frames)

    # 4) Process each target
    for target in targets:
        # a) Find the first valid source set with ≥3 tracked markers
        sources = None
        source_tids = []
        for candidate in reconstruct_mapping[target]:
            tids = [qtm.data.object.trajectory.find_trajectory(m) for m in candidate]
            valid = [(m, tid) for m, tid in zip(candidate, tids) if tid is not None]
            if len(valid) >= 3:
                sources = [m for m, _ in valid]
                source_tids = [tid for _, tid in valid]
                break
        if sources is None:
            print(f"ERROR: no available source set for {target}. Skipping.")
            continue

        # b) Determine existing frames for this target
        tid_t = qtm.data.object.trajectory.find_trajectory(target)
        present_frames = []
        if tid_t is not None:
            for f in frames:
                samp = qtm.data.series._3d.get_sample(tid_t, f)
                if samp and samp.get("position") is not None:
                    present_frames.append(f)
        present_count = len(present_frames)
        ratio = present_count / total if total else 0
        print(f"The ratio is of {ratio}.")

        # c) Build the static source matrix A
        A = np.vstack([_static_positions[m] for m in sources])  # k×3

        # d) Partial add: only fill missing frames
        if ratio > 0.25 and tid_t is not None:
            # Loop through frames and reconstruct only missing ones
            print("Replacing frame by frame")
            for f in frames:
                print(f"We are at frame {f}")
                if f in present_frames:
                    continue  # skip existing data
                # Gather dynamic source positions B
                B = np.vstack([
                    np.array(qtm.data.series._3d.get_sample(tid, f)["position"])
                    for tid in source_tids
                ])
                # Compute optimal R, t and apply to static target
                R, t = _umeyama(A, B)
                p0 = _static_positions[target]
                pos = R.dot(p0) + t
                # Write only this single-frame sample
                qtm.data.series._3d.set_samples(
                    tid_t,
                    {"start": f, "end": f},
                    [{"position": pos.tolist(), "residual": 0.0}]
                )
            print(f"Reconstructed missing frames for {target} (partial add, {present_count}/{total} existed).")
        else:
            # e) Full replace: reconstruct all frames
            print("Full recreation of the selected trajectory")
            recon_samples = []
            for f in frames:
                # Same Umeyama-based reconstruction for every frame
                B = np.vstack([
                    np.array(qtm.data.series._3d.get_sample(tid, f)["position"])
                    for tid in source_tids
                ])
                R, t = _umeyama(A, B)
                p0 = _static_positions[target]
                pos = R.dot(p0) + t
                recon_samples.append({"position": pos.tolist(), "residual": 0.0})
            # Create trajectory if absent
            if tid_t is None:
                tid_t = qtm.data.object.trajectory.add_trajectory(target)
            # Overwrite full range
            qtm.data.series._3d.set_samples(tid_t, rng, recon_samples)
            print(f"Reconstructed all frames for {target} (full replace, {present_count}/{total} existed).")

# ------------------------------------------------------------------------------
# CLI ENTRYPOINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Reconstruct missing markers using multiple fallback sets"
    )
    parser.add_argument("mode", choices=["static", "dynamic"],
                        help="Which step to run")
    parser.add_argument("--targets", nargs="*",
                        help="(dynamic) explicit list of targets to reconstruct")
    args = parser.parse_args()

    if args.mode == "static":
        static_calibration()
    else:
        dynamic_calibration(selected_targets=args.targets)
# #############################################################################
#                               MoRe-Lab QTM Scripting                       #
#               Python Motion Data Elaboration Toolbox for QTM               #
#
# This file is part of the MoRe-Lab QTM Scripting utilities.
# Copyright (C) 2025  Nicholas Ryan & Pär Halje
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
# Author: Nicholas Ryan & Pär Halje
# #############################################################################

import sys
import os
import inspect
import numpy as np

# Set up QTM Python API
this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if this_dir not in sys.path:
    sys.path.append(this_dir)
import qtm
from helpers.traj import get_unlabeled_marker_ids, get_labeled_marker_ids

def process_side(id_wb, id_wl, id_wr, side, prefix):
    debug = False

    # Left or right
    if side == "L":
        id_this = id_wl
        id_other = id_wr
    else:
        id_this = id_wr
        id_other = id_wl

    # Create new trajectory for the LSips or RSips
    qtm.data.object.trajectory.add_trajectory(f"{prefix}_{side}Sips")
    new_id = qtm.data.object.trajectory.find_trajectory(f"{prefix}_{side}Sips")
    qtm.data.object.trajectory.set_color(new_id, qtm.data.object.trajectory.get_color(id_this))

    # Identify nearest unidentified marker to waist markers and assign to new trajectory
    frames = qtm.gui.timeline.get_measured_range()
    frame_range = range(frames['start'], frames['end'])
    i = 0  # index for frame number
    traj_end = 0 # end of current SIPS trajectory
    while i < len(frame_range):
        qtm.gui.timeline.set_current_frame(i)
        frame = frame_range[i]

        # Estimate SIPS position
        pos_wb = np.array(qtm.data.series._3d.get_sample(id_wb, i)["position"]) # Back marker
        pos_this = np.array(qtm.data.series._3d.get_sample(id_this, i)["position"]) # Side marker, this side
        pos_other = np.array(qtm.data.series._3d.get_sample(id_other, i)["position"]) # Side marker, other side
        sips_estimate = pos_wb + (pos_this - pos_wb) * (1/3) # Estimate SIPS position as 1/3 of the way from back to side marker
        sips_estimate_other = pos_wb + (pos_other - pos_wb) * (1/3) # Estimate SIPS position on other side (used for not assigning the SIPS from the other side when this side is missing)

        # Find closest marker
        dist_limit = 100 # Max allowed distance to sips_estimate in mm.
        min_dist = float('inf')
        candidate_id = None
        id_unlabelled = get_unlabeled_marker_ids()  # repeat command because unidentified marker list has now changed
        for marker_id in id_unlabelled:
            sample = qtm.data.series._3d.get_sample(marker_id, i)
            if sample:
                dist = np.linalg.norm(np.array(sample["position"]) - sips_estimate)
                dist_other = np.linalg.norm(np.array(sample["position"]) - sips_estimate_other)
                if dist < min_dist and dist < dist_limit and dist < dist_other:
                    min_dist = dist
                    candidate_id = marker_id

        # Check if candidate is overlapping with previous SIPS trajectory
        if candidate_id:
            traj_range = qtm.data.series._3d.get_sample_range(candidate_id)
            if traj_range['start'] < traj_end:
                print(f"At frame {i}, the closest unidentified part is overlapping with previous {prefix}_{side}Sips parts.")
                candidate_id = None

        # Assign candidate to trajectory
        if candidate_id:
            if debug:
                print(f"At frame {i}, found candidate marker {candidate_id} at distance {min_dist:.2f} mm.")
            # Assign identified part to trajectory
            qtm.data.object.trajectory.move_parts(candidate_id, new_id)

            # Find new end point of trajectory and move to that frame
            part_count = qtm.data.object.trajectory.get_part_count(new_id)
            marker_info = qtm.data.object.trajectory.get_part(new_id, part_count - 1)
            traj_end = marker_info['range']['end']
            i = traj_end + 1
        else:
            # No canditate found, move to next frame
            i += 1

    # Check if the new trajectory is continuous
    gap_ranges = qtm.data.series._3d.get_gap_ranges(new_id)
    traj_length = qtm.data.series._3d.get_sample_range(new_id)
    if traj_length['start'] != 0 or frames['end'] != traj_length['end'] or gap_ranges:
        print(f"A new {prefix}_{side}Sips trajectory was created, but it is not continuous. Please check the gaps.")
    else:
        print(f"A new {prefix}_{side}Sips trajectory was created. It has no gaps.")

def check_trajectory_continuity(id):
    gap_ranges = qtm.data.series._3d.get_gap_ranges(id)
    traj_length = qtm.data.series._3d.get_sample_range(id)
    frames = qtm.gui.timeline.get_measured_range()
    if traj_length['start'] != 0 or frames['end'] != traj_length['end'] or gap_ranges:
        print(
            f"The {qtm.data.object.trajectory.get_label(id)} trajectory is not continuous. Please check the measurement.")
        return False
    else:
        return True

def fix_sips():
    print("Starting fix_sips.py...")

    # Get label prefix
    ids = get_labeled_marker_ids()
    for id in ids:
        label = qtm.data.object.trajectory.get_label(id)
        if label is not None:
            prefix = label.split("_")[0]
            break
    else:
        print("No labelled markers found. Please label the markers before running this script.")
        return
    print(f"Using prefix: {prefix}")

    # Get IDs of waist markers
    id_wb = qtm.data.object.trajectory.find_trajectory(f"{prefix}_WaistBack")
    if id_wb is None: print(f"Marker {prefix}_WaistBack is missing."); return
    id_wl = qtm.data.object.trajectory.find_trajectory(f"{prefix}_WaistLFront")
    if id_wl is None: print(f"Marker {prefix}_WaistLFront is missing."); return
    id_wr = qtm.data.object.trajectory.find_trajectory(f"{prefix}_WaistRFront")
    if id_wr is None: print(f"Marker {prefix}_WaistRFront is missing."); return

    # Check if waist markers are fully labelled
    if not check_trajectory_continuity(id_wb): return
    if not check_trajectory_continuity(id_wl): return
    if not check_trajectory_continuity(id_wr): return

    # Unidentify sips markers if they exist
    id_sl = qtm.data.object.trajectory.find_trajectory(f"{prefix}_LSips")
    if id_sl:
        qtm.data.object.trajectory.set_label(id_sl)
        print(f"Unidentified existing {prefix}_LSips trajectory.")
    id_sr = qtm.data.object.trajectory.find_trajectory(f"{prefix}_RSips")
    if id_sr:
        qtm.data.object.trajectory.set_label(id_sr)
        print(f"Unidentified existing {prefix}_RSips trajectory.")

    ## Split all unidentified trajectories into single parts and delete gap-filled parts
    id_unlabelled = get_unlabeled_marker_ids()
    filled_count = 0
    for marker_id in id_unlabelled:
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
        print(f"{filled_count} gap-filled parts have been deleted.")

    # Process left side
    print(f"Processing left side...")
    process_side(id_wb, id_wl, id_wr, "L", prefix)

    # Process right side
    print(f"Processing right side...")
    process_side(id_wb, id_wl, id_wr, "R", prefix)
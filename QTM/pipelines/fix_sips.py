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
from helpers.traj import get_unlabeled_marker_ids

def process_side(id_wb, id_wl, id_wr, side):
    # Left or right
    if side == "L":
        waist_markers = [id_wb, id_wl]
        color = qtm.data.object.trajectory.get_color(id_wl)
    else:
        waist_markers = [id_wb, id_wr]
        color = qtm.data.object.trajectory.get_color(id_wr)

    # Create new trajectory for the LSips or RSips
    qtm.data.object.trajectory.add_trajectory(f"Q_{side}Sips")
    new_id = qtm.data.object.trajectory.find_trajectory(f"Q_{side}Sips")
    qtm.data.object.trajectory.set_color(new_id, color)

    # Identify nearest unidentified marker to waist markers and assign to new trajectory
    frames = qtm.gui.timeline.get_measured_range()
    frame_range = range(frames['start'], frames['end'])
    i = 0  # index for frame number
    while i < len(frame_range):
        qtm.gui.timeline.set_current_frame(i)
        frame = frame_range[i]

        # Get coordinates of waist markers
        sample1 = qtm.data.series._3d.get_sample(waist_markers[0], i)
        sample2 = qtm.data.series._3d.get_sample(waist_markers[1], i)
        waist_cent = np.mean(np.array([sample1["position"], sample2["position"]]), axis=0)

        # Find closest marker
        min_dist = float('inf')
        sips_marker_id = None
        id_unlabeled = get_unlabeled_marker_ids()  # repeat command because unidentified marker list has now changed
        for marker_id in id_unlabeled:
            sample = qtm.data.series._3d.get_sample(marker_id, i)
            if sample:
                dist = np.linalg.norm(np.array(sample["position"]) - waist_cent)
                if dist < min_dist:
                    min_dist = dist
                    sips_marker_id = marker_id

        # Check candidate
        if sips_marker_id:
            traj_length = qtm.data.series._3d.get_sample_range(sips_marker_id)
            if traj_length['start'] < i:
                print(f"At frame {i}, the closest unidentified part is overlapping with previous Q_{side}Sips parts.")
                sips_marker_id = None

        if sips_marker_id:
            # Assign identified part to trajectory
            qtm.data.object.trajectory.move_parts(sips_marker_id, new_id)

        # Find new end point of trajectory and move to that frame
        part_count = qtm.data.object.trajectory.get_part_count(new_id)
        marker_info = qtm.data.object.trajectory.get_part(new_id, part_count - 1)
        traj_end = marker_info['range']['end']
        i = traj_end + 1


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
    id_wb = qtm.data.object.trajectory.find_trajectory("Q_WaistBack")
    id_wl = qtm.data.object.trajectory.find_trajectory("Q_WaistLFront")
    id_wr = qtm.data.object.trajectory.find_trajectory("Q_WaistRFront")
    if not check_trajectory_continuity(id_wb): return
    if not check_trajectory_continuity(id_wl): return
    if not check_trajectory_continuity(id_wr): return

    # Get IDs of waist markers
    id_sl = qtm.data.object.trajectory.find_trajectory("Q_LSips")
    id_sr = qtm.data.object.trajectory.find_trajectory("Q_RSips")

    # Unidentify sips markers if they exist
    if id_sl:
        qtm.data.object.trajectory.set_label(id_sl)
    if id_sr:
        qtm.data.object.trajectory.set_label(id_sr)

    ## split all unidentified trajectories into single parts and delete filled parts
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
        print(f"{filled_count} gap-filled parts have been deleted.")

    # Process left side
    process_side(id_wb, id_wl, id_wr, "L")

    # Process right side
    process_side(id_wb, id_wl, id_wr, "R")

    # Report what was done
    print("SIPS markers have been identified and assigned to the correct trajectories.")
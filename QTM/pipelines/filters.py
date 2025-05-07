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
from scipy.signal import butter, filtfilt
import qtm

trajectory_labels = [ ... ]

def apply_butterworth_filter(order=4, cutoff=6.0):
    """
    Apply a 4th-order Butterworth filter to all tracked trajectories.
    """
    # function body…


import numpy as np
from scipy.signal import butter, filtfilt
import qtm

trajectory_labels = [
    "Q_HeadL", "Q_HeadR", "Q_HeadFront", "Q_Chest", "Q_SpineThoracic2",
    "Q_SpineThoracic12", "Q_LShoulderTop", "Q_LArm", "Q_LElbowOut", "Q_LElbowIn",
    "Q_LWristIn", "Q_LWristOut", "Q_LHand2", "Q_RShoulderTop", "Q_RArm",
    "Q_RElbowOut", "Q_RElbowIn", "Q_RWristIn", "Q_RWristOut", "Q_RHand2",
    "Q_WaistLFront", "Q_WaistL", "Q_WaistBack", "Q_WaistR", "Q_WaistRFront",
    "Q_LThighMedial", "Q_LKneeOut", "Q_LKneeIn", "Q_LShinFrontHigh", "Q_LAnkleOut",
    "Q_LHeelBack", "Q_LForefoot2", "Q_LForefoot5", "Q_RThighMedial", "Q_RKneeOut",
    "Q_RKneeIn", "Q_RShinFrontHigh", "Q_RAnkleOut", "Q_RHeelBack", "Q_RForefoot2",
    "Q_RForefoot5", "Q_LSips", "Q_RSips", "Q_LThighHigh", "Q_LThighLow",
    "Q_RThighHigh", "Q_RThighLow", "Q_RShinFrontLow", "Q_LShinSide",
    "Q_LShinFrontLow", "Q_LForefoot1", "Q_RForefoot1", "Q_LAnkleIn", "Q_RAnkleIn",
    "Q_ChestLow"
]

def apply_butterworth_filter(order=4, cutoff=6.0):
    ids = {
        qtm.data.object.trajectory.find_trajectory(lbl)
        for lbl in trajectory_labels
    } - {None}

    if not ids:
        print("No trajectories found.")
        return

    for obj_id in ids:
        name = qtm.data.object.trajectory.get_label(obj_id)
        print(f"Filtering: {name}")
        ranges = qtm.data.series._3d.get_sample_ranges(obj_id)
        if not ranges or ranges[-1]["start"] == ranges[-1]["end"]:
            print(f"  skipping {name}")
            continue

        qtm.data.object.trajectory.smooth_trajectory(
            obj_id,
            "butterworth",
            ranges[-1],
            {"filter_order": order, "cutoff_frequency": cutoff}
        )
        print(f"  done.")

'''
def apply_forceplate_filter(order=10, cutoff=1.0, fs=1000):
    
    
    series_ids = qtm.data.series.force.get_series_ids()
    if not series_ids:
        print("No force data.")
        return

    sr = qtm.data.series.force.get_sample_ranges(series_ids[0])
    start, end = sr[0]["start"], sr[-1]["end"]
    data = qtm.data.series.force.get_samples(series_ids[0], {"start": start, "end": end})

    # extract components
    F = np.array([d["force"] for d in data])
    b, a = butter(order, cutoff/(0.5*fs), btype='low')
    Ff = filtfilt(b, a, F, axis=0)

    # write back
    new = []
    for i, d in enumerate(data):
        new.append({
            "force": Ff[i].tolist(),
            "moment": d.get("moment"),
            "center_of_pressure": d.get("center_of_pressure")
        })
    qtm.data.series.force.set_samples(series_ids[0], {"start": start, "end": end}, new)
    print("Forceplate filter applied.")
    '''
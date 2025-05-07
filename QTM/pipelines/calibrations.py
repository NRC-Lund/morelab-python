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

_local_offset = None

def vector_norm(v):
    return v / np.linalg.norm(v)

def static_calibration():
    global _local_offset
    print("Static calibration…")
    ids = {qtm.data.object.trajectory.find_trajectory(n) for n in
           ("Q_HeadFront","Q_HeadL","Q_HeadR")}
    if None in ids:
        print("Missing trajectories.")
        return

    rng = qtm.gui.timeline.get_measured_range()
    frame = (rng["start"] + rng["end"]) // 2
    front, left, right = (
        np.array(qtm.data.series._3d.get_sample(i, frame)["position"])
        for i in ids
    )
    origin = (left + right) / 2
    y = vector_norm(right - left)
    raw_x = front - origin
    x = -vector_norm(raw_x - np.dot(raw_x, y) * y)
    z = vector_norm(np.cross(y, x))
    R = np.column_stack((x, y, z))
    _local_offset = R.T.dot(front - origin)
    print("Static offset:", _local_offset)

def dynamic_calibration():
    global _local_offset
    if _local_offset is None:
        print("Run static_calibration first.")
        return

    left_id = qtm.data.object.trajectory.find_trajectory("Q_HeadL")
    right_id = qtm.data.object.trajectory.find_trajectory("Q_HeadR")
    if None in (left_id, right_id):
        print("Missing head markers.")
        return

    label = "Q_HeadFront"
    fid = qtm.data.object.trajectory.find_trajectory(label)
    if fid is None:
        fid = qtm.data.object.trajectory.add_trajectory(label)
    rng = qtm.gui.timeline.get_measured_range()
    global_up = np.array([0,0,1])
    out = []
    for f in range(rng["start"], rng["end"]+1):
        l = np.array(qtm.data.series._3d.get_sample(left_id,f)["position"])
        r = np.array(qtm.data.series._3d.get_sample(right_id,f)["position"])
        origin = (l+r)/2
        y = vector_norm(r-l)
        x = -vector_norm(np.cross(y, global_up))
        z = vector_norm(np.cross(x, y))
        R = np.column_stack((x, y, z))
        pos = origin + R.dot(_local_offset)
        out.append({"position": pos.tolist(), "residual": 0.0})
    qtm.data.series._3d.set_samples(fid, rng, out)
    print("Dynamic trajectory:", label)

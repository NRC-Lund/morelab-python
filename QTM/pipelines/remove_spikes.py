import sys
import os
import inspect

# Set up QTM Python API
this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if this_dir not in sys.path:
    sys.path.append(this_dir)
import qtm
from helpers.traj import _calc_marker_acceleration

def remove_spikes():
    accel_threshold = 250 # arbitrary value, can be adjusted
    total_frames = qtm.gui.timeline.get_measured_range()
    last_frame = total_frames['end']
    ids = qtm.data.series._3d.get_series_ids()

    # Delete all gap-filled parts of trajectories first
    for id in ids:
        label = qtm.data.object.trajectory.get_label(id)
        gap_filled_parts = []
        parts = qtm.data.object.trajectory.get_parts(id)
        for part, item in enumerate(parts):
            if item["type"] == "filled":
                gap_filled_parts.append(part)
        for part in reversed(gap_filled_parts):
            qtm.data.object.trajectory.delete_parts(id, [part])
            print(f"part {part} was deleted from {label} due to being gap-filled")
        if qtm.data.object.trajectory.get_part_count(id) == 0:
            qtm.data.object.trajectory.delete_trajectory(id)
            print(f"trajectory {id} deleted as all parts were removed from it")

    # identify and remove spikes and high residuals from trajectories
    ids = qtm.data.series._3d.get_series_ids()
    for id in ids:
        dt = 1/qtm.data.series._3d.get_frequency(id)
        label = qtm.data.object.trajectory.get_label(id)
        traj_details = qtm.data.series._3d.get_sample_range(id)
        traj_start = traj_details['start']
        traj_end = traj_details['end']
        traj_range = range(traj_start, traj_end+1)
        frames_to_delete = []
        for frame in traj_range:
            marker_details = qtm.data.series._3d.get_sample(id, frame)
            if marker_details == None:
                continue # skips to next frame if no marker data for current frame
            else:
                marker_accel = _calc_marker_acceleration(id, frame, dt)/1000 # divide by 1000 to convert from millimetres to metres
            if marker_accel > accel_threshold and frame != 0 and frame != last_frame: # first and last frames may have high acceleration due to lack of data
                print(f"Spike detected in {label} at frame {frame}")
                frames_to_delete.append(frame)
        for frame in frames_to_delete:
            qtm.data.series._3d.delete_sample(id, frame)
            print(f"frame {frame} in {label} was deleted")

        # Re-fill gaps (currently disabled due to error where the polynomial method requires marker data before and after the gap)
        # gap_ranges = qtm.data.series._3d.get_gap_ranges(id)
        # for gap in gap_ranges:
        #     qtm.data.object.trajectory.fill_trajectory(id, "polynomial", gap)
        
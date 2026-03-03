import qtm
from helpers.traj import _calc_marker_acceleration

def split_trajectory_at_spikes():
    accel_threshold = 150 # arbitrary value, can be adjusted
    total_frames = qtm.gui.timeline.get_measured_range()
    last_frame = total_frames['end']
    
    trajectories_selected = qtm.gui.selection.get_selections("trajectory")
    for trajectory in trajectories_selected:

        # Delete all gap-filled parts of trajectories first
        id = trajectory['id']
        dt = 1/qtm.data.series._3d.get_frequency(id)
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
            continue

        # identify spikes and split trajectories
        part_no = qtm.data.object.trajectory.get_part_count(id)
        print(f"All gap-filled parts removed from {label}. {part_no} remaining parts")
        traj_details = qtm.data.series._3d.get_sample_range(id)
        traj_start = traj_details['start']
        traj_end = traj_details['end']
        traj_range = range(traj_start, traj_end+1)
        for frame in traj_range:
            marker_details = qtm.data.series._3d.get_sample(id, frame)
            if marker_details == None:
                continue # skips to next frame if no marker data for current frame
            else:
                marker_accel = _calc_marker_acceleration(id, frame, dt)/1000 # divide by 1000 to convert from millimetres to metres
            if marker_accel > accel_threshold and frame != 0 and frame != last_frame: # first and last frames may have high acceleration due to lack of data
                print(f"Acceleration spike ({marker_accel} m/s*s) detected at frame {frame}")
                qtm.data.object.trajectory.split_part(id, frame -1)
        part_no_new = qtm.data.object.trajectory.get_part_count(id)
        print(f"{part_no_new} parts in {label} after splitting trajectory")
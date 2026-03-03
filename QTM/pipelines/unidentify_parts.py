import qtm
def unidentify_parts():
    trajectories_selected = qtm.gui.selection.get_selections("trajectory")
    frame = qtm.gui.timeline.get_current_frame()

    for trajectory in trajectories_selected:
        id = trajectory['id']
        label = qtm.data.object.trajectory.get_label(id)
        print(f"Trajectory: {label}")
        number_parts = qtm.data.object.trajectory.get_part_count(id)
        print(f"Number of Parts = {number_parts}")
        for part in range(number_parts):
            part_details = qtm.data.object.trajectory.get_part(id, part)
            start = part_details['range']['start']
            end = part_details['range']['end']
            if start <= frame <= end:
                print(f"Part to be de-identified = {part +1}")
                new_id = qtm.data.object.trajectory.add_trajectory()
                qtm.data.object.trajectory.move_parts(id, new_id, [part])
                break



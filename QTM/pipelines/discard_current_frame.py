import qtm
def discard_current_frame():
    trajectories_selected = qtm.gui.selection.get_selections("trajectory")
    frame = qtm.gui.timeline.get_current_frame()

    for trajectory in trajectories_selected:
        id = trajectory['id']
        label = qtm.data.object.trajectory.get_label(id)
        print(f"Splitting trajectory: {label}")
        number_parts = qtm.data.object.trajectory.get_part_count(id)
        print(f"Number of Parts = {number_parts}")
        for part in range(number_parts):
            part_details = qtm.data.object.trajectory.get_part(id, part)
            start = part_details['range']['start']
            end = part_details['range']['end']
            if start <= frame <= end:
                if start == frame:
                    qtm.data.object.trajectory.split_part(id, frame)
                    print(f"Frame at start of part. Discarding.")
                    print(part)
                    qtm.data.object.trajectory.delete_parts(id, [part])
                elif end == frame:
                    qtm.data.object.trajectory.split_part(id, frame -1)
                    print(f"Frame at end of part. Discarding.")
                    print(part +1)
                    qtm.data.object.trajectory.delete_parts(id, [part +1])
                elif start < frame < end:
                    qtm.data.object.trajectory.split_part(id, frame -1)
                    qtm.data.object.trajectory.split_part(id, frame)
                    print(f"Frame within part. Discarding.")
                    print(part +1)
                    qtm.data.object.trajectory.delete_parts(id, [part +1])
            break
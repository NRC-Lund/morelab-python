import qtm

def breakdown_trajectories():
    trajectories_selected = qtm.gui.selection.get_selections("trajectory")
    new_ids = []

    for trajectory in trajectories_selected:
        id = trajectory['id']
        number_parts = qtm.data.object.trajectory.get_part_count(id)
        for part in reversed(range(number_parts)):
            new_id = qtm.data.object.trajectory.add_trajectory()
            new_ids.append(new_id)
            qtm.data.object.trajectory.move_parts(id, new_id, [part])
    print(new_ids)
    return new_ids

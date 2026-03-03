import qtm
from helpers.traj import get_unlabeled_marker_ids

def select_unidentified_markers():
    frame = qtm.gui.timeline.get_current_frame()
    unlabeled_ids = get_unlabeled_marker_ids()
    to_be_selected = []
    for id in unlabeled_ids:
        number_parts = qtm.data.object.trajectory.get_part_count(id)
        for part in range(number_parts):
            part_details = qtm.data.object.trajectory.get_part(id, part)
            start = part_details['range']['start']
            end = part_details['range']['end']
            if start <= frame <= end:
                to_be_selected.append((id,part))
                break

    for selected_id, selected_part in to_be_selected:
        selection = [{"type": "trajectory", "id": selected_id, "part_index": selected_part}]
        qtm.gui.selection.select(selection)
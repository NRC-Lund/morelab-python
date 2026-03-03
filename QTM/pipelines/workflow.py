import sys, os, inspect, importlib
import qtm

from helpers.traj import get_labeled_marker_ids, get_unlabeled_marker_ids, _calc_marker_acceleration

import pipelines
from pipelines.breakdown_trajectories import breakdown_trajectories
importlib.reload(pipelines.breakdown_trajectories) # Reload to clear cache.

from pipelines.auto_label import \
    gui_auto_label_labelled, \
    gui_auto_label_unlabelled, \
    gui_auto_label_selected_trajectories, \
    gui_sal  
importlib.reload(pipelines.auto_label) # Reload to clear cache.

def apply_AIM_model():
    title = "Select AIM Model to Apply"
    filters = ["AIM files (*.qam)"]
    multiselect = False
    directory = qtm.settings.directory.get_aim_directory()
    model = qtm.gui.dialog.show_open_file_dialog(title, filters, multiselect, directory)
    model_path = model[0]

    qtm.settings.processing.aim.set_model_is_applied("project", model_path, True)
    settings = qtm.settings.processing.aim.get_settings("project")
    settings['keep_existing_labels'] = False
    qtm.processing.apply_aim(settings)

def apply_AIM_model_to_unlabelled():
    title = "Select AIM Model to Apply"
    filters = ["AIM files (*.qam)"]
    multiselect = False
    directory = qtm.settings.directory.get_aim_directory()
    model = qtm.gui.dialog.show_open_file_dialog(title, filters, multiselect, directory)
    model_path = model[0]

    qtm.settings.processing.aim.set_model_is_applied("project", model_path, True)
    settings = qtm.settings.processing.aim.get_settings("project")
    settings['keep_existing_labels'] = True
    qtm.processing.apply_aim(settings)

def delete_gap_filled_parts():
    marker_ids = qtm.data.object.trajectory.get_trajectory_ids()
    for id in marker_ids:
        part_count = qtm.data.object.trajectory.get_part_count(id)
        for count in reversed(range(part_count)):
            part = qtm.data.object.trajectory.get_part(id, count)
            if part['type'] == "filled":
                qtm.data.object.trajectory.delete_parts(id, [count])

def unlabel_spikes():
    accel_threshold = 150 # arbitrary value, can be adjusted
    total_frames = qtm.gui.timeline.get_measured_range()
    last_frame = total_frames['end']

    marker_ids = get_labeled_marker_ids() # redefine after previous changes
    for id in marker_ids:
        label = qtm.data.object.trajectory.get_label(id)
        dt = 1/qtm.data.series._3d.get_frequency(id)
        part_count = qtm.data.object.trajectory.get_part_count(id)
        for count in reversed(range(part_count)):
            part = qtm.data.object.trajectory.get_part(id, count)
            start = part['range']['start']
            end = part['range']['end']
            for frame in range(start,end +1):
                marker_details = qtm.data.series._3d.get_sample(id, frame)
                if marker_details == None:
                    continue # skips to next frame if no marker data for current frame
                else:
                    marker_accel = round(_calc_marker_acceleration(id, frame, dt)/1000,2) # divide by 1000 to convert from millimetres to metres
                if marker_accel > accel_threshold and frame != 0 and frame != last_frame: # first and last frames may have high acceleration due to lack of data
                    print(f"Acceleration spike ({marker_accel} m/s*s) detected at frame {frame}, part {count} in {label} unlabelled")
                    new_id = qtm.data.object.trajectory.add_trajectory()
                    qtm.data.object.trajectory.move_parts(id, new_id, [count])

def workflow():
    apply_AIM_model()
    delete_gap_filled_parts()

    # # Update label prefix
    # ids = qtm.data.object.trajectory.get_trajectory_ids()
    # labels = [qtm.data.object.trajectory.get_label(id) for id in ids if qtm.data.object.trajectory.get_label(id)]
    # current_prefix = [label.split("_")[0] for label in labels][0]
    # #ask to change prefix if necessary
    # new_prefix = qtm.gui.dialog.show_string_input_dialog("Set prefix", "Change prefix to:", current_prefix)
    # for id in ids:
    #     label = qtm.data.object.trajectory.get_label(id)
    #     if label:
    #         new_label = new_prefix + "_" + "_".join(label.split("_")[1:])
    #         qtm.data.object.trajectory.set_label(id, new_label)
    # print(f"Prefix updated to {new_prefix}")

    # Run auto-label on labelled markers to unidentify errors
    title = "Select npz file"
    filters = ["Reference distribution files (*.npz)"]
    multiselect = False
    path = qtm.file.get_path()
    directory = os.path.dirname(path)
    fname_npz = qtm.gui.dialog.show_open_file_dialog(title, filters, multiselect, directory)
    gui_auto_label_labelled(fname_npz[0])


    # Detect acceleration spikes in labeled trajectories and unidentify parts (backup to auto-label part size limitation)
    # unlabel_spikes()

    # # Run skeleton solver
    # title = "Skeleton Calibration Check"
    # message = "Has the skeleton been calibrated using the static calibration trial?"
    # buttons = ["Yes", "No"]
    # calibration_check = qtm.gui.dialog.show_message_box(title, message, buttons)
    # if calibration_check == "No":
    #     print("Skeleton must first be calibrated in static trial before applying to dynamic trials. Aborting.")
    #     sys.exit()
    # skeleton_settings = qtm.settings.processing.skeleton.get_settings("project")
    # qtm.processing.solve_skeletons(skeleton_settings)

    # # Run SAL to unidentify errors in labelling
    # title = "SAL Reference Distribution Check"
    # message = "Has a SAL reference distribution file been generated?"
    # buttons = ["Yes", "No"]
    # SAL_check = qtm.gui.dialog.show_message_box(title, message, buttons)
    # if SAL_check == "No":
    #     print("SAL reference distribution file must be generated before running SAL. Aborting")
    #     sys.exit()
    # 
    # gui_sal()

    # Select each unidentified trajectory and split it
    unlabeled_ids = get_unlabeled_marker_ids()
    for id in unlabeled_ids:
        qtm.gui.selection.set_selections([{"type": "trajectory", "id": id}])
        breakdown_trajectories()

    # # Re-apply AIM model to unlabelled only
    apply_AIM_model_to_unlabelled()

    # # Run auto-labeller on remaining unlabelled (will stop at parts less than 20 frames)
    # gui_auto_label_unlabelled()

    # Force auto-labeller to run on smaller parts
    # Get all unidentified trajectories
    new_unlabeled_ids = get_unlabeled_marker_ids()
    selection = []

    # Relabel all unidentified parts
    for id in new_unlabeled_ids:
        # Select all parts
        selected = qtm.gui.selection.set_selections([{"type": "trajectory", "id": id}])
        selection.append(selected)

        # Run auto-label on selected trajectories
        gui_auto_label_selected_trajectories(fname_npz[0])

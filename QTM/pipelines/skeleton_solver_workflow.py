import qtm
import os

def set_prefix(new_prefix):
    ids = qtm.data.object.trajectory.get_trajectory_ids()
    for id in ids:
        label = qtm.data.object.trajectory.get_label(id)

        if not label:
            continue

        parts = label.split("_", 1)

        # remove existing prefix if present
        suffix = parts[1] if len(parts) > 1 else parts[0]
        new_label = f"{new_prefix}_{suffix}"
        qtm.data.object.trajectory.set_label(id, new_label)

def calibrate_skeleton_new_prefix(file: str) -> tuple[str, str]:
    if qtm.file.is_open():
        qtm.file.close()

    qtm.file.open(file)
    path = os.path.dirname(qtm.file.get_path())

    ids = qtm.data.object.trajectory.get_trajectory_ids()
    labels = [qtm.data.object.trajectory.get_label(id) for id in ids if qtm.data.object.trajectory.get_label(id)]
    current_prefix = [label.split("_")[0] for label in labels][0]

    #ask to change prefix if necessary
    new_prefix = qtm.gui.dialog.show_string_input_dialog(
    "Set prefix",
    f"Change prefix to:\nCurrent path: {path}",
    current_prefix
)
    
    set_prefix(new_prefix)

    print(f"Use {new_prefix} prefix")
    qtm.file.save()
    qtm.gui.send_command("calibrate_skeletons")

    while True:
        #wait that the dialog box is closed
        try:
            if qtm.file.is_dirty():
                #sqve only when the calibration is finished
                qtm.file.save()
                break
        except RuntimeError:
            pass

    qtm.file.close()

    return new_prefix, path

def solve_skeletons(file: str, new_prefix: str):
    if qtm.file.is_open():
        qtm.file.close()

    qtm.file.open(file)


    # change prefix to match static trial
    set_prefix(new_prefix)

    # solve skeleton using skeleton instance calibrated in static trial
    settings = qtm.settings.processing.skeleton.get_settings("project")
    qtm.processing.solve_skeletons(settings)
    
    qtm.file.save()
    qtm.file.close()

# -------------------------------------------------
# MAIN WORKFLOW
# -------------------------------------------------

def main():

    parent_folder = qtm.gui.dialog.show_string_input_dialog(
    "Parent data folder",
    "Paste the path to the parent data folder:",
    os.path.expanduser("~")
)

    if not parent_folder:
        return

    parent_folder = parent_folder.strip()

    if not os.path.isdir(parent_folder):
        qtm.gui.dialog.show_message_box(
        "Invalid path",
        f"{parent_folder} is not a valid directory."
    )
        return
    
    while True:
    
    # Select static file
        title = "Select static trial for skeleton calibration"
        filters = ["QTM files (*.qtm)"]
        multiselect = False
        static = qtm.gui.dialog.show_open_file_dialog(title, filters, multiselect, parent_folder)

        if not static:
            return

        if isinstance(static, (list, tuple)):
            static = static[0]

        # open static file, change prefix, calibrate skeleton, save
        new_prefix, path = calibrate_skeleton_new_prefix(static)

        # select dynamic trials
        title = "Select dynamic trials for skeleton solving"
        filters = ["QTM files (*.qtm)"]
        multiselect = True
        dynamic_trials = qtm.gui.dialog.show_open_file_dialog(title, filters, multiselect, path)

        if not dynamic_trials:
            return

        # Loop through dyanimc trials and solve skeletons
        for trial in dynamic_trials:
            solve_skeletons(trial, new_prefix)

        # ---- Ask whether to process another participant ----
        again = qtm.gui.dialog.show_message_box(
            "Participant complete",
            "Process another participant?",
            ["Yes", "No"]
        )

        if again != "Yes":
            break



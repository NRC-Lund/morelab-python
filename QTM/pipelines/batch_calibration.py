import qtm
import time


def batch_calibrate_skeletons():
    files = qtm.gui.dialog.show_open_file_dialog("Load files for calibration", ["QTM files (*.qtm)"], True)
    if not files:
        print("No file selected, aborting")
        return
    else:
        for file in files:
            calibrate_file(file)


def calibrate_file(file: str):
    if qtm.file.is_open():
        qtm.file.close()
    qtm.file.open(file)
    ids = qtm.data.object.trajectory.get_trajectory_ids()
    labels = [qtm.data.object.trajectory.get_label(id) for id in ids if qtm.data.object.trajectory.get_label(id)]
    current_prefix = [label.split("_")[0] for label in labels][0]
    #ask to change prefix if necessary
    new_prefix = qtm.gui.dialog.show_string_input_dialog("Set prefix", "Change prefix to:", current_prefix)
    for id in ids:
        label = qtm.data.object.trajectory.get_label(id)
        if label:
            new_label = new_prefix + "_" + "_".join(label.split("_")[1:])
            qtm.data.object.trajectory.set_label(id, new_label)
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


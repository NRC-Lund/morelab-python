import os

import qtm


# Open one QTM file from a given starting folder.
def select_qtm_file(title, initial_folder):
    file = qtm.gui.dialog.show_open_file_dialog(
        title,
        ["QTM files (*.qtm)"],
        False,
        initial_folder,
    )
    if isinstance(file, (list, tuple)):
        return file[0] if file else ""
    return file or ""


# Ask before switching files if the current QTM file has unsaved changes.
def save_or_continue():
    if not qtm.file.is_open() or not qtm.file.is_dirty():
        return True

    choice = qtm.gui.dialog.show_message_box(
        "Unsaved changes",
        "The currently open file has unsaved changes.",
        ["Save and continue", "Continue without saving", "Cancel"],
    )
    if choice == "Save and continue":
        qtm.file.save()
    return choice != "Cancel"


# Choose the dynamic trial to inspect.
def open_dynamic_trial():
    message = "First select a dynamic trial with the missing marker(s)."
    if qtm.file.is_open():
        choice = qtm.gui.dialog.show_message_box(
            "Select dynamic trial",
            message,
            ["Use current file", "Open new file", "Cancel"],
        )
        if choice == "Cancel":
            return ""
        if choice == "Use current file":
            return qtm.file.get_path()
        if not save_or_continue():
            return ""
    else:
        choice = qtm.gui.dialog.show_message_box(
            "Select dynamic trial",
            message,
            ["Open file", "Cancel"],
        )
        if choice != "Open file":
            return ""

    initial_folder = (
        os.path.dirname(qtm.file.get_path())
        if qtm.file.is_open()
        else os.path.expanduser("~")
    )
    dynamic_file = select_qtm_file("Select dynamic trial", initial_folder)
    if dynamic_file:
        if qtm.file.is_open():
            qtm.file.close()
        qtm.file.open(dynamic_file)
    return dynamic_file


def format_marker_summary(missing, almost_missing):
    return (
        f"Missing markers: {', '.join(missing) if missing else 'none'}\n"
        f"Almost missing markers: "
        f"{', '.join(almost_missing) if almost_missing else 'none'}"
    )

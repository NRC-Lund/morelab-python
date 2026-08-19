import os

import qtm

from . import marker_filtering


STATE_FILE = os.path.join(
    os.path.dirname(__file__), "batch_marker_filtering_state.txt")


def load_last_folder():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as state_file:
            return state_file.read().strip()
    except OSError:
        return os.path.expanduser("~")


def save_last_folder(folder):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as state_file:
            state_file.write(folder)
    except OSError:
        pass


def select_trials(initial_folder):
    files = qtm.gui.dialog.show_open_file_dialog(
        "Select trials for marker filtering",
        ["QTM files (*.qtm)"],
        True,
        initial_folder,
    )
    if not files:
        return []
    return list(files) if isinstance(files, (list, tuple)) else [files]


def process_trial(file, cutoff):
    if qtm.file.is_open():
        qtm.file.close()

    print(f"Opening: {file}")
    qtm.file.open(file)

    trajectory_ids = marker_filtering.get_labeled_trajectory_ids()
    incomplete = marker_filtering.get_incomplete_trajectories(trajectory_ids)
    if incomplete:
        print("Filtering was not performed on this trial.")
        print(f"Incomplete trajectories: {', '.join(incomplete)}")
        action = qtm.gui.dialog.show_message_box(
            "Incomplete trajectories found",
            "Filtering was not performed on this trial.\n"
            f"Incomplete trajectories: {', '.join(incomplete)}",
            ["Stop batch", "Skip trial"],
        )
        qtm.file.close()
        return action

    marker_filtering.apply_butterworth_filter_to_trajectories(
        trajectory_ids,
        cutoff=cutoff,
    )
    qtm.file.save()
    print(f"Saved: {file}")
    qtm.file.close()
    return "Processed"


def batch_process_marker_filtering():
    start = qtm.gui.dialog.show_message_box(
        "Batch Process Marker Filtering",
        "Selected trials will be filtered and saved in place.\n"
        "Back up your files first if you want to keep the originals.",
        ["Continue", "Cancel"],
    )
    if start != "Continue":
        return

    cutoff = marker_filtering.get_cutoff_frequency()
    initial_folder = load_last_folder()

    while True:
        files = select_trials(initial_folder)
        if not files:
            print("No files selected, aborting.")
            return

        initial_folder = os.path.dirname(files[0])
        save_last_folder(initial_folder)
        processed = []
        skipped = []

        for file in files:
            result = process_trial(file, cutoff)
            if result == "Processed":
                processed.append(file)
            elif result == "Skip trial":
                skipped.append(file)
            else:
                print("Batch marker filtering stopped.")
                return

        print("Batch marker filtering complete.")
        print(f"Processed trials: {len(processed)}")
        for file in processed:
            print(f"  processed: {file}")
        print(f"Skipped trials: {len(skipped)}")
        for file in skipped:
            print(f"  skipped: {file}")

        again = qtm.gui.dialog.show_message_box(
            "Batch marker filtering complete",
            "Process more trials?",
            ["Yes", "No"],
        )
        if again != "Yes":
            return

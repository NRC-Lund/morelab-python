import os
import time

import qtm


STATE_FILE = os.path.join(os.path.dirname(__file__), "skeleton_solver_workflow_state.txt")


def log(message: str):
    print(f"{time.strftime('%H:%M:%S')} {message}")


def show_message(title: str, message: str):
    try:
        qtm.gui.message.add_message(title, message, "info")
    except (AttributeError, RuntimeError):
        log(f"{title}: {message}")


def load_last_static_file():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as state_file:
            static_file = state_file.read().strip()
    except OSError:
        return ""

    if os.path.isfile(static_file):
        return static_file

    return ""


def save_last_static_file(static_file: str):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as state_file:
            state_file.write(static_file)
    except OSError:
        pass


def get_initial_folder():
    static_file = load_last_static_file()

    if static_file:
        return os.path.dirname(static_file)

    return os.path.expanduser("~")


def select_static_file(initial_folder: str):
    title = "Select static trial"
    filters = ["QTM files (*.qtm)"]
    static = qtm.gui.dialog.show_open_file_dialog(
        title,
        filters,
        False,
        initial_folder,
    )

    if isinstance(static, (list, tuple)):
        return static[0] if static else ""

    return static or ""


def set_prefix(new_prefix):
    ids = qtm.data.object.trajectory.get_trajectory_ids()

    for trajectory_id in ids:
        label = qtm.data.object.trajectory.get_label(trajectory_id)

        if not label:
            continue

        parts = label.split("_", 1)
        suffix = parts[1] if len(parts) > 1 else parts[0]
        qtm.data.object.trajectory.set_label(trajectory_id, f"{new_prefix}_{suffix}")


def get_current_prefix():
    ids = qtm.data.object.trajectory.get_trajectory_ids()
    labels = [
        qtm.data.object.trajectory.get_label(trajectory_id)
        for trajectory_id in ids
        if qtm.data.object.trajectory.get_label(trajectory_id)
    ]

    if not labels:
        return ""

    return labels[0].split("_", 1)[0]


def prepare_static_file(file: str) -> tuple[str, str]:
    if qtm.file.is_open():
        log("Closing previously open file")
        qtm.file.close()

    log(f"Opening static trial: {file}")
    qtm.file.open(file)
    path = os.path.dirname(qtm.file.get_path())

    current_prefix = get_current_prefix()
    new_prefix = qtm.gui.dialog.show_string_input_dialog(
        "Set prefix",
        f"Change prefix to:\nCurrent path: {path}",
        current_prefix,
    )

    if not new_prefix:
        qtm.file.close()
        return "", path

    set_prefix(new_prefix)

    log(f"Using prefix: {new_prefix}")
    log("Saving static trial before calibration")
    qtm.file.save()
    log("Static trial prepared and saved")
    save_last_static_file(file)

    log("Starting skeleton calibration command")
    qtm.gui.send_command("calibrate_skeletons")
    log("Skeleton calibration command started")

    show_message(
        "Static calibration started",
        "When QTM finishes skeleton calibration, save the static file manually. "
        "Then run this script again and choose Solve dynamics.",
    )

    return new_prefix, path


def get_static_prefix_and_path(file: str) -> tuple[str, str]:
    if qtm.file.is_open():
        log("Closing previously open file")
        qtm.file.close()

    log(f"Opening calibrated static trial: {file}")
    qtm.file.open(file)
    path = os.path.dirname(qtm.file.get_path())
    prefix = get_current_prefix()
    log(f"Using prefix from calibrated static trial: {prefix}")

    qtm.file.close()
    log("Calibrated static trial closed")

    return prefix, path


def solve_skeletons(file: str, new_prefix: str):
    if qtm.file.is_open():
        log("Closing previously open file")
        qtm.file.close()

    log(f"Opening dynamic trial: {file}")
    qtm.file.open(file)

    set_prefix(new_prefix)

    log(f"Solving skeletons for dynamic trial: {file}")
    settings = qtm.settings.processing.skeleton.get_settings("project")
    qtm.processing.solve_skeletons(settings)
    log(f"Skeleton solving complete: {file}")

    log(f"Saving dynamic trial: {file}")
    qtm.file.save()
    log(f"Dynamic trial saved: {file}")

    log(f"Closing dynamic trial: {file}")
    qtm.file.close()
    log(f"Dynamic trial closed: {file}")


def main():
    log("Starting skeleton solver workflow")

    while True:
        last_static_file = load_last_static_file()
        workflow = qtm.gui.dialog.show_message_box(
            "Skeleton solver workflow",
            "Choose workflow step.",
            ["Prepare static", "Solve dynamics", "Cancel"],
        )

        if workflow == "Cancel":
            return

        if workflow == "Solve dynamics" and last_static_file:
            use_last = qtm.gui.dialog.show_message_box(
                "Static trial",
                f"Use previous static trial?\n{last_static_file}",
                ["Yes", "Select different", "Cancel"],
            )

            if use_last == "Cancel":
                return

            static = last_static_file if use_last == "Yes" else select_static_file(get_initial_folder())
        else:
            static = select_static_file(get_initial_folder())

        if not static:
            return

        save_last_static_file(static)

        if workflow == "Prepare static":
            new_prefix, _ = prepare_static_file(static)

            if not new_prefix:
                return

            return

        new_prefix, path = get_static_prefix_and_path(static)

        title = "Select dynamic trials for skeleton solving"
        filters = ["QTM files (*.qtm)"]
        dynamic_trials = qtm.gui.dialog.show_open_file_dialog(
            title,
            filters,
            True,
            path,
        )

        if not dynamic_trials:
            return

        for trial in dynamic_trials:
            solve_skeletons(trial, new_prefix)

        log("Participant complete")
        again = qtm.gui.dialog.show_message_box(
            "Participant complete",
            "Process another participant?",
            ["Yes", "No"],
        )

        if again != "Yes":
            break

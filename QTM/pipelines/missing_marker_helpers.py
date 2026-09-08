import os

import numpy as np
import qtm

from .gap_fill_relational import get_marker_prefix
from .full_body_gap_fill_relational import segments


ALMOST_MISSING_THRESHOLD = 0.10


# Build the expected marker list from the same segment definitions used for
# full-body relational gap filling.
def get_expected_marker_names():
    marker_names = []
    for _, segment in segments:
        marker_names += [
            marker for marker in segment.base_marker_names
            if marker not in marker_names
        ]
    return marker_names


# Merge the segment reconstruction rules used by the full-body marker set.
def get_marker_rules():
    rules = {}
    for _, segment in segments:
        for marker, marker_rules in segment.base_gap_fill_rules.items():
            rules.setdefault(marker, []).extend(marker_rules)
    return rules


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


# Choose the dynamic trial to inspect. The user can keep the currently open
# file, open a different one, or cancel.
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

    initial_folder = os.path.dirname(qtm.file.get_path()) if qtm.file.is_open() else os.path.expanduser("~")
    dynamic_file = select_qtm_file("Select dynamic trial", initial_folder)
    if dynamic_file:
        if qtm.file.is_open():
            qtm.file.close()
        qtm.file.open(dynamic_file)
    return dynamic_file


# Count how many frames contain marker samples within the measured range.
def count_samples_in_range(trajectory_id, measured_range):
    count = 0
    for sample_range in qtm.data.series._3d.get_sample_ranges(trajectory_id):
        start = max(sample_range["start"], measured_range["start"])
        end = min(sample_range["end"], measured_range["end"])
        count += max(0, end - start + 1)
    return count


def remove_prefix(label, prefix):
    return label[len(prefix):] if label.startswith(prefix) else label


def format_marker_summary(missing, almost_missing):
    return (
        f"Missing markers: {', '.join(missing) if missing else 'none'}\n"
        f"Almost missing markers: {', '.join(almost_missing) if almost_missing else 'none'}"
    )


# Find expected markers that exist in the marker list but have no samples, or
# have very few samples. These are candidates for static-trial reconstruction.
def scan_missing_markers():
    prefix = get_marker_prefix()
    measured_range = qtm.gui.timeline.get_measured_range()
    total_frames = measured_range["end"] - measured_range["start"] + 1
    missing = []
    almost_missing = []
    not_in_marker_list = []

    for marker in get_expected_marker_names():
        label = f"{prefix}{marker}"
        trajectory_id = qtm.data.object.trajectory.find_trajectory(label)
        if trajectory_id is None:
            not_in_marker_list.append(label)
            continue

        sample_count = count_samples_in_range(trajectory_id, measured_range)
        if sample_count == 0:
            missing.append(label)
        elif sample_count / total_frames < ALMOST_MISSING_THRESHOLD:
            almost_missing.append(label)

    candidates = missing + almost_missing
    return prefix, candidates, missing, almost_missing, not_in_marker_list


# Read one static-pose position per marker from the middle of the static trial.
def get_static_positions(prefix):
    positions = {}
    rng = qtm.gui.timeline.get_measured_range()
    frame = (rng["start"] + rng["end"]) // 2

    for marker in get_expected_marker_names():
        trajectory_id = qtm.data.object.trajectory.find_trajectory(f"{prefix}{marker}")
        if trajectory_id is None:
            continue
        sample = qtm.data.series._3d.get_sample(trajectory_id, frame)
        if sample and sample.get("position") is not None:
            positions[marker] = sample["position"]

    return positions


def first_static_rule(target, rules, positions):
    for rule in rules.get(target, []):
        if len(rule) == 3 and target in positions and all(marker in positions for marker in rule):
            return rule
    return None


def calculate_static_offset(target, rule, positions):
    origin, line, plane = [np.array(positions[marker]) for marker in rule]
    target_position = np.array(positions[target])

    x_axis = line - origin
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = plane - origin
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)
    return [
        float(np.dot(target_position - origin, x_axis)),
        float(np.dot(target_position - origin, y_axis)),
        float(np.dot(target_position - origin, z_axis)),
    ]


# Report whether each dynamic candidate has enough static marker information to
# define a virtual marker and absolute offset.
def report_static_marker_relationships(dynamic_prefix, candidates):
    static_prefix = get_marker_prefix()
    positions = get_static_positions(static_prefix)
    rules = get_marker_rules()

    print(f"Detected static marker prefix: {static_prefix}")
    for label in candidates:
        target = remove_prefix(label, dynamic_prefix)
        rule = first_static_rule(target, rules, positions)
        if rule:
            references = [f"{dynamic_prefix}{marker}" for marker in rule]
            offset = calculate_static_offset(target, rule, positions)
            print(f"{label}: virtual rule available using {references}")
            print(f"{label}: static offset mm {offset}")
        else:
            print(f"{label}: no 3-marker static rule available")

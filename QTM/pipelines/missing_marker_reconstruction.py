import numpy as np
import qtm

from .gap_fill_relational import get_marker_prefix
from .missing_marker_detection import (
    get_marker_rules,
    get_static_positions,
    remove_prefix,
)


def first_static_rule(target, rules, positions):
    for rule in rules.get(target, []):
        if (
                len(rule) == 3
                and target in positions
                and all(marker in positions for marker in rule)):
            return rule
    return None


# Express the static target marker position in the local frame defined by the
# selected origin, X-axis, and XY-plane markers.
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


# Store the marker-based reconstruction details available from the static trial.
def get_marker_reconstruction_options(dynamic_prefix, candidates):
    static_prefix = get_marker_prefix()
    positions = get_static_positions(static_prefix)
    rules = get_marker_rules()
    options = {}

    print(f"Detected static marker prefix: {static_prefix}")
    for label in candidates:
        target = remove_prefix(label, dynamic_prefix)
        rule = first_static_rule(target, rules, positions)
        if rule:
            options[target] = {
                "references": rule,
                "offset": calculate_static_offset(target, rule, positions),
            }
    return options


# Apply marker-based virtual reconstruction over the full dynamic measured range.
def apply_marker_based_reconstructions(dynamic_prefix, candidates, marker_options):
    measured_range = qtm.gui.timeline.get_measured_range()
    reconstructed = 0

    for label in candidates:
        target = remove_prefix(label, dynamic_prefix)
        if target not in marker_options:
            continue

        target_id = qtm.data.object.trajectory.find_trajectory(label)
        references = [
            qtm.data.object.trajectory.find_trajectory(f"{dynamic_prefix}{marker}")
            for marker in marker_options[target]["references"]
        ]
        if target_id is None or any(reference is None for reference in references):
            print(f"{label}: marker-based reconstruction skipped; marker missing")
            continue

        qtm.data.object.trajectory.fill_trajectory(
            target_id,
            "virtual",
            measured_range,
            {
                "origin": references[0],
                "line": references[1],
                "plane": references[2],
                "offset": marker_options[target]["offset"],
                "is_relative_offset": False,
            },
        )
        reconstructed += 1
        print(f"{label}: marker-based reconstruction applied")

    return reconstructed

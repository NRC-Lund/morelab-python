import numpy as np
import qtm

from .gap_fill_relational import get_marker_prefix
from .missing_marker_detection import get_static_positions, remove_prefix


def get_skeleton_id(prefix):
    return qtm.data.object.skeleton.find_skeleton(prefix.rstrip("_"))


def get_segment_id_by_name(skeleton_id, segment_name):
    for segment_id in qtm.data.object.skeleton.get_segment_ids(skeleton_id):
        if qtm.data.object.skeleton.get_segment_name(segment_id) == segment_name:
            return segment_id
    return None


# Map marker names to skeleton segment names for the currently solved skeleton.
def get_marker_segment_map(prefix):
    skeleton_id = get_skeleton_id(prefix)
    if skeleton_id is None:
        return {}

    marker_segments = {}
    for segment_id in qtm.data.object.skeleton.get_segment_ids(skeleton_id):
        segment_name = qtm.data.object.skeleton.get_segment_name(segment_id)
        markers = qtm.data.object.skeleton.get_segment_markers(segment_id) or []
        for marker in markers:
            name = marker.get("name") if isinstance(marker, dict) else None
            if name:
                marker_segments.setdefault(name, []).append(segment_name)
    return marker_segments


def get_marker_segments(marker_segments, prefix, target):
    return marker_segments.get(target) or marker_segments.get(f"{prefix}{target}") or []


def get_segment_global_transform(segment_id, frame):
    transform = qtm.data.series.skeleton.get_sample(segment_id, frame)
    if transform is None:
        return None

    parent_id = qtm.data.object.skeleton.get_segment_parent_id(segment_id)
    if parent_id is None:
        return np.array(transform)

    parent_transform = get_segment_global_transform(parent_id, frame)
    if parent_transform is None:
        return None
    return np.matmul(parent_transform, np.array(transform))


def transform_point(transform, point):
    point = np.array([point[0], point[1], point[2], 1.0])
    return np.matmul(transform, point)[:3]


# Store the static marker position relative to its solved skeleton segment.
def get_skeleton_reconstruction_options(dynamic_prefix, candidates, marker_options):
    static_prefix = get_marker_prefix()
    skeleton_id = get_skeleton_id(static_prefix)
    fallback_labels = [
        label for label in candidates
        if remove_prefix(label, dynamic_prefix) not in marker_options
    ]
    if skeleton_id is None:
        if fallback_labels:
            qtm.gui.dialog.show_message_box(
                "Skeleton not found",
                "Skeleton-based reconstruction is needed, but the static trial "
                "does not have a solved skeleton.",
                ["OK"],
            )
            print("Static skeleton options unavailable; no solved skeleton found.")
        return {}

    positions = get_static_positions(static_prefix)
    marker_segments = get_marker_segment_map(static_prefix)
    rng = qtm.gui.timeline.get_measured_range()
    frame = (rng["start"] + rng["end"]) // 2
    options = {}

    for label in candidates:
        target = remove_prefix(label, dynamic_prefix)
        if target in marker_options or target not in positions:
            continue

        segment_names = get_marker_segments(marker_segments, static_prefix, target)
        if not segment_names:
            continue

        segment_id = get_segment_id_by_name(skeleton_id, segment_names[0])
        transform = get_segment_global_transform(segment_id, frame)
        if transform is None:
            continue

        local_position = transform_point(np.linalg.inv(transform), positions[target])
        options[target] = {
            "segment": segment_names[0],
            "local_position": [float(value) for value in local_position],
        }
    return options


# Report which reconstruction method is available while the dynamic trial is open.
def report_reconstruction_methods(
        dynamic_prefix, candidates, marker_options, skeleton_options):
    marker_segments = get_marker_segment_map(dynamic_prefix)

    for label in candidates:
        target = remove_prefix(label, dynamic_prefix)
        if target in marker_options:
            references = [
                f"{dynamic_prefix}{marker}"
                for marker in marker_options[target]["references"]
            ]
            print(f"{label}: marker-based reconstruction available using {references}")
            print(f"{label}: static offset mm {marker_options[target]['offset']}")
        elif target in skeleton_options:
            segment = skeleton_options[target]["segment"]
            segment_names = get_marker_segments(marker_segments, dynamic_prefix, target)
            if segment in segment_names:
                print(f"{label}: skeleton reconstruction available using {segment}")
                print(
                    f"{label}: static local position mm "
                    f"{skeleton_options[target]['local_position']}")
            else:
                print(f"{label}: static skeleton rule found, but dynamic segment unavailable")
        else:
            print(f"{label}: no marker-based or skeleton-based reconstruction available")


def confirm_skeleton_reconstructions(dynamic_prefix, candidates, marker_options,
                                     skeleton_options):
    labels = [
        label for label in candidates
        if remove_prefix(label, dynamic_prefix) in skeleton_options
        and remove_prefix(label, dynamic_prefix) not in marker_options
    ]
    if not labels:
        return False

    if get_skeleton_id(dynamic_prefix) is None:
        qtm.gui.dialog.show_message_box(
            "Skeleton not found",
            "Skeleton-based reconstruction is needed, but this dynamic trial "
            "does not have a solved skeleton.",
            ["OK"],
        )
        print("Skeleton-based reconstruction skipped; no solved skeleton found.")
        return False

    choice = qtm.gui.dialog.show_message_box(
        "Skeleton reconstruction",
        "The following marker(s) do not have 3 reference markers and will be "
        "reconstructed from the solved skeleton:\n\n"
        f"{', '.join(labels)}\n\n"
        "This may be less accurate if the missing marker affected the skeleton "
        "segment pose.",
        ["Continue", "Cancel"],
    )
    return choice == "Continue"


# Apply skeleton-based reconstruction over the full dynamic measured range.
def apply_skeleton_based_reconstructions(
        dynamic_prefix, candidates, marker_options, skeleton_options):
    measured_range = qtm.gui.timeline.get_measured_range()
    frames = range(measured_range["start"], measured_range["end"] + 1)
    skeleton_id = get_skeleton_id(dynamic_prefix)
    reconstructed = 0

    if skeleton_id is None:
        print("Skeleton-based reconstruction skipped; no solved skeleton found.")
        return reconstructed

    for label in candidates:
        target = remove_prefix(label, dynamic_prefix)
        if target in marker_options or target not in skeleton_options:
            continue

        target_id = qtm.data.object.trajectory.find_trajectory(label)
        segment_id = get_segment_id_by_name(
            skeleton_id, skeleton_options[target]["segment"])
        if target_id is None or segment_id is None:
            print(f"{label}: skeleton-based reconstruction skipped")
            continue

        local_position = skeleton_options[target]["local_position"]
        samples = []
        for frame in frames:
            transform = get_segment_global_transform(segment_id, frame)
            position = transform_point(transform, local_position)
            samples.append({"position": position.tolist(), "residual": 0.0})

        qtm.data.series._3d.set_samples(target_id, measured_range, samples)
        reconstructed += 1
        print(f"{label}: skeleton-based reconstruction applied")

    return reconstructed

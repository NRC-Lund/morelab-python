import qtm

from .gap_fill_relational import get_marker_prefix
from .full_body_gap_fill_relational import segments

# Markers with fewer samples than this fraction of the measured range are
# treated as almost missing.
ALMOST_MISSING_THRESHOLD = 0.10


# Build the expected marker list from the full-body segment definitions.
def get_expected_marker_names():
    marker_names = []
    for _, segment in segments:
        marker_names += [
            marker for marker in segment.base_marker_names
            if marker not in marker_names
        ]
    return marker_names


# Merge the segment marker-reference rules used by the full-body marker set.
def get_marker_rules():
    rules = {}
    for _, segment in segments:
        for marker, marker_rules in segment.base_gap_fill_rules.items():
            rules.setdefault(marker, []).extend(marker_rules)
    return rules


def remove_prefix(label, prefix):
    return label[len(prefix):] if label.startswith(prefix) else label


# Count how many frames contain marker samples within the measured range.
def count_samples_in_range(trajectory_id, measured_range):
    count = 0
    for sample_range in qtm.data.series._3d.get_sample_ranges(trajectory_id):
        start = max(sample_range["start"], measured_range["start"])
        end = min(sample_range["end"], measured_range["end"])
        count += max(0, end - start + 1)
    return count


# Find expected markers that exist in the marker list but have no samples, or
# have very few samples.
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
        trajectory_id = qtm.data.object.trajectory.find_trajectory(
            f"{prefix}{marker}")
        if trajectory_id is None:
            continue
        sample = qtm.data.series._3d.get_sample(trajectory_id, frame)
        if sample and sample.get("position") is not None:
            positions[marker] = sample["position"]

    return positions

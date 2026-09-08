import qtm


def add_marker_prefix(marker_names, gap_fill_rules, prefix):
    marker_names = [f"{prefix}{name}" for name in marker_names]
    gap_fill_rules = {
        f"{prefix}{marker}": [
            [f"{prefix}{reference}" for reference in rule]
            for rule in rules
        ]
        for marker, rules in gap_fill_rules.items()
    }
    return marker_names, gap_fill_rules


def get_marker_prefix():
    for trajectory_id in qtm.data.object.trajectory.get_trajectory_ids():
        label = qtm.data.object.trajectory.get_label(trajectory_id)
        if label and "_" in label:
            return label.split("_")[0] + "_"
    return ""


def get_all_gap_ranges(trajectory_id, measured_range):
    gaps = list(qtm.data.series._3d.get_gap_ranges(trajectory_id))
    samples = qtm.data.series._3d.get_sample_ranges(trajectory_id)

    if not samples:
        return [measured_range.copy()]
    if samples[0]["start"] > measured_range["start"]:
        gaps.insert(0, {
            "start": measured_range["start"],
            "end": samples[0]["start"],
        })
    if samples[-1]["end"] < measured_range["end"]:
        gaps.append({
            "start": samples[-1]["end"],
            "end": measured_range["end"],
        })
    return gaps


def get_available_references(marker, marker_names, gap, margin, measured_range):
    start = max(measured_range["start"], gap["start"] - margin)
    end = min(measured_range["end"], gap["end"] + margin)
    available = []
    for reference in marker_names:
        if reference == marker:
            continue
        trajectory_id = qtm.data.object.trajectory.find_trajectory(reference)
        if trajectory_id is None:
            continue
        for sample in qtm.data.series._3d.get_sample_ranges(trajectory_id):
            if sample["start"] <= start and sample["end"] >= end:
                available.append(reference)
                break
    return available


def select_gap_fill_rule(marker, available, gap_fill_rules):
    for rule in gap_fill_rules.get(marker, []):
        if all(reference in available for reference in rule):
            keys = ("origin", "line", "plane")
            return {
                key: qtm.data.object.trajectory.find_trajectory(reference)
                for key, reference in zip(keys, rule)
            }, rule
    return None


def gap_fill_relational_pass(
        marker_names,
        gap_fill_rules,
        max_gap_length=25,
        polynomial_threshold=10,
        reference_margin=5):
    measured_range = qtm.gui.timeline.get_measured_range()
    total_filled = 0

    for marker in marker_names:
        trajectory_id = qtm.data.object.trajectory.find_trajectory(marker)
        if trajectory_id is None:
            print(f"{marker}: marker not found")
            continue
        gaps = get_all_gap_ranges(trajectory_id, measured_range)
        if not gaps:
            print(f"{marker}: no gaps detected")
            continue

        filled = polynomial = relational = too_long = no_rule = 0

        for gap in gaps:
            gap_length = gap["end"] - gap["start"]
            if gap_length < polynomial_threshold:
                try:
                    qtm.data.object.trajectory.fill_trajectory(
                        trajectory_id, "polynomial", gap)
                    filled += 1
                    polynomial += 1
                except RuntimeError as error:
                    if "polynomial fill requires two samples" not in str(error):
                        raise
                    print(
                        f"{marker}: polynomial unavailable for "
                        f"{gap['start']}-{gap['end']}; trying relational"
                    )
                else:
                    continue
            if gap_length > max_gap_length:
                too_long += 1
                continue

            available = get_available_references(
                marker, marker_names, gap, reference_margin, measured_range)
            selected_rule = select_gap_fill_rule(
                marker, available, gap_fill_rules)
            if selected_rule is None:
                no_rule += 1
                continue

            references, labels = selected_rule
            qtm.data.object.trajectory.fill_trajectory(
                trajectory_id, "relational", gap, references)
            print(f"{marker}: filled {gap['start']}–{gap['end']} using {labels}")
            filled += 1
            relational += 1

        print(
            f"{marker}: {filled} filled "
            f"({polynomial} polynomial, {relational} relational); "
            f"{too_long + no_rule} unfilled "
            f"({too_long} too long, {no_rule} without a rule)"
        )
        total_filled += filled

    return total_filled


def gap_fill_relational(
        base_marker_names,
        base_gap_fill_rules,
        max_gap_length=25,
        polynomial_threshold=10,
        reference_margin=5,
        ask_max_gap_length=True,
        max_passes=3):
    if ask_max_gap_length:
        value = qtm.gui.dialog.show_string_input_dialog(
            "Max gap fill range",
            "What is the max gap length (in frames) you'd like to fill?",
            str(max_gap_length),
        )
        max_gap_length = max_gap_length if value is None else int(value)

    marker_names, gap_fill_rules = add_marker_prefix(
        base_marker_names, base_gap_fill_rules, get_marker_prefix())

    total_filled = 0
    for pass_number in range(1, max_passes + 1):
        print(f"--- Pass {pass_number} ---")
        filled_this_pass = gap_fill_relational_pass(
            marker_names,
            gap_fill_rules,
            max_gap_length,
            polynomial_threshold,
            reference_margin,
        )
        total_filled += filled_this_pass
        print(f"Pass {pass_number}: {filled_this_pass} gaps filled")
        if filled_this_pass == 0:
            print("No new gaps filled; stopping.")
            break

    return total_filled

import qtm


def get_cutoff_frequency(default=6.0):
    value = qtm.gui.dialog.show_string_input_dialog(
        "Butterworth filter",
        "Enter cutoff frequency (Hz):",
        str(default),
    )
    return default if value is None else float(value)


def get_labeled_trajectory_ids():
    return [
        trajectory_id
        for trajectory_id in qtm.data.object.trajectory.get_trajectory_ids()
        if qtm.data.object.trajectory.get_label(trajectory_id)
    ]


def get_selected_trajectory_ids():
    return [
        selection["id"]
        for selection in qtm.gui.selection.get_selections("trajectory")
        if "id" in selection
    ]


def trajectory_is_complete(trajectory_id, measured_range):
    ranges = qtm.data.series._3d.get_sample_ranges(trajectory_id)
    return (
        len(ranges) == 1
        and ranges[0]["start"] <= measured_range["start"]
        and ranges[0]["end"] >= measured_range["end"]
    )


def get_incomplete_trajectories(trajectory_ids):
    measured_range = qtm.gui.timeline.get_measured_range()
    return [
        qtm.data.object.trajectory.get_label(trajectory_id) or str(trajectory_id)
        for trajectory_id in trajectory_ids
        if not trajectory_is_complete(trajectory_id, measured_range)
    ]


def apply_butterworth_filter_to_trajectories(
        trajectory_ids,
        cutoff=None,
        order=4):
    incomplete = get_incomplete_trajectories(trajectory_ids)
    if incomplete:
        print("Filtering was not performed on this trial.")
        print(f"Incomplete trajectories: {', '.join(incomplete)}")
        return False

    measured_range = qtm.gui.timeline.get_measured_range()
    cutoff = get_cutoff_frequency() if cutoff is None else cutoff
    for trajectory_id in trajectory_ids:
        label = qtm.data.object.trajectory.get_label(trajectory_id)
        print(f"Filtering: {label}")
        qtm.data.object.trajectory.smooth_trajectory(
            trajectory_id,
            "butterworth",
            measured_range,
            {"filter_order": order, "cutoff_frequency": cutoff},
        )
        print("  done.")
    return True


def apply_butterworth_filter_to_marker_set():
    apply_butterworth_filter_to_trajectories(get_labeled_trajectory_ids())


def apply_butterworth_filter_to_selected_trajectories():
    apply_butterworth_filter_to_trajectories(get_selected_trajectory_ids())

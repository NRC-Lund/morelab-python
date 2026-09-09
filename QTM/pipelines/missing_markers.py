import os

import qtm

from .missing_marker_helpers import (
    apply_marker_based_reconstructions,
    apply_skeleton_based_reconstructions,
    confirm_skeleton_reconstructions,
    format_marker_summary,
    get_marker_reconstruction_options,
    get_skeleton_reconstruction_options,
    open_dynamic_trial,
    report_reconstruction_methods,
    save_or_continue,
    scan_missing_markers,
    select_qtm_file,
)


# Select dynamic/static trials, classify missing markers, and apply available
# marker-based or skeleton-based reconstruction.
def create_missing_marker_from_static_trial():
    dynamic_file = open_dynamic_trial()
    if not dynamic_file:
        print("No dynamic trial selected, aborting.")
        return

    prefix, candidates, missing, almost_missing, not_in_marker_list = scan_missing_markers()
    print(f"Dynamic trial: {dynamic_file}")
    print(f"Detected marker prefix: {prefix}")
    print(f"Missing markers: {missing}")
    print(f"Almost missing markers: {almost_missing}")
    if not_in_marker_list:
        print(f"Markers not found in marker list: {not_in_marker_list}")

    if not save_or_continue():
        return

    choice = qtm.gui.dialog.show_message_box(
        "Select static trial",
        f"{format_marker_summary(missing, almost_missing)}\n\n"
        "Select the static trial you want to use to reconstruct these marker(s).",
        ["Continue", "Cancel"],
    )
    if choice != "Continue":
        return

    static_file = select_qtm_file("Select static trial", os.path.dirname(dynamic_file))
    if not static_file:
        print("No static trial selected, aborting.")
        return

    if qtm.file.is_open():
        qtm.file.close()
    qtm.file.open(static_file)
    print(f"Static trial: {static_file}")
    marker_options = get_marker_reconstruction_options(prefix, candidates)
    skeleton_options = get_skeleton_reconstruction_options(
        prefix, candidates, marker_options)

    if qtm.file.is_open():
        qtm.file.close()
    qtm.file.open(dynamic_file)
    print(f"Reopened dynamic trial: {dynamic_file}")
    report_reconstruction_methods(
        prefix, candidates, marker_options, skeleton_options)
    marker_reconstructions = apply_marker_based_reconstructions(
        prefix, candidates, marker_options)
    print(f"Marker-based reconstructions applied: {marker_reconstructions}")
    if confirm_skeleton_reconstructions(
            prefix, candidates, marker_options, skeleton_options):
        skeleton_reconstructions = apply_skeleton_based_reconstructions(
            prefix, candidates, marker_options, skeleton_options)
    else:
        skeleton_reconstructions = 0
    print(f"Skeleton-based reconstructions applied: {skeleton_reconstructions}")

    qtm.gui.dialog.show_message_box(
        "Missing marker reconstruction complete",
        "Missing marker reconstruction complete.\n\n"
        f"Marker-based reconstructions: {marker_reconstructions}\n"
        f"Skeleton-based reconstructions: {skeleton_reconstructions}\n\n"
        "Review the reconstructed marker(s) before saving the trial.",
        ["OK"],
    )

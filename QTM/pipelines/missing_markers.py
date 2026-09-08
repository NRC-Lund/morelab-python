import os

import qtm

from .missing_marker_helpers import (
    format_marker_summary,
    open_dynamic_trial,
    report_static_marker_relationships,
    save_or_continue,
    scan_missing_markers,
    select_qtm_file,
)


# Select a dynamic trial, identify missing/almost-missing markers, select the
# matching static trial, and report available static marker relationships.
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
        "Select the static trial you want to use to reconstruct the missing marker(s).\n\n"
        f"{format_marker_summary(missing, almost_missing)}",
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
    report_static_marker_relationships(prefix, candidates)
    print("Setup complete. Marker reconstruction has not been applied yet.")

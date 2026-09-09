# #############################################################################
#                               MoRe-Lab QTM Scripting                       #
#               Python Motion Data Elaboration Toolbox for QTM               #
#
# This file is part of the MoRe-Lab QTM Scripting utilities.
# Copyright (C) 2025
#
# MoRe-Lab QTM Scripting is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MoRe-Lab QTM Scripting is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Author: Pär Halje (par.halje@med.lu.se)
# Author: Nicholas Ryan (nicholas.ryan@med.lu.se)
# Author: Zachary Flahaut (zflah021@uottawa.ca)
# Author: Victor Leroy (victor.leroy@med.lu.se)
# #############################################################################

import sys, os, inspect, importlib

# Ensure project root on path
this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if this_dir not in sys.path:
    sys.path.append(this_dir)

# Impoprt Qualisys modules
import qtm
from helpers.printing import try_print_except
from helpers.menu_tools import add_menu_item

# Import MoRe-Lab modules
import pipelines.marker_filtering
importlib.reload(pipelines.marker_filtering) # Reload to clear cache.
import pipelines.batch_marker_filtering
importlib.reload(pipelines.batch_marker_filtering) # Reload to clear cache.
import pipelines.gap_fill_relational
importlib.reload(pipelines.gap_fill_relational) # Reload shared gap-filling logic.
import pipelines.pelvis_gap_fill_relational
importlib.reload(pipelines.pelvis_gap_fill_relational) # Reload to clear cache.
import pipelines.head_gap_fill_relational
importlib.reload(pipelines.head_gap_fill_relational) # Reload to clear cache.
import pipelines.torso_gap_fill_relational
importlib.reload(pipelines.torso_gap_fill_relational) # Reload to clear cache.
import pipelines.arm_gap_fill_relational
importlib.reload(pipelines.arm_gap_fill_relational) # Reload to clear cache.
import pipelines.thigh_gap_fill_relational
importlib.reload(pipelines.thigh_gap_fill_relational) # Reload to clear cache.
import pipelines.shank_gap_fill_relational
importlib.reload(pipelines.shank_gap_fill_relational) # Reload to clear cache.
import pipelines.foot_gap_fill_relational
importlib.reload(pipelines.foot_gap_fill_relational) # Reload to clear cache.
import pipelines.full_body_gap_fill_relational
importlib.reload(pipelines.full_body_gap_fill_relational) # Reload to clear cache.
import pipelines.missing_marker_detection
importlib.reload(pipelines.missing_marker_detection) # Reload to clear cache.
import pipelines.missing_marker_dialogs
importlib.reload(pipelines.missing_marker_dialogs) # Reload to clear cache.
import pipelines.missing_marker_reconstruction
importlib.reload(pipelines.missing_marker_reconstruction) # Reload to clear cache.
import pipelines.missing_marker_skeleton
importlib.reload(pipelines.missing_marker_skeleton) # Reload to clear cache.
import pipelines.missing_marker_helpers
importlib.reload(pipelines.missing_marker_helpers) # Reload to clear cache.
import pipelines.missing_markers
importlib.reload(pipelines.missing_markers) # Reload to clear cache.
import pipelines.auto_label
importlib.reload(pipelines.auto_label) # Reload to clear cache.

MENU_NAME = "MoreLab"

def _setup_commands():
    cmds = [
        ("Apply Butterworth Filter to Marker Set", pipelines.marker_filtering.apply_butterworth_filter_to_marker_set),
        ("Apply Butterworth Filter to Selected Trajectories", pipelines.marker_filtering.apply_butterworth_filter_to_selected_trajectories),
        ("Batch Process Marker Filtering", pipelines.batch_marker_filtering.batch_process_marker_filtering),
        ("Fill Gaps in Pelvis (Relational)", pipelines.pelvis_gap_fill_relational.pelvis_gap_fill_relational),
        ("Fill Gaps in Head (Relational)", pipelines.head_gap_fill_relational.head_gap_fill_relational),
        ("Fill Gaps in Torso (Relational)", pipelines.torso_gap_fill_relational.torso_gap_fill_relational),
        ("Fill Gaps in Arm (Relational)", pipelines.arm_gap_fill_relational.arm_gap_fill_relational),
        ("Fill Gaps in Thigh (Relational)", pipelines.thigh_gap_fill_relational.thigh_gap_fill_relational),
        ("Fill Gaps in Shank (Relational)", pipelines.shank_gap_fill_relational.shank_gap_fill_relational),
        ("Fill Gaps in Foot (Relational)", pipelines.foot_gap_fill_relational.foot_gap_fill_relational),
        ("Fill Gaps in Full Body (Relational)", pipelines.full_body_gap_fill_relational.full_body_gap_fill_relational),
        ("Create Missing Marker from Static Trial", pipelines.missing_markers.create_missing_marker_from_static_trial),
        ("Generate reference distribution", pipelines.auto_label.gui_generate_reference_distribution),
        ("Auto label everything", pipelines.auto_label.gui_auto_label_everything),
        ("Auto label labelled", pipelines.auto_label.gui_auto_label_labelled),
        ("Auto label unlabelled", pipelines.auto_label.gui_auto_label_unlabelled),
        ("Auto label selected trajectories", pipelines.auto_label.gui_auto_label_selected_trajectories),
        ("Generate SAL reference distribution", pipelines.auto_label.gui_generate_sal_ref),
        ("SAL", pipelines.auto_label.gui_sal),
    ]
    for label, fn in cmds:
        qtm.gui.add_command(label)
        qtm.gui.set_command_execute_function(label, fn)

def _setup_menu():
    mid = qtm.gui.insert_menu_submenu(None, MENU_NAME, None)
    fmid = qtm.gui.insert_menu_submenu(mid, "Filters", None)
    mmid = qtm.gui.insert_menu_submenu(mid, "Missing Markers", None)
    gmid = qtm.gui.insert_menu_submenu(mid, "Gap Filling (Relational)", None)
    lmid = qtm.gui.insert_menu_submenu(mid, "Auto label", None)

    add_menu_item(fmid, "Apply Butterworth Filter to Marker Set", "Apply Butterworth Filter to Marker Set")
    add_menu_item(fmid, "Apply Butterworth Filter to Selected Trajectories", "Apply Butterworth Filter to Selected Trajectories")
    add_menu_item(fmid, "Batch Process Marker Filtering", "Batch Process Marker Filtering")
    add_menu_item(mmid, "Create Missing Marker from Static Trial", "Create Missing Marker from Static Trial")
    add_menu_item(gmid, "Head", "Fill Gaps in Head (Relational)")
    add_menu_item(gmid, "Pelvis", "Fill Gaps in Pelvis (Relational)")
    add_menu_item(gmid, "Torso", "Fill Gaps in Torso (Relational)")
    add_menu_item(gmid, "Arm & Hand", "Fill Gaps in Arm (Relational)")
    add_menu_item(gmid, "Thigh", "Fill Gaps in Thigh (Relational)")
    add_menu_item(gmid, "Shank", "Fill Gaps in Shank (Relational)")
    add_menu_item(gmid, "Foot", "Fill Gaps in Foot (Relational)")
    add_menu_item(gmid, "Full Body", "Fill Gaps in Full Body (Relational)")
    add_menu_item(lmid, "Generate reference distribution", "Generate reference distribution")
    add_menu_item(lmid, "Auto label everything", "Auto label everything")
    add_menu_item(lmid, "Auto label labelled", "Auto label labelled")
    add_menu_item(lmid, "Auto label unlabelled", "Auto label unlabelled")
    add_menu_item(lmid, "Auto label selected trajectories (only if no overlap)", "Auto label selected trajectories")
    qtm.gui.insert_menu_separator(lmid,255)
    add_menu_item(lmid, "Generate SAL reference distribution", "Generate SAL reference distribution")
    add_menu_item(lmid, "Check selected trajectories using skeleton", "SAL")

def add_menu():
    try:
        print("Reloading modules...")
        _setup_commands()
        _setup_menu()
        print("Done!")
    except Exception as e:
        try_print_except(str(e), "Press 'Reload scripts' to try again.")

if __name__ == "__main__":
    add_menu()

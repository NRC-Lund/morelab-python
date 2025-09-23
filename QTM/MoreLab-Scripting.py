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
import pipelines.fix_sips
importlib.reload(pipelines.fix_sips) # Reload to clear cache.
import pipelines.calibrations
importlib.reload(pipelines.calibrations) # Reload to clear cache.
import pipelines.other
importlib.reload(pipelines.other) # Reload to clear cache.
import pipelines.custom_filters
importlib.reload(pipelines.custom_filters) # Reload to clear cache.
import pipelines.remove_spikes
importlib.reload(pipelines.remove_spikes) # Reload to clear cache.
import pipelines.pelvis_gap_fill_relational
importlib.reload(pipelines.pelvis_gap_fill_relational) # Reload to clear cache.
import pipelines.thigh_gap_fill_relational
importlib.reload(pipelines.thigh_gap_fill_relational) # Reload to clear cache.
import pipelines.shank_gap_fill_relational
importlib.reload(pipelines.shank_gap_fill_relational) # Reload to clear cache.
import pipelines.foot_gap_fill_relational
importlib.reload(pipelines.foot_gap_fill_relational) # Reload to clear cache.
from pipelines.auto_label import \
    gui_generate_reference_distribution, \
    gui_auto_label_everything, \
    gui_auto_label_labelled, \
    gui_auto_label_unlabelled, \
    gui_auto_label_selected_trajectories, \
    gui_remove_spikes
importlib.reload(pipelines.auto_label) # Reload to clear cache.



MENU_NAME = "MoreLab"

def _setup_commands():
    cmds = [
        ("Custom Filter", pipelines.custom_filters.apply_butterworth_filter),
        ("Static calibration", pipelines.calibrations.static_calibration),
        ("Dynamic calibration", pipelines.calibrations.dynamic_calibration),
        ("Fix SIPS", pipelines.fix_sips.fix_sips),
        ("Remove Spikes", pipelines.remove_spikes.remove_spikes),
        ("Fill Gaps in Pelvis (Relational)", pipelines.pelvis_gap_fill_relational.pelvis_gap_fill_relational),
        ("Fill Gaps in Thigh (Relational)", pipelines.thigh_gap_fill_relational.thigh_gap_fill_relational),
        ("Fill Gaps in Shank (Relational)", pipelines.shank_gap_fill_relational.shank_gap_fill_relational),
        ("Fill Gaps in Foot (Relational)", pipelines.foot_gap_fill_relational.foot_gap_fill_relational),
        ("Generate reference distribution", gui_generate_reference_distribution),
        ("Auto label everything", gui_auto_label_everything),
        ("Auto label labelled", gui_auto_label_labelled),
        ("Auto label unlabelled", gui_auto_label_unlabelled),
        ("Auto label selected trajectories", gui_auto_label_selected_trajectories),
        ("Remove spikes and fill", gui_remove_spikes),
    ]
    for label, fn in cmds:
        qtm.gui.add_command(label)
        qtm.gui.set_command_execute_function(label, fn)

def _setup_menu():
    mid = qtm.gui.insert_menu_submenu(None, MENU_NAME, None)
    fmid = qtm.gui.insert_menu_submenu(mid, "Filters", None)
    amid = qtm.gui.insert_menu_submenu(mid, "Add trajectory", None)
    gmid = qtm.gui.insert_menu_submenu(mid, "Gap Filling (Relational)", None)
    lmid = qtm.gui.insert_menu_submenu(mid, "Auto label", None)

    add_menu_item(fmid, "Apply Butterworth Filter", "Custom Filter")
    # add_menu_item(fmid, "Apply ForcePlate Filter", "Custom Force Plate Filter")
    add_menu_item(amid, "Static Calibration", "Static calibration")
    add_menu_item(amid, "Dynamic Calibration", "Dynamic calibration")
    add_menu_item(mid, "Fix SIPS markers", "Fix SIPS")
    add_menu_item(mid, "Remove Spikes", "Remove Spikes")
    add_menu_item(gmid, "Pelvis", "Fill Gaps in Pelvis (Relational)")
    add_menu_item(gmid, "Thigh", "Fill Gaps in Thigh (Relational)")
    add_menu_item(gmid, "Shank", "Fill Gaps in Shank (Relational)")
    add_menu_item(gmid, "Foot", "Fill Gaps in Foot (Relational)")
    add_menu_item(lmid, "Generate reference distribution", "Generate reference distribution")
    add_menu_item(lmid, "Auto label everything", "Auto label everything")
    add_menu_item(lmid, "Auto label labeled", "Auto label labeled")
    add_menu_item(lmid, "Auto label unlabeled", "Auto label unlabeled")
    add_menu_item(lmid, "Auto label selected trajectories (only if no overlap)", "Auto label selected trajectories")
    #add_menu_item(lmid, "Remove spikes and fill", "Remove spikes and fill")

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

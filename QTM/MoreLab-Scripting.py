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
import pipelines.filters
importlib.reload(pipelines.filters) # Reload to clear cache.
import pipelines.calibrations
importlib.reload(pipelines.calibrations) # Reload to clear cache.
import pipelines.fix_sips
importlib.reload(pipelines.fix_sips) # Reload to clear cache.

MENU_NAME = "MoreLab"

def _setup_commands():
    cmds = [
        ("Custom Filter", pipelines.filters.apply_butterworth_filter),
        ("Static calibration", pipelines.calibrations.static_calibration),
        ("Dynamic calibration", pipelines.calibrations.dynamic_calibration),
        ("Fix SIPS", pipelines.fix_sips.fix_sips)
    ]
    for label, fn in cmds:
        qtm.gui.add_command(label)
        qtm.gui.set_command_execute_function(label, fn)

def _setup_menu():
    mid = qtm.gui.insert_menu_submenu(None, MENU_NAME, None)
    fmid = qtm.gui.insert_menu_submenu(mid, "Filters", None)
    amid = qtm.gui.insert_menu_submenu(mid, "Add trajectory", None)

    add_menu_item(fmid, "Apply Butterworth Filter", "Custom Filter")
    # add_menu_item(fmid, "Apply ForcePlate Filter", "Custom Force Plate Filter")
    add_menu_item(amid, "Static Calibration", "Static calibration")
    add_menu_item(amid, "Dynamic Calibration", "Dynamic calibration")
    add_menu_item(mid, "Fix SIPS markers", "Fix SIPS")

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

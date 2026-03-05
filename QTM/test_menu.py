import sys, os, inspect, importlib

# Ensure project root on path
this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if this_dir not in sys.path:
    sys.path.append(this_dir)

# Impoprt Qualisys modules
import qtm
from helpers.printing import try_print_except
from helpers.menu_tools import add_menu_item

import pipelines.unidentify_parts
importlib.reload(pipelines.unidentify_parts) # Reload to clear cache.
import pipelines.discard_current_frame
importlib.reload(pipelines.discard_current_frame) # Reload to clear cache.
import pipelines.select_unidentified_markers
importlib.reload(pipelines.select_unidentified_markers) # Reload to clear cache.
import pipelines.apply_aim_model
importlib.reload(pipelines.apply_aim_model) # Reload to clear cache.
import pipelines.breakdown_trajectories
importlib.reload(pipelines.breakdown_trajectories) # Reload to clear cache.
import pipelines.skeleton_solver_workflow
importlib.reload(pipelines.skeleton_solver_workflow) # Reload to clear cache.

MENU_NAME = "Test"

def _setup_commands():
    cmds = [
        ("Unidentify Parts", pipelines.unidentify_parts.unidentify_parts),
        ("Discard Current Frame", pipelines.discard_current_frame.discard_current_frame),
        ("Select Unidentified Markers", pipelines.select_unidentified_markers.select_unidentified_markers),
        ("Apply Aim Model to Unidentified Trajectories", pipelines.apply_aim_model.apply_aim_model),
        ("Breakdown Selected Trajectories into Parts", pipelines.breakdown_trajectories.breakdown_trajectories),
        ("Batch Process Skeleton Solving", pipelines.skeleton_solver_workflow.main)
    ]
    for label, fn in cmds:
        qtm.gui.add_command(label)
        qtm.gui.set_command_execute_function(label, fn)

def _setup_menu():
    mid = qtm.gui.insert_menu_submenu(None, MENU_NAME, None)
    add_menu_item(mid, "Unidentify Parts", "Unidentify Parts")
    add_menu_item(mid, "Discard Current Frame", "Discard Current Frame")
    add_menu_item(mid, "Select Unidentified Markers", "Select Unidentified Markers")
    add_menu_item(mid, "Apply Aim Model to Unidentified Trajectories", "Apply Aim Model to Unidentified Trajectories")
    add_menu_item(mid, "Breakdown Selected Trajectories into Parts", "Breakdown Selected Trajectories into Parts")
    add_menu_item(mid, "Batch Process Skeleton Solving", "Batch Process Skeleton Solving")

def _add_shortcut():
    ctrl_u = {"ctrl": True, "alt": False, "shift": False, "key": "u"}
    qtm.gui.set_accelerator(ctrl_u, "Unidentify Parts")
    ctrl_d = {"ctrl": True, "alt": False, "shift": False, "key": "d"}
    qtm.gui.set_accelerator(ctrl_d, "Discard Current Frame")
    ctrl_i = {"ctrl": True, "alt": False, "shift": False, "key": "i"}
    qtm.gui.set_accelerator(ctrl_i, "Select Unidentified Markers")
    ctrl_b = {"ctrl": True, "alt": False, "shift": False, "key": "b"}
    qtm.gui.set_accelerator(ctrl_b, "Breakdown Selected Trajectories into Parts")

def add_menu():
    try:
        print("Reloading modules...")
        _setup_commands()
        _setup_menu()
        _add_shortcut()
        print("Done!")
    except Exception as e:
        try_print_except(str(e), "Press 'Reload scripts' to try again.")

if __name__ == "__main__":
    add_menu()
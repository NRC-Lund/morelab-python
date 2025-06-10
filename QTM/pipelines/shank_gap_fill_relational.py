# -------------------- Imports -------------------- #
import sys
import os
import inspect
import numpy as np

# Set up QTM Python API
this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if this_dir not in sys.path:
    sys.path.append(this_dir)

import qtm

# -------------------- Marker Definitions -------------------- #
# Define pelvic markers and rename them based on prefix in QTM
marker_prefix = "Q_" # This must be changed depending on the prefix used in QTM
base_marker_names = ["RShinFrontHigh", "LShinFrontHigh", "RShinFrontLow", "LShinFrontLow", "RShinSide", "LShinSide", "RKneeOut", "LKneeOut", "RKneeIn", "LKneeIn"]
marker_names = [f"{marker_prefix}{name}" for name in base_marker_names] # add prefix to list of markers

# Define hierarchy for reference markers in relational gap filling method
base_gap_fill_rules = {
    "RShinFrontHigh": [
        # 3 marker options
        ["RShinFrontLow", "RShinSide", "RKneeOut"],
        ["RShinFrontLow", "RShinSide", "RKneeIn"],   
                   
        # 2 marker options
        ["RShinFrontLow", "RShinSide"],                 
        ["RShinFrontLow", "RKneeOut"],          
        ["RShinSide", "RKneeOut"], 
        ["RShinFrontLow", "RKneeIn"],
        ["RShinSide", "RKneeIn"],                   

        # 1 marker options
        ["RShinFrontLow"],                          
        ["RShinSide"],
        ["RKneeOut"],
        ["RKneeIn"],
    ],

    "LShinFrontHigh": [
        # 3 marker options
        ["LShinFrontLow", "LShinSide", "LKneeOut"],
        ["LShinFrontLow", "LShinSide", "LKneeIn"],   
                   
        # 2 marker options
        ["LShinFrontLow", "LShinSide"],                 
        ["LShinFrontLow", "LKneeOut"],          
        ["LShinSide", "LKneeOut"], 
        ["LShinFrontLow", "LKneeIn"],
        ["LShinSide", "LKneeIn"],                   

        # 1 marker options
        ["LShinFrontLow"],                          
        ["LShinSide"],
        ["LKneeOut"],
        ["LKneeIn"],
    ],

    "RShinFrontLow": [
        # 3 marker options
        ["RShinFrontHigh", "RShinSide", "RKneeOut"],
        ["RShinFrontHigh", "RShinSide", "RKneeIn"],     
                   
        # 2 marker options
        ["RShinFrontHigh", "RShinSide"],                 
        ["RShinFrontHigh", "RKneeOut"],          
        ["RShinSide", "RKneeOut"], 
        ["RShinFrontHigh", "RKneeIn"],
        ["RShinSide", "RKneeIn"],                   

        # 1 marker options
        ["RShinFrontHigh"],                          
        ["RShinSide"],
        ["RKneeOut"],
        ["RKneeIn"],
    ],

    "LShinFrontLow": [
        # 3 marker options
        ["LShinFrontHigh", "LShinSide", "LKneeOut"],
        ["LShinFrontHigh", "LShinSide", "LKneeIn"],     
                   
        # 2 marker options
        ["LShinFrontHigh", "LShinSide"],                 
        ["LShinFrontHigh", "LKneeOut"],          
        ["LShinSide", "LKneeOut"], 
        ["LShinFrontHigh", "LKneeIn"],
        ["LShinSide", "LKneeIn"],                   

        # 1 marker options
        ["LShinFrontHigh"],                          
        ["LShinSide"],
        ["LKneeOut"],
        ["LKneeIn"],
    ],

    "RShinSide": [
        # 3 marker options
        ["RShinFrontHigh", "RShinFrontLow", "RKneeOut"],
        ["RShinFrontHigh", "RShinFrontLow", "RKneeIn"],     
                   
        # 2 marker options
        ["RShinFrontHigh", "RShinFrontLow"],                 
        ["RShinFrontHigh", "RKneeOut"],          
        ["RShinFrontLow", "RKneeOut"], 
        ["RShinFrontHigh", "RKneeIn"],
        ["RShinFrontLow", "RKneeIn"],                   

        # 1 marker options
        ["RShinFrontHigh"],                          
        ["RShinFrontLow"],
        ["RKneeOut"],
        ["RKneeIn"],
    ],

    "LShinSide": [
        # 3 marker options
        ["LShinFrontHigh", "LShinFrontLow", "LKneeOut"],
        ["LShinFrontHigh", "LShinFrontLow", "LKneeIn"],     
                   
        # 2 marker options
        ["LShinFrontHigh", "LShinFrontLow"],                 
        ["LShinFrontHigh", "LKneeOut"],          
        ["LShinFrontLow", "LKneeOut"], 
        ["LShinFrontHigh", "LKneeIn"],
        ["LShinFrontLow", "LKneeIn"],                   

        # 1 marker options
        ["LShinFrontHigh"],                          
        ["LShinFrontLow"],
        ["LKneeOut"],
        ["LKneeIn"],
    ],

}

# Add prefix to set of rules too
gap_fill_rules = {
    f"{marker_prefix}{k}": [[f"{marker_prefix}{name}" for name in rule] for rule in rules]
    for k, rules in base_gap_fill_rules.items()
}

# -------------------- Gap Filling Functions -------------------- #
def get_all_gap_ranges(id, total_frames):
    # Returns a complete list of gaps including any at the start or end of the recording.
    gap_ranges = qtm.data.series._3d.get_gap_ranges(id)
    sample_ranges = qtm.data.series._3d.get_sample_ranges(id)
    all_gaps = []
    # Check for gap at start
    if sample_ranges:
        if sample_ranges[0]["start"] > 0:
            all_gaps.append({"start": 0, "end": sample_ranges[0]["start"]})
    else:
        # No data at all — entire range is a gap
        return [{"start": 0, "end": total_frames}]
    # Add internal gaps
    all_gaps.extend(gap_ranges)
    # Check for gap at end
    if sample_ranges[-1]["end"] < total_frames["end"]:
        all_gaps.append({"start": sample_ranges[-1]["end"], "end": total_frames["end"]})

    return all_gaps

# Input missing marker plus start and end of gap to be filled. Function will assess and output the available reference markers to fill the gap.
def ref_markers_available(missing_marker, marker_names, gap_start, gap_end):
    ref_marker_names = [name for name in marker_names if name != missing_marker] # excludes missing marker from options as refence marker.
    valid_markers = []
    for marker in ref_marker_names:
        traj_id = qtm.data.object.trajectory.find_trajectory(marker)
        valid_ranges = qtm.data.series._3d.get_sample_ranges(traj_id)
        for frame in valid_ranges:
            if frame['start'] <= gap_start and frame['end'] >= gap_end:
                valid_markers.append(marker)
                break
    return valid_markers

# Select best combination of markers to fill gap, given the available markers identified.
def select_gap_fill_rule(missing_marker, valid_markers, gap_fill_rules):
    rules = gap_fill_rules.get(missing_marker, [])
    for rule in rules:
        if all(ref in valid_markers for ref in rule):
            ref_markers = {"origin": None, "line": None, "plane": None}
            if len(rule) >= 1:
                ref_markers["origin"] = qtm.data.object.trajectory.find_trajectory(rule[0])
            if len(rule) >= 2:
                ref_markers["line"] = qtm.data.object.trajectory.find_trajectory(rule[1])
            if len(rule) >= 3:
                ref_markers["plane"] = qtm.data.object.trajectory.find_trajectory(rule[2])

            ref_markers = {k: v for k, v in ref_markers.items() if v is not None} # removes empty keys where only 1 or 2 marker options are identified
            return ref_markers  # First matching set of markers
    return None  # No valid rule found

def fill_gap(id, gap, ref_markers):
    qtm.data.object.trajectory.fill_trajectory(id, "relational", gap, ref_markers)

def shank_gap_fill_relational():
    # Define max gap size to fill
    title = "Max gap fill range"
    message = "What is the max gap length (in frames) you'd like to fill?"
    input = "25"
    input_string = qtm.gui.dialog.show_string_input_dialog(title, message, input)
    if input_string is None: # default to 25 if user cancels dialogue
        max_range = 25
    else:
        max_range = int(input_string)

    total_frames = qtm.gui.timeline.get_measured_range()
    for marker in marker_names:
        # find marker id based on label
        id = qtm.data.object.trajectory.find_trajectory(marker)
        # find gaps in marker
        gaps  = get_all_gap_ranges(id, total_frames)
        if not gaps:
            print(f"No gaps detected in {marker}")
            continue # skips marker if it doesn't contain any gaps.
        gaps_filled = len(gaps)
        gaps_unfilled = 0
        for gap in gaps:
            gap_start = gap["start"]
            gap_end = gap["end"]
            if gap["end"] - gap["start"] > max_range:
                gaps_filled -= 1
                gaps_unfilled += 1
                continue # skips gap if longer than 25 frames
            # check which reference markers are available
            valid_markers = ref_markers_available(marker, marker_names, gap_start, gap_end)
            ref_markers = select_gap_fill_rule(marker, valid_markers, gap_fill_rules)
            # use appropriate reference markers to fill the gap
            fill_gap(id, gap, ref_markers)
        print(f"{gaps_filled} gaps in {marker} were filled. {gaps_unfilled} gaps were not filled due to being longer than {max_range} frames")
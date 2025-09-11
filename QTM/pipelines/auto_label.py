import qtm
import numpy as np
import os

def gui_generate_reference_distribution():
    #print(qtm.get_version())
    fname_qtm = qtm.file.get_path()
    fname_npz = os.path.splitext(fname_qtm)[0] + ".npz"
    fname_npz = qtm.gui.dialog.show_save_file_dialog("Save reference distribution", 
                                                     ["NumPy files (*.npz)"], 
                                                     os.path.basename(fname_npz), 
                                                     os.path.dirname(fname_npz))
    if not fname_npz:
        print("No file selected, aborting")
        return    
    generate_reference_distribution(fname_npz)

def generate_reference_distribution(fname_npz):
    print(f"Generating reference distribution and saving to {fname_npz}...")
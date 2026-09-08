from .gap_fill_relational import gap_fill_relational


# -------------------- Marker Definitions -------------------- #
# Define pelvic markers and rename them based on prefix in QTM
base_marker_names = ["RShinFrontHigh", "LShinFrontHigh", "RShinFrontLow", "LShinFrontLow", "RShinSide", "LShinSide", "RKneeOut", "LKneeOut", "RKneeIn", "LKneeIn"]

# Define hierarchy for reference markers in relational gap filling method
base_gap_fill_rules = {
    "RShinFrontHigh": [
        # 3 marker options
        ["RShinFrontLow", "RShinSide", "RKneeOut"],
        ["RShinFrontLow", "RShinSide", "RKneeIn"],   
                   
        # 2 marker options
        ["RShinFrontLow", "RShinSide"],                 
        # ["RShinFrontLow", "RKneeOut"],          
        # ["RShinSide", "RKneeOut"], 
        # ["RShinFrontLow", "RKneeIn"],
        # ["RShinSide", "RKneeIn"],                   

        # 1 marker options
        ["RShinFrontLow"],                          
        ["RShinSide"],
        # ["RKneeOut"],
        # ["RKneeIn"],
    ],

    "LShinFrontHigh": [
        # 3 marker options
        ["LShinFrontLow", "LShinSide", "LKneeOut"],
        ["LShinFrontLow", "LShinSide", "LKneeIn"],   
                   
        # 2 marker options
        ["LShinFrontLow", "LShinSide"],                 
        # ["LShinFrontLow", "LKneeOut"],          
        # ["LShinSide", "LKneeOut"], 
        # ["LShinFrontLow", "LKneeIn"],
        # ["LShinSide", "LKneeIn"],                   

        # 1 marker options
        ["LShinFrontLow"],                          
        ["LShinSide"],
        # ["LKneeOut"],
        # ["LKneeIn"],
    ],

    "RShinFrontLow": [
        # 3 marker options
        ["RShinFrontHigh", "RShinSide", "RKneeOut"],
        ["RShinFrontHigh", "RShinSide", "RKneeIn"],     
                   
        # 2 marker options
        ["RShinFrontHigh", "RShinSide"],                 
        # ["RShinFrontHigh", "RKneeOut"],          
        # ["RShinSide", "RKneeOut"], 
        # ["RShinFrontHigh", "RKneeIn"],
        # ["RShinSide", "RKneeIn"],                   

        # 1 marker options
        ["RShinFrontHigh"],                          
        ["RShinSide"],
        # ["RKneeOut"],
        # ["RKneeIn"],
    ],

    "LShinFrontLow": [
        # 3 marker options
        ["LShinFrontHigh", "LShinSide", "LKneeOut"],
        ["LShinFrontHigh", "LShinSide", "LKneeIn"],     
                   
        # 2 marker options
        ["LShinFrontHigh", "LShinSide"],                 
        # ["LShinFrontHigh", "LKneeOut"],          
        # ["LShinSide", "LKneeOut"], 
        # ["LShinFrontHigh", "LKneeIn"],
        # ["LShinSide", "LKneeIn"],                   

        # 1 marker options
        ["LShinFrontHigh"],                          
        ["LShinSide"],
        # ["LKneeOut"],
        # ["LKneeIn"],
    ],

    "RShinSide": [
        # 3 marker options
        ["RShinFrontHigh", "RShinFrontLow", "RKneeOut"],
        ["RShinFrontHigh", "RShinFrontLow", "RKneeIn"],     
                   
        # 2 marker options
        ["RShinFrontHigh", "RShinFrontLow"],                 
        # ["RShinFrontHigh", "RKneeOut"],          
        # ["RShinFrontLow", "RKneeOut"], 
        # ["RShinFrontHigh", "RKneeIn"],
        # ["RShinFrontLow", "RKneeIn"],                   

        # 1 marker options
        ["RShinFrontHigh"],                          
        ["RShinFrontLow"],
        # ["RKneeOut"],
        # ["RKneeIn"],
    ],

    "LShinSide": [
        # 3 marker options
        ["LShinFrontHigh", "LShinFrontLow", "LKneeOut"],
        ["LShinFrontHigh", "LShinFrontLow", "LKneeIn"],     
                   
        # 2 marker options
        ["LShinFrontHigh", "LShinFrontLow"],                 
        # ["LShinFrontHigh", "LKneeOut"],          
        # ["LShinFrontLow", "LKneeOut"], 
        # ["LShinFrontHigh", "LKneeIn"],
        # ["LShinFrontLow", "LKneeIn"],                   

        # 1 marker options
        ["LShinFrontHigh"],                          
        ["LShinFrontLow"],
        # ["LKneeOut"],
        # ["LKneeIn"],
    ],

    "RKneeOut": [
        # 1 marker options
        ["RKneeIn"],
    ],

    "LKneeOut": [
        # 1 marker options
        ["LKneeIn"],
    ],

    "RKneeIn": [
        # 1 marker options
        ["RKneeOut"],
    ],

    "LKneeIn": [
        # 1 marker options
        ["LKneeOut"],
    ],

}

def shank_gap_fill_relational():
    gap_fill_relational(base_marker_names, base_gap_fill_rules)

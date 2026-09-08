from .gap_fill_relational import gap_fill_relational


# -------------------- Marker Definitions -------------------- #
# Define thigh markers and rename them based on prefix in QTM
base_marker_names = ["RThighHigh", "LThighHigh", "RThighLow", "LThighLow", "RThighMedial", "LThighMedial", "RKneeOut", "LKneeOut", "RKneeIn", "LKneeIn"]

# Define hierarchy for reference markers in relational gap filling method
base_gap_fill_rules = {
    "RThighHigh": [
        # 3 marker options
        ["RThighLow", "RThighMedial", "RKneeOut"],
        ["RThighLow", "RThighMedial", "RKneeIn"],     
                   
        # 2 marker options
        ["RThighLow", "RThighMedial"],                 
        # ["RThighLow", "RKneeOut"],          
        # ["RThighMedial", "RKneeOut"], 
        # ["RThighLow", "RKneeIn"],
        # ["RThighMedial", "RKneeIn"],                   

        # 1 marker options
        ["RThighLow"],                          
        ["RThighMedial"],
        # ["RKneeOut"],
        # ["RKneeIn"],
    ],

    "LThighHigh": [
        # 3 marker options
        ["LThighLow", "LThighMedial", "LKneeOut"],
        ["LThighLow", "LThighMedial", "LKneeIn"],     
                   
        # 2 marker options
        ["LThighLow", "LThighMedial"],                 
        # ["LThighLow", "LKneeOut"],          
        # ["LThighMedial", "LKneeOut"], 
        # ["LThighLow", "LKneeIn"],
        # ["LThighMedial", "LKneeIn"],                   

        # 1 marker options
        ["LThighLow"],                          
        ["LThighMedial"],
        # ["LKneeOut"],
        # ["LKneeIn"],
    ],

    "RThighLow": [
        # 3 marker options
        ["RThighHigh", "RThighMedial", "RKneeOut"],
        ["RThighHigh", "RThighMedial", "RKneeIn"],     
                   
        # 2 marker options
        ["RThighHigh", "RThighMedial"],                 
        # ["RThighHigh", "RKneeOut"],          
        # ["RThighMedial", "RKneeOut"], 
        # ["RThighHigh", "RKneeIn"],
        # ["RThighMedial", "RKneeIn"],                   

        # 1 marker options
        ["RThighHigh"],                          
        ["RThighMedial"],
        # ["RKneeOut"],
        # ["RKneeIn"],
    ],

    "LThighLow": [
        # 3 marker options
        ["LThighHigh", "LThighMedial", "LKneeOut"],
        ["LThighHigh", "LThighMedial", "LKneeIn"],     
                   
        # 2 marker options
        ["LThighHigh", "LThighMedial"],                 
        # ["LThighHigh", "LKneeOut"],          
        # ["LThighMedial", "LKneeOut"], 
        # ["LThighHigh", "LKneeIn"],
        # ["LThighMedial", "LKneeIn"],                   

        # 1 marker options
        ["LThighHigh"],                          
        ["LThighMedial"],
        # ["LKneeOut"],
        # ["LKneeIn"],
    ],

    "RThighMedial": [
        # 3 marker options
        ["RThighHigh", "RThighLow", "RKneeOut"],
        ["RThighHigh", "RThighLow", "RKneeIn"],     
                   
        # 2 marker options
        ["RThighHigh", "RThighLow"],                 
        # ["RThighHigh", "RKneeOut"],          
        # ["RThighLow", "RKneeOut"], 
        # ["RThighHigh", "RKneeIn"],
        # ["RThighLow", "RKneeIn"],                   

        # 1 marker options
        ["RThighHigh"],                          
        ["RThighLow"],
        # ["RKneeOut"],
        # ["RKneeIn"],
    ],

    "LThighMedial": [
        # 3 marker options
        ["LThighHigh", "LThighLow", "LKneeOut"],
        ["LThighHigh", "LThighLow", "LKneeIn"],     
                   
        # 2 marker options
        ["LThighHigh", "LThighLow"],                 
        # ["LThighHigh", "LKneeOut"],          
        # ["LThighLow", "LKneeOut"], 
        # ["LThighHigh", "LKneeIn"],
        # ["LThighLow", "LKneeIn"],                   

        # 1 marker options
        ["LThighHigh"],                          
        ["LThighLow"],
        # ["LKneeOut"],
        # ["LKneeIn"],
    ],

}

def thigh_gap_fill_relational():
    gap_fill_relational(base_marker_names, base_gap_fill_rules)

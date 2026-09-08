from .gap_fill_relational import gap_fill_relational


# -------------------- Marker Definitions -------------------- #
# Define pelvic markers and rename them based on prefix in QTM
base_marker_names = ["WaistR", "WaistL", "WaistRFront", "WaistLFront", "RSips", "LSips"]

# Define hierarchy for reference markers in relational gap filling method
base_gap_fill_rules = {
    "WaistR": [
        # 3 marker options
        ["WaistRFront", "RSips", "LSips"],
        ["WaistRFront", "WaistLFront", "RSips"],
        ["WaistRFront", "WaistLFront", "LSips"],     
        ["WaistLFront", "RSips", "LSips"],
                   
        # 2 marker options
        ["WaistRFront", "RSips"],                 
        ["WaistRFront", "WaistLFront"],          
        ["RSips", "LSips"],                   
        ["WaistRFront", "LSips"],               
        ["WaistLFront", "RSips"],               
        ["WaistLFront", "LSips"],

        # 1 marker options
        ["WaistRFront"],                          
        ["RSips"],                              
        ["WaistLFront"],                         
        ["LSips"]
    ],

    "WaistL": [
        # 3 marker options
        ["WaistLFront", "LSips", "RSips"],
        ["WaistLFront", "WaistRFront", "LSips"],
        ["WaistLFront", "WaistRFront", "RSips"],    
        ["WaistRFront", "LSips", "RSips"],
                   
        # 2 marker options
        ["WaistLFront", "LSips"],              
        ["WaistLFront", "WaistRFront"],          
        ["LSips", "RSips"],                      
        ["WaistLFront", "RSips"],                
        ["WaistRFront", "LSips"],                 
        ["WaistRFront", "RSips"],

        # 1 marker options
        ["WaistLFront"],                          
        ["LSips"],                                
        ["WaistRFront"],                         
        ["RSips"]
    ],

    "WaistRFront": [
        # 3 marker options
        ["WaistLFront", "LSips", "RSips"],
        ["WaistLFront", "RSips", "WaistR"],
        ["WaistLFront", "LSips", "WaistR"],
        ["RSips", "LSips", "WaistR"],
        ["RSips", "LSips", "WaistL"],
        ["LSips", "WaistR", "WaistL"],
        ["RSips", "WaistR", "WaistL"],

        # 2 marker options
        ["WaistLFront", "RSips"],
        ["WaistLFront", "LSips"],
        ["RSips", "LSips"],
        ["RSips", "WaistR"],
        ["RSips", "WaistL"],
        ["LSips", "WaistR"],
        ["LSips", "WaistL"],
        ["WaistLFront", "WaistR"],
        ["WaistLFront", "WaistL"],

        # 1 marker options
        ["WaistLFront"],
        ["RSips"],
        ["LSips"],
        ["WaistR"]
    ],
    
    "WaistLFront": [
        # 3 marker options
        ["WaistRFront", "RSips", "LSips"],
        ["WaistRFront", "LSips", "WaistL"],
        ["WaistRFront", "RSips", "WaistL"],
        ["LSips", "RSips", "WaistL"],
        ["LSips", "RSips", "WaistR"],
        ["RSips", "WaistR", "WaistL"],
        ["LSips", "WaistR", "WaistL"],

        # 2 marker options
        ["WaistRFront", "LSips"],
        ["WaistRFront", "RSips"],
        ["LSips", "RSips"],
        ["LSips", "WaistL"],
        ["LSips", "WaistR"],
        ["RSips", "WaistL"],
        ["RSips", "WaistR"],
        ["WaistRFront", "WaistL"],
        ["WaistRFront", "WaistR"],

        # 1 marker options
        ["WaistRFront"],
        ["LSips"],
        ["RSips"],
        ["WaistL"]
    ],

    "RSips": [
        # 3-marker options
        ["LSips", "WaistLFront", "WaistRFront"],
        ["WaistRFront", "LSips", "WaistR"],
        ["WaistRFront", "LSips", "WaistL"],
        ["WaistRFront", "WaistLFront", "WaistR"],
        ["WaistRFront", "WaistLFront", "WaistL"],
        ["WaistRFront", "WaistR", "WaistL"],
        ["WaistLFront", "LSips", "WaistR"],

        # 2-marker options
        ["WaistRFront", "LSips"],
        ["WaistLFront", "LSips"],
        ["WaistRFront", "WaistLFront"],
        ["WaistRFront", "WaistR"],
        ["WaistRFront", "WaistL"],
        ["WaistLFront", "WaistR"],
        ["WaistLFront", "WaistL"],
        ["LSips", "WaistR"],
        ["LSips", "WaistL"],

        # 1-marker options
        ["WaistRFront"],
        ["WaistLFront"],
        ["LSips"],
        ["WaistR"],
        ["WaistL"]
    ],

    "LSips": [
        # 3-marker options
        ["RSips", "WaistRFront", "WaistLFront"],
        ["WaistLFront", "WaistRFront", "WaistL"],
        ["WaistLFront", "WaistRFront", "WaistR"],
        ["WaistLFront", "RSips", "WaistL"],
        ["WaistLFront", "RSips", "WaistR"],
        ["WaistLFront", "WaistL", "WaistR"],
        ["WaistRFront", "RSips", "WaistL"],

        # 2-marker options
        ["WaistLFront", "RSips"],
        ["WaistRFront", "RSips"],
        ["WaistLFront", "WaistRFront"],
        ["WaistLFront", "WaistL"],
        ["WaistLFront", "WaistR"],
        ["WaistRFront", "WaistL"],
        ["WaistRFront", "WaistR"],
        ["RSips", "WaistL"],
        ["RSips", "WaistR"],

        # 1-marker options
        ["WaistLFront"],
        ["WaistRFront"],
        ["RSips"],
        ["WaistL"],
        ["WaistR"]
    ]
}

def pelvis_gap_fill_relational():
    gap_fill_relational(base_marker_names, base_gap_fill_rules)

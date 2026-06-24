from .gap_fill_relational import add_marker_prefix, gap_fill_relational


# -------------------- Marker Definitions -------------------- #
# Define pelvic markers and rename them based on prefix in QTM
base_marker_names = ["RHeelBack", "LHeelBack", "RForefoot1", "LForefoot1", "RForefoot2", "LForefoot2", "RForefoot5", "LForefoot5", "RAnkleOut", "LAnkleOut", "RAnkleIn", "LAnkleIn"]

# Define hierarchy for reference markers in relational gap filling method
base_gap_fill_rules = {
    "RHeelBack": [
        # 3 marker options
        ["RForefoot1", "RForefoot5", "RAnkleOut"],
        ["RForefoot1", "RForefoot5", "RAnkleIn"],
        ["RAnkleOut", "RAnkleIn", "RForefoot1"],  
        ["RAnkleOut", "RAnkleIn", "RForefoot2"], 
        ["RAnkleOut", "RAnkleIn", "RForefoot5"],
        ["RAnkleOut", "RForefoot1", "RForefoot2"],  
        ["RAnkleOut", "RForefoot2", "RForefoot5"],
        ["RAnkleIn", "RForefoot1", "RForefoot2"],  
        ["RAnkleIn", "RForefoot2", "RForefoot5"],
        
        # 2 marker options
        ["RAnkleOut", "RAnkleIn"],
        ["RAnkleOut", "RForefoot1"],  
        ["RAnkleOut", "RForefoot2"], 
        ["RAnkleOut", "RForefoot5"],
        ["RAnkleIn", "RForefoot1"],  
        ["RAnkleIn", "RForefoot2"], 
        ["RAnkleIn", "RForefoot5"],

        # 1 marker options
        ["RAnkleOut"],                          
        ["RAnkleIn"],                              
        ["RForefoot1"],                         
        ["RForefoot2"],
        ["RForefoot5"]
    ],

    "LHeelBack": [
        # 3 marker options
        ["LForefoot1", "LForefoot5", "LAnkleOut"],
        ["LForefoot1", "LForefoot5", "LAnkleIn"],
        ["LAnkleOut", "LAnkleIn", "LForefoot1"],  
        ["LAnkleOut", "LAnkleIn", "LForefoot2"], 
        ["LAnkleOut", "LAnkleIn", "LForefoot5"],
        ["LAnkleOut", "LForefoot1", "LForefoot2"],  
        ["LAnkleOut", "LForefoot2", "LForefoot5"],
        ["LAnkleIn", "LForefoot1", "LForefoot2"],  
        ["LAnkleIn", "LForefoot2", "LForefoot5"],
        
        # 2 marker options
        ["LAnkleOut", "LAnkleIn"],
        ["LAnkleOut", "LForefoot1"],  
        ["LAnkleOut", "LForefoot2"], 
        ["LAnkleOut", "LForefoot5"],
        ["LAnkleIn", "LForefoot1"],  
        ["LAnkleIn", "LForefoot2"], 
        ["LAnkleIn", "LForefoot5"],

        # 1 marker options
        ["LAnkleOut"],                          
        ["LAnkleIn"],                              
        ["LForefoot1"],                         
        ["LForefoot2"],
        ["LForefoot5"]
    ],

    "RAnkleOut": [
        # 3 marker options
        ["RHeelBack", "RForefoot1", "RForefoot5"],
        ["RHeelBack", "RForefoot1", "RForefoot2"],
        ["RHeelBack", "RForefoot5", "RForefoot2"],
        ["RHeelBack", "RAnkleIn", "RForefoot1"],  
        ["RHeelBack", "RAnkleIn", "RForefoot2"], 
        ["RHeelBack", "RAnkleIn", "RForefoot5"],
        ["RAnkleIn", "RForefoot1", "RForefoot2"],  
        ["RAnkleIn", "RForefoot1", "RForefoot5"],
        ["RAnkleIn", "RForefoot2", "RForefoot5"],
        
        # 2 marker options
        ["RHeelBack", "RAnkleIn"],
        ["RAnkleIn", "RForefoot1"],  
        ["RAnkleIn", "RForefoot2"], 
        ["RAnkleIn", "RForefoot5"],
        ["RHeelBack", "RForefoot1"],  
        ["RHeelBack", "RForefoot2"], 
        ["RHeelBack", "RForefoot5"],

        # 1 marker options
        ["RAnkleIn"],
        ["RHeelBack"],                          
        ["RForefoot1"],                         
        ["RForefoot2"],
        ["RForefoot5"]
    ],

    "LAnkleOut": [
        # 3 marker options
        ["LHeelBack", "LForefoot1", "LForefoot5"],
        ["LHeelBack", "LForefoot1", "LForefoot2"],
        ["LHeelBack", "LForefoot5", "LForefoot2"],
        ["LHeelBack", "LAnkleIn", "LForefoot1"],  
        ["LHeelBack", "LAnkleIn", "LForefoot2"], 
        ["LHeelBack", "LAnkleIn", "LForefoot5"],
        ["LAnkleIn", "LForefoot1", "LForefoot2"],  
        ["LAnkleIn", "LForefoot1", "LForefoot5"],
        ["LAnkleIn", "LForefoot2", "LForefoot5"],
        
        # 2 marker options
        ["LHeelBack", "LAnkleIn"],
        ["LAnkleIn", "LForefoot1"],  
        ["LAnkleIn", "LForefoot2"], 
        ["LAnkleIn", "LForefoot5"],
        ["LHeelBack", "LForefoot1"],  
        ["LHeelBack", "LForefoot2"], 
        ["LHeelBack", "LForefoot5"],

        # 1 marker options
        ["LAnkleIn"],
        ["LHeelBack"],                          
        ["LForefoot1"],                         
        ["LForefoot2"],
        ["LForefoot5"]
    ],

    "RAnkleIn": [
        # 3 marker options
        ["RHeelBack", "RForefoot1", "RForefoot5"],
        ["RHeelBack", "RForefoot1", "RForefoot2"],
        ["RHeelBack", "RForefoot5", "RForefoot2"],
        ["RHeelBack", "RAnkleOut", "RForefoot1"],
        ["RHeelBack", "RAnkleOut", "RForefoot2"],
        ["RHeelBack", "RAnkleOut", "RForefoot5"],
        ["RAnkleOut", "RForefoot1", "RForefoot2"],
        ["RAnkleOut", "RForefoot1", "RForefoot5"],
        ["RAnkleOut", "RForefoot2", "RForefoot5"],

        # 2 marker options
        ["RHeelBack", "RAnkleOut"],
        ["RHeelBack", "RForefoot1"],
        ["RHeelBack", "RForefoot2"],
        ["RHeelBack", "RForefoot5"],
        ["RAnkleOut", "RForefoot1"],
        ["RAnkleOut", "RForefoot2"],
        ["RAnkleOut", "RForefoot5"],

        # 1 marker options
        ["RAnkleOut"],
        ["RHeelBack"],
        ["RForefoot1"],
        ["RForefoot2"],
        ["RForefoot5"]
    ],

    "LAnkleIn": [
        # 3 marker options
        ["LHeelBack", "LForefoot1", "LForefoot5"],
        ["LHeelBack", "LForefoot1", "LForefoot2"],
        ["LHeelBack", "LForefoot5", "LForefoot2"],
        ["LHeelBack", "LAnkleOut", "LForefoot1"],
        ["LHeelBack", "LAnkleOut", "LForefoot2"],
        ["LHeelBack", "LAnkleOut", "LForefoot5"],
        ["LAnkleOut", "LForefoot1", "LForefoot2"],
        ["LAnkleOut", "LForefoot1", "LForefoot5"],
        ["LAnkleOut", "LForefoot2", "LForefoot5"],

        # 2 marker options
        ["LHeelBack", "LAnkleOut"],
        ["LHeelBack", "LForefoot1"],
        ["LHeelBack", "LForefoot2"],
        ["LHeelBack", "LForefoot5"],
        ["LAnkleOut", "LForefoot1"],
        ["LAnkleOut", "LForefoot2"],
        ["LAnkleOut", "LForefoot5"],

        # 1 marker options
        ["LAnkleOut"],
        ["LHeelBack"],
        ["LForefoot1"],
        ["LForefoot2"],
        ["LForefoot5"]
    ],
    
    "RForefoot1": [
        # 3 marker options
        ["RHeelBack", "RForefoot5", "RForefoot2"],
        ["RForefoot2", "RForefoot5", "RAnkleOut"],  
        ["RForefoot2", "RForefoot5", "RAnkleIn"], 
        ["RForefoot2", "RHeelBack", "RAnkleOut"],
        ["RForefoot2", "RHeelBack", "RAnkleIn"], 
        ["RForefoot2", "RAnkleOut", "RAnkleIn"],
        ["RForefoot5", "RHeelBack", "RAnkleOut"],
        ["RForefoot5", "RHeelBack", "RAnkleIn"], 
        ["RForefoot5", "RAnkleOut", "RAnkleIn"],
        ["RHeelBack", "RAnkleOut", "RAnkleIn"],  
        
        # 2 marker options
        ["RForefoot2", "RForefoot5"],
        ["RForefoot2", "RHeelBack"],
        ["RForefoot2", "RAnkleOut"],
        ["RForefoot2", "RAnkleIn"],
        ["RForefoot5", "RHeelBack"],
        ["RForefoot5", "RAnkleOut"],
        ["RForefoot5", "RAnkleIn"],
        ["RHeelBack", "RAnkleOut"],
        ["RHeelBack", "RAnkleIn"],
        ["RAnkleOut", "RAnkleIn"],

        # 1 marker options
        ["RForefoot2"],
        ["RForefoot5"],                          
        ["RHeelBack"],                         
        ["RAnkleOut"],
        ["RAnkleIn"]
    ],

    "LForefoot1": [
        # 3 marker options
        ["LHeelBack", "LForefoot5", "LForefoot2"],
        ["LForefoot2", "LForefoot5", "LAnkleOut"],  
        ["LForefoot2", "LForefoot5", "LAnkleIn"], 
        ["LForefoot2", "LHeelBack", "LAnkleOut"],
        ["LForefoot2", "LHeelBack", "LAnkleIn"], 
        ["LForefoot2", "LAnkleOut", "LAnkleIn"],
        ["LForefoot5", "LHeelBack", "LAnkleOut"],
        ["LForefoot5", "LHeelBack", "LAnkleIn"], 
        ["LForefoot5", "LAnkleOut", "LAnkleIn"],
        ["LHeelBack", "LAnkleOut", "LAnkleIn"],  
        
        # 2 marker options
        ["LForefoot2", "LForefoot5"],
        ["LForefoot2", "LHeelBack"],
        ["LForefoot2", "LAnkleOut"],
        ["LForefoot2", "LAnkleIn"],
        ["LForefoot5", "LHeelBack"],
        ["LForefoot5", "LAnkleOut"],
        ["LForefoot5", "LAnkleIn"],
        ["LHeelBack", "LAnkleOut"],
        ["LHeelBack", "LAnkleIn"],
        ["LAnkleOut", "LAnkleIn"],

        # 1 marker options
        ["LForefoot2"],
        ["LForefoot5"],                          
        ["LHeelBack"],                         
        ["LAnkleOut"],
        ["LAnkleIn"]
    ],

    "RForefoot2": [
        # 3 marker options
        ["RHeelBack", "RForefoot1", "RForefoot5"],
        ["RForefoot1", "RForefoot5", "RAnkleOut"],  
        ["RForefoot1", "RForefoot5", "RAnkleIn"], 
        ["RForefoot1", "RHeelBack", "RAnkleOut"],
        ["RForefoot1", "RHeelBack", "RAnkleIn"], 
        ["RForefoot1", "RAnkleOut", "RAnkleIn"],
        ["RForefoot5", "RHeelBack", "RAnkleOut"],
        ["RForefoot5", "RHeelBack", "RAnkleIn"], 
        ["RForefoot5", "RAnkleOut", "RAnkleIn"],
        ["RHeelBack", "RAnkleOut", "RAnkleIn"],  
        
        # 2 marker options
        ["RForefoot1", "RForefoot5"],
        ["RForefoot1", "RHeelBack"],
        ["RForefoot1", "RAnkleOut"],
        ["RForefoot1", "RAnkleIn"],
        ["RForefoot5", "RHeelBack"],
        ["RForefoot5", "RAnkleOut"],
        ["RForefoot5", "RAnkleIn"],
        ["RHeelBack", "RAnkleOut"],
        ["RHeelBack", "RAnkleIn"],
        ["RAnkleOut", "RAnkleIn"],

        # 1 marker options
        ["RForefoot1"],
        ["RForefoot5"],                          
        ["RHeelBack"],                         
        ["RAnkleOut"],
        ["RAnkleIn"]
    ],

    "LForefoot2": [
        # 3 marker options
        ["LHeelBack", "LForefoot1", "LForefoot5"],
        ["LForefoot1", "LForefoot5", "LAnkleOut"],  
        ["LForefoot1", "LForefoot5", "LAnkleIn"], 
        ["LForefoot1", "LHeelBack", "LAnkleOut"],
        ["LForefoot1", "LHeelBack", "LAnkleIn"], 
        ["LForefoot1", "LAnkleOut", "LAnkleIn"],
        ["LForefoot5", "LHeelBack", "LAnkleOut"],
        ["LForefoot5", "LHeelBack", "LAnkleIn"], 
        ["LForefoot5", "LAnkleOut", "LAnkleIn"],
        ["LHeelBack", "LAnkleOut", "LAnkleIn"],  
        
        # 2 marker options
        ["LForefoot1", "LForefoot5"],
        ["LForefoot1", "LHeelBack"],
        ["LForefoot1", "LAnkleOut"],
        ["LForefoot1", "LAnkleIn"],
        ["LForefoot5", "LHeelBack"],
        ["LForefoot5", "LAnkleOut"],
        ["LForefoot5", "LAnkleIn"],
        ["LHeelBack", "LAnkleOut"],
        ["LHeelBack", "LAnkleIn"],
        ["LAnkleOut", "LAnkleIn"],

        # 1 marker options
        ["LForefoot1"],
        ["LForefoot5"],                          
        ["LHeelBack"],                         
        ["LAnkleOut"],
        ["LAnkleIn"]
    ],

    "RForefoot5": [
        # 3 marker options
        ["RHeelBack", "RForefoot1", "RForefoot2"],
        ["RForefoot1", "RForefoot2", "RAnkleOut"],  
        ["RForefoot1", "RForefoot2", "RAnkleIn"], 
        ["RForefoot1", "RHeelBack", "RAnkleOut"],
        ["RForefoot1", "RHeelBack", "RAnkleIn"], 
        ["RForefoot1", "RAnkleOut", "RAnkleIn"],
        ["RForefoot2", "RHeelBack", "RAnkleOut"],
        ["RForefoot2", "RHeelBack", "RAnkleIn"], 
        ["RForefoot2", "RAnkleOut", "RAnkleIn"],
        ["RHeelBack", "RAnkleOut", "RAnkleIn"],  
        
        # 2 marker options
        ["RForefoot1", "RForefoot2"],
        ["RForefoot1", "RHeelBack"],
        ["RForefoot1", "RAnkleOut"],
        ["RForefoot1", "RAnkleIn"],
        ["RForefoot2", "RHeelBack"],
        ["RForefoot2", "RAnkleOut"],
        ["RForefoot2", "RAnkleIn"],
        ["RHeelBack", "RAnkleOut"],
        ["RHeelBack", "RAnkleIn"],
        ["RAnkleOut", "RAnkleIn"],

        # 1 marker options
        ["RForefoot1"],
        ["RForefoot2"],                          
        ["RHeelBack"],                         
        ["RAnkleOut"],
        ["RAnkleIn"]
    ],

    "LForefoot5": [
        # 3 marker options
        ["LHeelBack", "LForefoot1", "LForefoot2"],
        ["LForefoot1", "LForefoot2", "LAnkleOut"],  
        ["LForefoot1", "LForefoot2", "LAnkleIn"], 
        ["LForefoot1", "LHeelBack", "LAnkleOut"],
        ["LForefoot1", "LHeelBack", "LAnkleIn"], 
        ["LForefoot1", "LAnkleOut", "LAnkleIn"],
        ["LForefoot2", "LHeelBack", "LAnkleOut"],
        ["LForefoot2", "LHeelBack", "LAnkleIn"], 
        ["LForefoot2", "LAnkleOut", "LAnkleIn"],
        ["LHeelBack", "LAnkleOut", "LAnkleIn"],  
        
        # 2 marker options
        ["LForefoot1", "LForefoot2"],
        ["LForefoot1", "LHeelBack"],
        ["LForefoot1", "LAnkleOut"],
        ["LForefoot1", "LAnkleIn"],
        ["LForefoot2", "LHeelBack"],
        ["LForefoot2", "LAnkleOut"],
        ["LForefoot2", "LAnkleIn"],
        ["LHeelBack", "LAnkleOut"],
        ["LHeelBack", "LAnkleIn"],
        ["LAnkleOut", "LAnkleIn"],

        # 1 marker options
        ["LForefoot1"],
        ["LForefoot2"],                          
        ["LHeelBack"],                         
        ["LAnkleOut"],
        ["LAnkleIn"]
    ],

}

marker_names, gap_fill_rules = add_marker_prefix(
    base_marker_names, base_gap_fill_rules, "Q_")


def foot_gap_fill_relational():
    gap_fill_relational(marker_names, gap_fill_rules)

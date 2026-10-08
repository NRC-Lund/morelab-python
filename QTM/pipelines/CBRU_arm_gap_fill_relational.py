from .gap_fill_relational import gap_fill_relational

base_marker_names = [
    "RUAPA",
    "RUAPP",
    "RUADP",
    "RUADA",
    "RELBL",
    "RELBM",
    "RFAPP",
    "RFAPA",
    "RFADP",
    "RFADA",
    "RRAD",
    "RULN",
    "RHAND",

    "LUAPA",
    "LUAPP",
    "LUADP",
    "LUADA",
    "LELBL",
    "LELBM",
    "LFAPP",
    "LFAPA",
    "LFADP",
    "LFADA",
    "LRAD",
    "LULN",
    "LHAND"
]

base_gap_fill_rules = {

    # Right Upper Arm - Proximal Anterior
    "RUAPA": [
        ["RUAPP", "RUADA"],
        ["RUAPP", "RUADP"],
        ["RUADA", "RUADP"],
    ],

    # Right Upper Arm - Proximal Posterior
    "RUAPP": [
        ["RUAPA", "RUADP"],
        ["RUAPA", "RUADA"],
        ["RUADP", "RELBL"],
    ],

    # Right Upper Arm - Distal Posterior
    "RUADP": [
        ["RUAPP", "RELBM"],
        ["RUAPP", "RELBL"],
        ["RUADA", "RELBM"],
    ],

    # Right Upper Arm - Distal Anterior
    "RUADA": [
        ["RUAPA", "RELBL"],
        ["RUAPA", "RELBM"],
        ["RUADP", "RELBL"],
    ],

    # Right Elbow - Lateral
    "RELBL": [
        ["RUADP", "RFAPP"],
        ["RUADA", "RFAPA"],
        ["RELBM", "RFAPP"],
    ],

    # Right Elbow - Medial
    "RELBM": [
        ["RUADP", "RFADP"],
        ["RUADA", "RFADA"],
        ["RELBL", "RFADP"],
    ],

    # Right Forearm - Proximal Posterior
    "RFAPP": [
        ["RELBL", "RFADP"],
        ["RELBL", "RFADA"],
        ["RELBM", "RFADP"],
    ],

    # Right Forearm - Proximal Anterior
    "RFAPA": [
        ["RELBL", "RFADA"],
        ["RELBM", "RFAPP"],
        ["RELBM", "RFADA"],
    ],

    # Right Forearm - Distal Posterior
    "RFADP": [
        ["RFAPP", "RRAD"],
        ["RFAPP", "RULN"],
        ["RFAPA", "RRAD"],
        ["RELBM", "RRAD"],
    ],

    # Right Forearm - Distal Anterior
    "RFADA": [
        ["RFAPA", "RRAD"],
        ["RFAPA", "RULN"],
        ["RFAPP", "RRAD"],
        ["RELBL", "RRAD"],
    ],

    # Right Radius
    "RRAD": [
        ["RFADP", "RFADA"],
        ["RFAPP", "RFADA"],
        ["RFADP", "RULN"],
        ["RFADA", "RULN"],
    ],

    # Right Ulna
    "RULN": [
        ["RFADP", "RFADA"],
        ["RFAPP", "RFADA"],
        ["RFADP", "RRAD"],
        ["RFADA", "RRAD"],
    ],

    # Right Hand
    "RHAND": [
        ["RFADP", "RFADA"],
        ["RFAPP", "RFAPA"],
        ["RRAD", "RULN"],
    ],


    # Left Upper Arm - Proximal Anterior
    "LUAPA": [
        ["LUAPP", "LUADA"],
        ["LUAPP", "LUADP"],
        ["LUADA", "LUADP"],
    ],

    # Left Upper Arm - Proximal Posterior
    "LUAPP": [
        ["LUAPA", "LUADP"],
        ["LUAPA", "LUADA"],
        ["LUADP", "LELBL"],
    ],

    # Left Upper Arm - Distal Posterior
    "LUADP": [
        ["LUAPP", "LELBM"],
        ["LUAPP", "LELBL"],
        ["LUADA", "LELBM"],
    ],

    # Left Upper Arm - Distal Anterior
    "LUADA": [
        ["LUAPA", "LELBL"],
        ["LUAPA", "LELBM"],
        ["LUADP", "LELBL"],
    ],

    # Left Elbow - Lateral
    "LELBL": [
        ["LUADP", "LFAPP"],
        ["LUADA", "LFAPA"],
        ["LELBM", "LFAPP"],
    ],

    # Left Elbow - Medial
    "LELBM": [
        ["LUADP", "LFADP"],
        ["LUADA", "LFADA"],
        ["LELBL", "LFADP"],
    ],

    # Left Forearm - Proximal Posterior
    "LFAPP": [
        ["LELBL", "LFADP"],
        ["LELBL", "LFADA"],
        ["LELBM", "LFADP"],
    ],

    # Left Forearm - Proximal Anterior
    "LFAPA": [
        ["LELBL", "LFADA"],
        ["LELBM", "LFAPP"],
        ["LELBM", "LFADA"],
    ],

    # Left Forearm - Distal Posterior
    "LFADP": [
        ["LFAPP", "LRAD"],
        ["LFAPP", "LULN"],
        ["LFAPA", "LRAD"],
        ["LELBM", "LRAD"],
    ],

    # Left Forearm - Distal Anterior
    "LFADA": [
        ["LFAPA", "LRAD"],
        ["LFAPA", "LULN"],
        ["LFAPP", "LRAD"],
        ["LELBL", "LRAD"],
    ],

    # Left Radius
    "LRAD": [
        ["LFADP", "LFADA"],
        ["LFAPP", "LFADA"],
        ["LFADP", "LULN"],
        ["LFADA", "LULN"],
    ],

    # Left Ulna
    "LULN": [
        ["LFADP", "LFADA"],
        ["LFAPP", "LFADA"],
        ["LFADP", "LRAD"],
        ["LFADA", "LRAD"],
    ],

    # Left Hand
    "LHAND": [
        ["LFADP", "LFADA"],
        ["LFAPP", "LFAPA"],
        ["LRAD", "LULN"],
    ],
}

def CBRU_arm_gap_fill_relational():
    gap_fill_relational(base_marker_names, base_gap_fill_rules)

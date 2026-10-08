from .gap_fill_relational import gap_fill_relational


# -------------------- Marker Definitions -------------------- #
# Define pelvic markers and rename them based on prefix in QTM
base_marker_names = [
    "RANL",
    "RANM",
    "RHEE",
    "RM1",
    "RM5",

    "LANL",
    "LANM",
    "LHEE",
    "LM1",
    "LM5"]

# Define hierarchy for reference markers in relational gap filling method
base_gap_fill_rules = {

    # Right Ankle Lateral
    "RANL": [
        ["RANM", "RHEE"],
        ["RANM", "RM5"],
        ["RHEE", "RM5"],
        ["RANM", "RM1"],
    ],

    # Right Ankle Medial
    "RANM": [
        ["RANL", "RHEE"],
        ["RANL", "RM1"],
        ["RHEE", "RM1"],
        ["RANL", "RM5"],
    ],

    # Right Heel
    "RHEE": [
        ["RANL", "RANM"],
        ["RANL", "RM5"],
        ["RANM", "RM1"],
        ["RANL", "RM1"],
        ["RANM", "RM5"],
    ],

    # Right 1st Metatarsal
    "RM1": [
        ["RANM", "RM5"],
        ["RANM", "RHEE"],
        ["RANL", "RM5"],
        ["RANL", "RANM"],
        ["RHEE", "RM5"],
    ],

    # Right 5th Metatarsal
    "RM5": [
        ["RANL", "RM1"],
        ["RANL", "RHEE"],
        ["RANL", "RANM"],
        ["RANM", "RM1"],
        ["RHEE", "RM1"],
    ],

    # Left Ankle Lateral
    "LANL": [
        ["LANM", "LHEE"],
        ["LANM", "LM5"],
        ["LHEE", "LM5"],
        ["LANM", "LM1"],
    ],

    # Left Ankle Medial
    "LANM": [
        ["LANL", "LHEE"],
        ["LANL", "LM1"],
        ["LHEE", "LM1"],
        ["LANL", "LM5"],
    ],

    # Left Heel
    "LHEE": [
        ["LANL", "LANM"],
        ["LANL", "LM5"],
        ["LANM", "LM1"],
        ["LANL", "LM1"],
        ["LANM", "LM5"],
    ],

    # Left 1st Metatarsal
    "LM1": [
        ["LANM", "LM5"],
        ["LANM", "LHEE"],
        ["LANL", "LM5"],
        ["LANL", "LANM"],
        ["LHEE", "LM5"],
    ],

    # Left 5th Metatarsal
    "LM5": [
        ["LANL", "LM1"],
        ["LANL", "LHEE"],
        ["LANL", "LANM"],
        ["LANM", "LM1"],
        ["LHEE", "LM1"],
    ],
}

def CBRU_foot_gap_fill_relational():
    gap_fill_relational(base_marker_names, base_gap_fill_rules)

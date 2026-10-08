from .gap_fill_relational import gap_fill_relational


# -------------------- Marker Definitions -------------------- #
# Define pelvic markers and rename them based on prefix in QTM
base_marker_names = [
    "RASI",
    "RIC",
    "PVUR",
    "PVLR",
    "LASI",
    "LIC",
    "PVUL",
    "PVLL"]

# Define hierarchy for reference markers in relational gap filling method
base_gap_fill_rules = {

    "RASI": [
        ["LASI", "RIC"],
        ["LASI", "LIC"],
        ["LASI", "PVUR"],
        ["LASI", "PVUL"],
        ["RIC", "PVUR"],
        ["RIC", "PVUL"],
        ["LIC", "PVUR"],
        ["LIC", "PVUL"],
    ],


    "RIC": [
        ["RASI", "LASI"],
        ["RASI", "LIC"],
        ["RASI", "PVUR"],
        ["RASI", "PVUL"],
        ["LASI", "LIC"],
        ["LASI", "PVUR"],
        ["LASI", "PVUL"],
        ["PVUR", "PVUL"],
    ],


    "PVUR": [
        ["PVUL", "RASI"],
        ["PVUL", "LASI"],
        ["PVLR", "RASI"],
        ["PVLR", "LASI"],
        ["RASI", "LASI"],
        ["RASI", "PVLR"],
        ["LASI", "PVLR"],
        ["RIC", "PVUL"],
        ["LIC", "PVUL"],
    ],


    "PVLR": [
        ["PVLL", "RASI"],
        ["PVLL", "LASI"],
        ["PVUR", "RASI"],
        ["PVUR", "LASI"],
        ["RASI", "LASI"],
        ["RASI", "PVUR"],
        ["LASI", "PVUR"],
        ["RIC", "PVLL"],
        ["LIC", "PVLL"],
    ],


    "LASI": [
        ["RASI", "LIC"],
        ["RASI", "RIC"],
        ["RASI", "PVUL"],
        ["RASI", "PVUR"],
        ["LIC", "PVUL"],
        ["LIC", "PVUR"],
        ["RIC", "PVUL"],
        ["RIC", "PVUR"],
    ],

    "LIC": [
        ["LASI", "RASI"],
        ["LASI", "RIC"],
        ["LASI", "PVUL"],
        ["LASI", "PVUR"],
        ["RASI", "RIC"],
        ["RASI", "PVUL"],
        ["RASI", "PVUR"],
        ["PVUL", "PVUR"],
    ],


    "PVUL": [
        ["PVUR", "LASI"],
        ["PVUR", "RASI"],
        ["PVLL", "LASI"],
        ["PVLL", "RASI"],
        ["LASI", "RASI"],
        ["LASI", "PVLL"],
        ["RASI", "PVLL"],
        ["LIC", "PVUR"],
        ["RIC", "PVUR"],
    ],


    "PVLL": [
        ["PVLR", "LASI"],
        ["PVLR", "RASI"],
        ["PVUL", "LASI"],
        ["PVUL", "RASI"],
        ["LASI", "RASI"],
        ["LASI", "PVUL"],
        ["RASI", "PVUL"],
        ["LIC", "PVLR"],
        ["RIC", "PVLR"],
    ],
}

def CBRU_pelvis_gap_fill_relational():
    gap_fill_relational(base_marker_names, base_gap_fill_rules)

from .gap_fill_relational import gap_fill_relational


# -------------------- Marker Definitions -------------------- #
# Define thigh markers and rename them based on prefix in QTM
base_marker_names = [
    "RGT",
    "RTHPP",
    "RTHPA",
    "RTHDP",
    "RTHDA",
    "RKNL",
    "RKNM",
    "RSHPA",
    "RSHPP",
    "RSHDA",
    "RSHDP",
    "RANL",
    "RANM",
    "LGT",
    "LTHPP",
    "LTHPA",
    "LTHDP",
    "LTHDA",
    "LKNL",
    "LKNM",
    "LSHPA",
    "LSHPP",
    "LSHDA",
    "LSHDP",
    "LANL",
    "LANM"]

# Define hierarchy for reference markers in relational gap filling method
base_gap_fill_rules = {

    "RGT": [
        ["RTHPP", "RTHPA"],
        ["RTHPP", "RTHDP"],
        ["RTHPA", "RTHDA"],
        ["RTHDP", "RTHDA"],
    ],


    "RTHPP": [
        ["RGT", "RTHPA"],
        ["RGT", "RTHDP"],
        ["RTHPA", "RTHDP"],
        ["RGT", "RTHDA"],
    ],


    "RTHPA": [
        ["RGT", "RTHPP"],
        ["RGT", "RTHDA"],
        ["RGT", "RTHDP"],
        ["RTHPP", "RTHDA"],
    ],


    "RTHDP": [
        ["RTHPP", "RKNL"],
        ["RTHPP", "RKNM"],
        ["RTHDA", "RKNL"],
        ["RTHDA", "RKNM"],
        ["RTHPA", "RKNL"],
        ["RTHPA", "RKNM"],
    ],


    "RTHDA": [
        ["RTHPA", "RKNL"],
        ["RTHPA", "RKNM"],
        ["RTHDP", "RKNL"],
        ["RTHDP", "RKNM"],
        ["RTHPP", "RKNL"],
        ["RTHPP", "RKNM"],
    ],


    "RKNL": [
        ["RKNM", "RTHDP"],
        ["RKNM", "RTHDA"],
        ["RTHDP", "RSHPP"],
        ["RTHDA", "RSHPA"],
        ["RTHDP", "RSHPA"],
        ["RTHDA", "RSHPP"],
    ],


    "RKNM": [
        ["RKNL", "RTHDP"],
        ["RKNL", "RTHDA"],
        ["RTHDP", "RSHPP"],
        ["RTHDA", "RSHPA"],
        ["RTHDP", "RSHPA"],
        ["RTHDA", "RSHPP"],
    ],


    "RSHPP": [
        ["RKNL", "RSHPA"],
        ["RKNM", "RSHPA"],
        ["RKNL", "RSHDP"],
        ["RKNM", "RSHDP"],
        ["RKNL", "RSHDA"],
    ],


    "RSHPA": [
        ["RKNL", "RSHPP"],
        ["RKNM", "RSHPP"],
        ["RKNL", "RSHDA"],
        ["RKNM", "RSHDA"],
        ["RKNL", "RSHDP"],
    ],


    "RSHDP": [
        ["RSHPP", "RANL"],
        ["RSHPP", "RANM"],
        ["RSHPA", "RANL"],
        ["RSHPA", "RANM"],
        ["RKNL", "RANL"],
        ["RKNM", "RANM"],
    ],


    "RSHDA": [
        ["RSHPA", "RANL"],
        ["RSHPA", "RANM"],
        ["RSHPP", "RANL"],
        ["RSHPP", "RANM"],
        ["RKNL", "RANL"],
        ["RKNM", "RANM"],
    ],


    "RANL": [
        ["RANM", "RSHDP"],
        ["RANM", "RSHDA"],
        ["RSHDP", "RSHDA"],
        ["RSHPP", "RSHDA"],
        ["RSHPA", "RSHDP"],
    ],


    "RANM": [
        ["RANL", "RSHDP"],
        ["RANL", "RSHDA"],
        ["RSHDP", "RSHDA"],
        ["RSHPP", "RSHDA"],
        ["RSHPA", "RSHDP"],
    ],


    "LGT": [
        ["LTHPP", "LTHPA"],
        ["LTHPP", "LTHDP"],
        ["LTHPA", "LTHDA"],
        ["LTHDP", "LTHDA"],
    ],


    "LTHPP": [
        ["LGT", "LTHPA"],
        ["LGT", "LTHDP"],
        ["LTHPA", "LTHDP"],
        ["LGT", "LTHDA"],
    ],


    "LTHPA": [
        ["LGT", "LTHPP"],
        ["LGT", "LTHDA"],
        ["LGT", "LTHDP"],
        ["LTHPP", "LTHDA"],
    ],


    "LTHDP": [
        ["LTHPP", "LKNL"],
        ["LTHPP", "LKNM"],
        ["LTHDA", "LKNL"],
        ["LTHDA", "LKNM"],
        ["LTHPA", "LKNL"],
        ["LTHPA", "LKNM"],
    ],


    "LTHDA": [
        ["LTHPA", "LKNL"],
        ["LTHPA", "LKNM"],
        ["LTHDP", "LKNL"],
        ["LTHDP", "LKNM"],
        ["LTHPP", "LKNL"],
        ["LTHPP", "LKNM"],
    ],


    "LKNL": [
        ["LKNM", "LTHDP"],
        ["LKNM", "LTHDA"],
        ["LTHDP", "LSHPP"],
        ["LTHDA", "LSHPA"],
        ["LTHDP", "LSHPA"],
        ["LTHDA", "LSHPP"],
    ],


    "LKNM": [
        ["LKNL", "LTHDP"],
        ["LKNL", "LTHDA"],
        ["LTHDP", "LSHPP"],
        ["LTHDA", "LSHPA"],
        ["LTHDP", "LSHPA"],
        ["LTHDA", "LSHPP"],
    ],


    "LSHPP": [
        ["LKNL", "LSHPA"],
        ["LKNM", "LSHPA"],
        ["LKNL", "LSHDP"],
        ["LKNM", "LSHDP"],
        ["LKNL", "LSHDA"],
    ],


    "LSHPA": [
        ["LKNL", "LSHPP"],
        ["LKNM", "LSHPP"],
        ["LKNL", "LSHDA"],
        ["LKNM", "LSHDA"],
        ["LKNL", "LSHDP"],
    ],


    "LSHDP": [
        ["LSHPP", "LANL"],
        ["LSHPP", "LANM"],
        ["LSHPA", "LANL"],
        ["LSHPA", "LANM"],
        ["LKNL", "LANL"],
        ["LKNM", "LANM"],
    ],


    "LSHDA": [
        ["LSHPA", "LANL"],
        ["LSHPA", "LANM"],
        ["LSHPP", "LANL"],
        ["LSHPP", "LANM"],
        ["LKNL", "LANL"],
        ["LKNM", "LANM"],
    ],


    "LANL": [
        ["LANM", "LSHDP"],
        ["LANM", "LSHDA"],
        ["LSHDP", "LSHDA"],
        ["LSHPP", "LSHDA"],
        ["LSHPA", "LSHDP"],
    ],


    "LANM": [
        ["LANL", "LSHDP"],
        ["LANL", "LSHDA"],
        ["LSHDP", "LSHDA"],
        ["LSHPP", "LSHDA"],
        ["LSHPA", "LSHDP"],
    ],
}

def CBRU_leg_gap_fill_relational():
    gap_fill_relational(base_marker_names, base_gap_fill_rules)

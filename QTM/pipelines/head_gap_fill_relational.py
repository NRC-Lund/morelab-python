from .gap_fill_relational import gap_fill_relational


base_marker_names = ["HeadL", "HeadR", "HeadFront"]

base_gap_fill_rules = {
    "HeadL": [
        ["HeadR", "HeadFront"],
        ["HeadR"],
        ["HeadFront"],
    ],
    "HeadR": [
        ["HeadL", "HeadFront"],
        ["HeadL"],
        ["HeadFront"],
    ],
    "HeadFront": [
        ["HeadL", "HeadR"],
        ["HeadL"],
        ["HeadR"],
    ],
}

def head_gap_fill_relational():
    gap_fill_relational(base_marker_names, base_gap_fill_rules)

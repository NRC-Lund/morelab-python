from .gap_fill_relational import add_marker_prefix, gap_fill_relational


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

marker_names, gap_fill_rules = add_marker_prefix(
    base_marker_names, base_gap_fill_rules, "Q_")


def head_gap_fill_relational():
    gap_fill_relational(marker_names, gap_fill_rules)

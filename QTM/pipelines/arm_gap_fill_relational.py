from .gap_fill_relational import add_marker_prefix, gap_fill_relational


base_marker_names = [
    "RArm",
    "RElbowOut",
    "RElbowIn",
    "RWristOut",
    "RWristIn",
    "RHand2",
    "LArm",
    "LElbowOut",
    "LElbowIn",
    "LWristOut",
    "LWristIn",
    "LHand2",
]

base_gap_fill_rules = {
    "RArm": [["RElbowOut", "RElbowIn"]],
    "RElbowOut": [["RArm", "RElbowIn"]],
    "RElbowIn": [["RArm", "RElbowOut"]],
    "RWristOut": [["RWristIn", "RHand2"]],
    "RWristIn": [["RWristOut", "RHand2"]],
    "RHand2": [["RWristOut", "RWristIn"]],

    "LArm": [["LElbowOut", "LElbowIn"]],
    "LElbowOut": [["LArm", "LElbowIn"]],
    "LElbowIn": [["LArm", "LElbowOut"]],
    "LWristOut": [["LWristIn", "LHand2"]],
    "LWristIn": [["LWristOut", "LHand2"]],
    "LHand2": [["LWristOut", "LWristIn"]],
}

marker_names, gap_fill_rules = add_marker_prefix(
    base_marker_names, base_gap_fill_rules, "Q_")


def arm_gap_fill_relational():
    gap_fill_relational(marker_names, gap_fill_rules)

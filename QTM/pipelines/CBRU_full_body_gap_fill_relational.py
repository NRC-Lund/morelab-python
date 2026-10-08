from .gap_fill_relational import gap_fill_relational
from . import CBRU_head_gap_fill_relational
from . import CBRU_torso_gap_fill_relational
from . import CBRU_arm_gap_fill_relational
from . import CBRU_pelvis_gap_fill_relational
from . import CBRU_leg_gap_fill_relational
from . import CBRU_foot_gap_fill_relational


segments = [
    ("Head", CBRU_head_gap_fill_relational),
    ("Torso", CBRU_torso_gap_fill_relational),
    ("Arm & Hand", CBRU_arm_gap_fill_relational),
    ("Pelvis", CBRU_pelvis_gap_fill_relational),
    ("Leg", CBRU_leg_gap_fill_relational),
    ("Foot", CBRU_foot_gap_fill_relational),
]

def CBRU_full_body_gap_fill_relational():
    marker_names = []
    gap_fill_rules = {}

    for _, segment in segments:
        marker_names += [
            marker for marker in segment.base_marker_names
            if marker not in marker_names
        ]
        for marker, rules in segment.base_gap_fill_rules.items():
            gap_fill_rules.setdefault(marker, []).extend(rules)

    gap_fill_relational(marker_names, gap_fill_rules)

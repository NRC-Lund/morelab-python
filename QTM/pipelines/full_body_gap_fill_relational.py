from .gap_fill_relational import gap_fill_relational
from . import head_gap_fill_relational
from . import torso_gap_fill_relational
from . import arm_gap_fill_relational
from . import pelvis_gap_fill_relational
from . import thigh_gap_fill_relational
from . import shank_gap_fill_relational
from . import foot_gap_fill_relational


segments = [
    ("Head", head_gap_fill_relational),
    ("Torso", torso_gap_fill_relational),
    ("Arm & Hand", arm_gap_fill_relational),
    ("Pelvis", pelvis_gap_fill_relational),
    ("Thigh", thigh_gap_fill_relational),
    ("Shank", shank_gap_fill_relational),
    ("Foot", foot_gap_fill_relational),
]


def full_body_gap_fill_relational():
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

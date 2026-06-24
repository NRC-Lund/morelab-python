import qtm

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
    value = qtm.gui.dialog.show_string_input_dialog(
        "Max gap fill range",
        "What is the max gap length (in frames) you'd like to fill?",
        "25",
    )
    max_gap_length = 25 if value is None else int(value)

    for name, segment in segments:
        print(f"--- {name} ---")
        gap_fill_relational(
            segment.marker_names,
            segment.gap_fill_rules,
            max_gap_length=max_gap_length,
            ask_max_gap_length=False,
        )

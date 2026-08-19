# pipelines/__init__.py

# bring in the pipelines sub-modules themselves
from . import marker_filtering, calibrations, fix_sips, other

# and then grab whatever functions you want at the top-level
from .marker_filtering import apply_butterworth_filter_to_marker_set
from .calibrations  import static_calibration, dynamic_calibration
from .other         import other_stuff

__all__ = [
    # if you ever do `from pipelines import *`
    "marker_filtering",
    "calibrations",
    "fix_sips",
    "other",
    "apply_butterworth_filter_to_marker_set",
    "static_calibration",
    "dynamic_calibration",
    "other_stuff",
]

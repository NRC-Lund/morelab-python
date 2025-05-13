# pipelines/__init__.py

# bring in the pipelines sub-modules themselves
from . import custom_filters, calibrations, fix_sips, other

# and then grab whatever functions you want at the top-level
from .custom_filters       import apply_butterworth_filter
from .calibrations  import static_calibration, dynamic_calibration
from .other         import other_stuff

__all__ = [
    # if you ever do `from pipelines import *`
    "custom_filters",
    "calibrations",
    "fix_sips",
    "other",
    "apply_butterworth_filter",
    "static_calibration",
    "dynamic_calibration",
    "other_stuff",
]
# pipelines/__init__.py

# Bring all of your pipeline functions into the pipelines namespace
from .filters import apply_butterworth_filter
from .calibrations import static_calibration, dynamic_calibration
from .other import other_stuff

__all__ = [
    "apply_butterworth_filter",
    "static_calibration",
    "dynamic_calibration",
    "other_stuff",
]
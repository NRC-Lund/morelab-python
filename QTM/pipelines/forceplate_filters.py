import numpy as np
from scipy.signal import butter, filtfilt
import qtm


def apply_forceplate_filter(order=10, cutoff=1.0, fs=1000):
    series_ids = qtm.data.series.force.get_series_ids()
    if not series_ids:
        print("No force data.")
        return

    sr = qtm.data.series.force.get_sample_ranges(series_ids[0])
    start, end = sr[0]["start"], sr[-1]["end"]
    data = qtm.data.series.force.get_samples(
        series_ids[0], {"start": start, "end": end})

    F = np.array([d["force"] for d in data])
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    Ff = filtfilt(b, a, F, axis=0)

    new = []
    for i, d in enumerate(data):
        new.append({
            "force": Ff[i].tolist(),
            "moment": d.get("moment"),
            "center_of_pressure": d.get("center_of_pressure"),
        })
    qtm.data.series.force.set_samples(
        series_ids[0], {"start": start, "end": end}, new)
    print("Forceplate filter applied.")

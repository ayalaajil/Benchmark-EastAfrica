"""
Regression test for the NeuralGCM zarr corruption discovered while running
verification: every NeuralGCM pred_YYYY-MM-DD.zarr stored the exact same
garbage int64 as its init_time coordinate (independent of which date the file
was actually for), which crashed xarray's CF time decoding with an
OverflowError. benchmark_ea.analysis_io.open_pred_da works around this by
deriving init_time from the (authoritative) filename instead of trusting the
store's own coordinate, opened with decode_times=False so a bad encoding in
one file/model can never crash loading of any other.
"""

import numpy as np
import pandas as pd
import xarray as xr

from benchmark_ea.analysis_io import open_pred_da


def _write_pred_zarr(path, init_date, corrupt_init_time=False):
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])
    data = np.zeros((1, 1, 1, len(lat), len(lon)))
    # A deliberately garbage int64 offset, mirroring the real corruption:
    # identical across every file regardless of the actual init date.
    init_time_value = np.array([9221120237041090560]) if corrupt_init_time else np.array([0])
    ds = xr.Dataset(
        {"total_precipitation": (("init_time", "sample", "lead_day", "lat", "lon"), data)},
        coords={
            "init_time": ("init_time", init_time_value,
                         {"units": f"days since {init_date} 00:00:00",
                          "calendar": "proleptic_gregorian"}),
            "sample": [0], "lead_day": [1], "lat": lat, "lon": lon,
        },
    )
    ds.to_zarr(path)


def test_open_pred_da_recovers_date_from_filename_despite_corrupt_coordinate(tmp_path):
    path = tmp_path / "pred_2024-03-07.zarr"
    _write_pred_zarr(path, "2024-03-07", corrupt_init_time=True)

    # A naive xr.open_zarr(path) would raise OverflowError/ValueError here —
    # this is exactly the crash the fix avoids.
    da = open_pred_da(path)
    assert da.init_time.values[0] == np.datetime64("2024-03-07")


def test_open_pred_da_matches_filename_even_when_coordinate_is_healthy(tmp_path):
    path = tmp_path / "pred_2024-06-15.zarr"
    _write_pred_zarr(path, "2024-06-15", corrupt_init_time=False)

    da = open_pred_da(path)
    assert da.init_time.values[0] == np.datetime64("2024-06-15")


def test_open_pred_da_concat_preserves_per_file_dates(tmp_path):
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    parts = []
    for d in dates:
        path = tmp_path / f"pred_{d}.zarr"
        _write_pred_zarr(path, d, corrupt_init_time=True)
        parts.append(open_pred_da(path))
    combined = xr.concat(parts, dim="init_time")
    assert list(combined.init_time.values) == [np.datetime64(d) for d in dates]

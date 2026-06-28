"""
Shared loaders for the East-Africa skill analyses.

Keeps prediction loading and the CHIRPS/TAMSAT truth switch in one place so the
CRPSS-map and ACC scripts stay short and the climatology source is trivial to
change (just pass 'chirps' or 'tamsat').
"""
import glob, os, sys
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.abspath("."))
from benchmark_ea.config import BenchmarkConfig
from benchmark_ea.truth import chirps as chirps_io
from benchmark_ea.truth import tamsat as tamsat_io

MODELS = ["fourcastnet", "gencast", "graphcast"]
LABELS = {"fourcastnet": "FourCastNet", "gencast": "GenCast", "graphcast": "GraphCast"}
COLORS = {"fourcastnet": "#d4a017", "gencast": "#2196F3", "graphcast": "#E53935"}

START, END, OBS_END = "2024-03-01", "2024-05-31", "2024-06-07"  # END + max lead (7d)
INIT_DATES = pd.date_range(START, END, freq="D")

_TRUTH = {"chirps": (chirps_io, "chirps"), "tamsat": (tamsat_io, "tamsat")}


def load_predictions():
    """{model: DataArray(init_time, sample, lead_day, lat, lon)} of precip."""
    preds = {}
    for m in MODELS:
        files = sorted(glob.glob(f"data/predictions/{m}/pred_2024-*.zarr"))
        preds[m] = xr.concat([xr.open_zarr(f)["total_precipitation"] for f in files],
                             dim="init_time")
    return preds


def load_truth(name="chirps"):
    """
    Load a truth/climatology source on the model grid.

    Returns
    -------
    obs_2d : {date: (lat, lon) array}  daily rainfall, NaN over ocean
    lat, lon : 1-D coordinate arrays
    """
    name = name.lower()
    if name not in _TRUTH:
        raise ValueError(f"truth must be one of {list(_TRUTH)}, got {name!r}")
    io, subdir = _TRUTH[name]
    cfg = BenchmarkConfig()
    da = io.load(START, OBS_END, cfg.lat_vals, cfg.lon_vals,
                 f"{cfg.data_dir}/{subdir}", download_missing=False)
    obs_2d = {pd.Timestamp(t).date(): da.sel(time=t).values for t in da.time.values}
    return obs_2d, da.lat.values, da.lon.values


def gather_pairs(preds, obs_2d, model, lead):
    """Align a model's forecasts with truth at a given lead day.

    Returns
    -------
    fc_all : (case, sample, lat, lon)
    ob_all : (case, lat, lon)
    """
    fc_da = preds[model].sel(lead_day=lead)
    fcs, obs = [], []
    for init in INIT_DATES:
        vd = (init + pd.Timedelta(days=lead)).date()
        if vd not in obs_2d:
            continue
        try:
            fc = fc_da.sel(init_time=init).values
        except KeyError:
            continue
        fcs.append(fc)
        obs.append(obs_2d[vd])
    return np.stack(fcs), np.stack(obs)

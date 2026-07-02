"""
CHIRPS day-of-year climatology baseline.

For each verification date the "forecast" is the set of observed CHIRPS
values on the same calendar day-of-year (DOY) across the reference years.
This gives a proper probabilistic baseline with the same spatial coverage
as the verification data — strictly out-of-sample relative to the
evaluation year.

The resulting ensemble has one member per reference year (default: 2000-2020,
21 members).

Output format follows the canonical benchmark zarr spec (models/base.py):
  total_precipitation: (init_time=1, sample=n_ref_years, lead_day, lat, lon)
  units: mm/day
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from benchmark_ea.config import BenchmarkConfig
from benchmark_ea.models.base import ModelAdapter
from benchmark_ea.truth import chirps as chirps_io

_DEFAULT_REF_YEARS = list(range(2000, 2021))


class ClimatologyAdapter(ModelAdapter):
    name        = "climatology"
    is_ensemble = True

    def __init__(self, ref_years: list[int] | None = None):
        self.ref_years: list[int] = ref_years or _DEFAULT_REF_YEARS

    def run_inference(self, config: BenchmarkConfig) -> Path:
        """
        Build climatology zarr files for every init_time in the eval window.

        For init_time t and lead_day l the valid date is t+l.  The climatology
        "ensemble" consists of CHIRPS observations on DOY(t+l) from each
        reference year.  Leap-day handling: DOY 366 falls back to DOY 365.
        """
        out_dir = self.predictions_path(config)
        out_dir.mkdir(parents=True, exist_ok=True)

        init_times = pd.date_range(config.eval_start, config.eval_end, freq="D")
        max_lead   = max(config.lead_days)

        def _zarr_path(t):
            return out_dir / f"pred_{t.strftime('%Y-%m-%d')}.zarr"

        # Resume-friendly: figure out which init dates still need writing BEFORE
        # any expensive CHIRPS download/load. If everything is already present,
        # return immediately rather than re-downloading ~20 GB of reference years.
        pending = [t for t in init_times
                   if config.overwrite or not _zarr_path(t).exists()]
        if not pending:
            print(f"Climatology: all {len(init_times)} files present → nothing to do.")
            return out_dir
        print(f"Climatology: {len(pending)} of {len(init_times)} init dates to write.")

        print(f"Loading CHIRPS for {len(self.ref_years)} reference years …")
        ref_data = self._load_ref_data(config)

        for n, init_time in enumerate(pending, 1):
            zarr_path = _zarr_path(init_time)

            # (n_leads, n_members, lat, lon)
            lead_slices = []
            for lead in config.lead_days:
                valid_date = init_time + pd.Timedelta(days=lead)
                doy        = int(valid_date.day_of_year)
                members    = _doy_members(ref_data, doy)  # (n_years, lat, lon)
                lead_slices.append(members)

            # stack → (n_leads, n_members, lat, lon)
            stacked = np.stack(lead_slices, axis=0)  # (n_leads, n_years, lat, lon)
            # reorder → (n_members, n_leads, lat, lon) for canonical format
            stacked = stacked.transpose(1, 0, 2, 3)

            ds = xr.Dataset(
                {
                    "total_precipitation": xr.DataArray(
                        stacked[np.newaxis],  # add init_time dim → (1, n_mem, n_lead, lat, lon)
                        dims=["init_time", "sample", "lead_day", "lat", "lon"],
                        coords={
                            "init_time": [init_time.to_datetime64()],
                            "sample":    np.arange(stacked.shape[0]),
                            "lead_day":  config.lead_days,
                            "lat":       config.lat_vals,
                            "lon":       config.lon_vals,
                        },
                        attrs={"units": "mm/day"},
                    )
                }
            )
            ds.to_zarr(zarr_path, mode="w")
            if n % 30 == 0 or n == len(pending):
                print(f"  {n}/{len(pending)} written  (latest {init_time.date()})")

        print(f"Climatology written → {out_dir}  ({len(pending)} new files)")
        return out_dir

    def _load_ref_data(self, config: BenchmarkConfig) -> xr.DataArray:
        """
        Load all reference-year CHIRPS data into a single DataArray.

        Uses the lean per-year loader (``chirps_io.load_year_ea``): each year's
        global file is downloaded, cropped+regridded to the EA grid, cached as a
        ~1 MB file, and the ~1 GB global file is then DISCARDED. So we never store
        the 21 global reference files. (Verification's ``chirps_io.load`` path is
        untouched.)
        """
        slices = []
        for year in self.ref_years:
            try:
                da = chirps_io.load_year_ea(
                    year,
                    config.lat_vals,
                    config.lon_vals,
                    config.chirps_cache_dir,
                    download_missing=True,
                )
                slices.append(da)
            except Exception as exc:
                print(f"  Skipping CHIRPS {year}: {exc}")

        if not slices:
            raise RuntimeError("No reference years loaded — check CHIRPS cache.")

        return xr.concat(slices, dim="time")


# ── helpers ───────────────────────────────────────────────────────────────────

def _doy_members(da: xr.DataArray, doy: int) -> np.ndarray:
    """
    Return all observations with the given day-of-year as a numpy array.

    DOY 366 (Dec 31 in a leap year) is collapsed to 365 so that *every*
    reference year contributes exactly one member — otherwise only leap
    reference years would match DOY 366, giving a member count that differs
    from the other lead days and breaking the downstream ``np.stack``.
    """
    if doy == 366:
        doy = 365
    mask = da.time.dt.dayofyear == doy
    if not mask.any():
        # Defensive fallback: no year has this DOY at all.
        mask = da.time.dt.dayofyear == (doy - 1)
    return da.sel(time=mask).values  # (n_years, lat, lon)

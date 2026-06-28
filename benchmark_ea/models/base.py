"""
Abstract base for all model adapters.

One zarr store per init_time:  pred_YYYY-MM-DD.zarr

Dataset variables:
  total_precipitation : float32 (init_time=1, sample, lead_day, lat, lon)
      Units: mm/day  (adapters are responsible for unit conversion).
      Deterministic models use sample=1.

Coordinates:
  init_time : datetime64[ns]  (length 1 — the initialisation date)
  sample    : int             (0-indexed ensemble member)
  lead_day  : int             (1-indexed calendar days ahead, e.g. [1, 3, 5, 7])
  lat, lon  : float32         (ascending, matching BenchmarkConfig grid)
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import xarray as xr

from benchmark_ea.config import BenchmarkConfig


class ModelAdapter(ABC):
    name: str          # unique slug → used for predictions_dir / results_dir
    is_ensemble: bool  # False for deterministic models (they write sample=1)

    @abstractmethod
    def run_inference(self, config: BenchmarkConfig) -> Path:
        """
        Run inference for config.eval_start … config.eval_end and write one
        pred_YYYY-MM-DD.zarr per init_time under predictions_path(config).

        Returns the predictions directory.
        """
        ...

    def predictions_path(self, config: BenchmarkConfig) -> Path:
        return config.predictions_dir / self.name

    @staticmethod
    def should_skip(zarr_path: Path, required_leads: list[int]) -> bool:
        """
        Return True only if zarr_path exists AND contains all required lead days.
        If the file exists but is incomplete (e.g. from a previous test run with
        fewer lead days), return False so it gets regenerated.
        """
        if not zarr_path.exists():
            return False
        try:
            ds = xr.open_zarr(str(zarr_path))
            saved = set(int(v) for v in ds.lead_day.values)
            missing = set(required_leads) - saved
            if missing:
                print(f"    → incomplete (missing lead_days {sorted(missing)}), regenerating")
                return False
            return True
        except Exception:
            return False

    @staticmethod
    def assemble_output(
        canonical_output: xr.Dataset,
        raw_full: "xr.Dataset | None",
        date,
        config: BenchmarkConfig,
        *,
        precip_raw_vars: tuple[str, ...] = (),
        sample_dim: "str | None" = None,
    ) -> xr.Dataset:
        """
        Assemble the dataset to persist for one init date.

        When ``config.save_variables == 'precip'`` (or no raw output is given),
        only the canonical daily ``total_precipitation`` is saved. When
        ``'all'``, every other variable in ``raw_full`` is regridded/subset to
        the East Africa grid (instantaneous snapshot per lead day, ensemble mean
        or all members per ``config.extra_var_members``) and merged in.

        ``precip_raw_vars`` lists the raw precip-related variable names to drop
        from the extra set (they are already represented canonically).
        """
        if config.save_variables != "all" or raw_full is None:
            return canonical_output

        # Imported here to avoid a heavy import when only precip is saved.
        from benchmark_ea import regrid

        extra = regrid.extra_vars_to_canonical(
            raw_full, date, config,
            drop_vars=tuple(precip_raw_vars),
            sample_dim=sample_dim,
        )
        return xr.merge([canonical_output, extra], compat="override")

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BenchmarkConfig:
    # Domain — 1° grid matching the GenCast/GraphCast output resolution
    lat_min: float = -12.0
    lat_max: float =  15.0
    lon_min: float =  28.0
    lon_max: float =  52.0
    grid_res: float = 1.0

    # Evaluation period
    eval_start: str = "2024-03-01"
    eval_end:   str = "2024-04-30"

    # Lead days to verify (1-indexed; 1 = next-day forecast)
    lead_days: list = field(default_factory=lambda: [1, 3, 5, 7])

    # Exceedance thresholds in mm/day for categorical metrics and BSS
    precip_thresholds_mm: list = field(default_factory=lambda: [1.0, 5.0, 10.0, 20.0])

    # ── Inference saving options ─────────────────────────────────────────────
    # Which variables to persist per init date:
    #   "precip" → only the canonical daily total_precipitation (smallest)
    #   "all"    → every variable the model outputs, regridded to the EA grid
    save_variables: str = "precip"
    # For ensemble models, how many members to keep for the *non-precip* extra
    # variables when save_variables == "all":
    #   "mean" → store the ensemble mean only (sample=1, cheap, bounds memory)
    #   "all"  → store every member (full fidelity, large)
    # Precipitation is ALWAYS stored for every member regardless of this.
    extra_var_members: str = "mean"
    # Ensemble size for stochastic models (GenCast).
    n_members: int = 10
    # Regenerate predictions even if a complete zarr already exists.
    overwrite: bool = False

    # Directory layout
    data_dir:    str = "data"
    results_dir: str = "results"
    # Override for the predictions output directory. When None, defaults to
    # <data_dir>/predictions. Set by the --output-dir CLI flag.
    output_dir:  str | None = None

    @property
    def lat_vals(self) -> np.ndarray:
        return np.arange(
            self.lat_min, self.lat_max + self.grid_res / 2, self.grid_res,
            dtype=np.float32,
        )

    @property
    def lon_vals(self) -> np.ndarray:
        return np.arange(
            self.lon_min, self.lon_max + self.grid_res / 2, self.grid_res,
            dtype=np.float32,
        )

    @property
    def predictions_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        return Path(self.data_dir) / "predictions"

    def __post_init__(self):
        if self.save_variables not in ("precip", "all"):
            raise ValueError(
                f"save_variables must be 'precip' or 'all', got {self.save_variables!r}"
            )
        if self.extra_var_members not in ("mean", "all"):
            raise ValueError(
                f"extra_var_members must be 'mean' or 'all', got {self.extra_var_members!r}"
            )

    @property
    def chirps_cache_dir(self) -> Path:
        return Path(self.data_dir) / "chirps"

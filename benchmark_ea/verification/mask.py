"""
Common land mask for cross-reference verification.

CHIRPS is NaN over ocean and large inland lakes at the source resolution, so
its regridded field already carries a land/ocean boundary — but ERA5 (a
reanalysis) has values everywhere, land or ocean. Without an explicit common
mask, ERA5-verified scores silently pool all 700 grid cells (including ~188
ocean cells) while CHIRPS/TAMSAT-verified scores only ever see the ~512 land
cells, which is not a fair cross-reference comparison and violates the
"single common land mask" the methodology states.

This module derives that mask from CHIRPS validity — the reference product
that actually carries land/ocean information — and applies it uniformly to
every truth source, so all three references are scored over the identical
set of cells.
"""

import numpy as np


def land_mask(chirps_da, valid_frac=0.99):
    """Boolean (lat, lon) mask: True where CHIRPS is finite on at least
    ``valid_frac`` of the loaded days.

    A near-1.0 threshold (rather than requiring every single day) tolerates
    the odd missing/corrupt CHIRPS day without reclassifying a land cell as
    ocean. Ocean and large inland-lake cells, which are NaN on every day,
    fail this threshold regardless of the exact value chosen.
    """
    valid = np.isfinite(chirps_da.values)          # (time, lat, lon)
    frac  = valid.mean(axis=0)                      # (lat, lon)
    mask  = frac >= valid_frac
    print(f"  land mask: {int(mask.sum())} / {mask.size} cells "
          f"({100 * mask.mean():.1f}%) from CHIRPS validity "
          f"(>= {100 * valid_frac:.0f}% of days)")
    return mask


def apply_land_mask(da_2d_dict, mask):
    """Return a new {date: (lat, lon) array} with non-land cells set to NaN.

    Applied identically to CHIRPS, ERA5 and TAMSAT so every downstream
    ``~np.isnan(obs)`` gather sees the same cell set regardless of reference.
    """
    return {d: np.where(mask, v, np.nan) for d, v in da_2d_dict.items()}

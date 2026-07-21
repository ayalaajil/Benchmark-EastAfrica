"""East Africa per-country rainfall climatology (MAM 2024, CHIRPS / TAMSAT)."""

import argparse
import os
import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import TwoSlopeNorm

from benchmark_ea.config import BenchmarkConfig
from benchmark_ea.domain import COUNTRIES, area_weights, country_masks
from benchmark_ea.truth import chirps as chirps_io
from benchmark_ea.truth import tamsat as tamsat_io
from benchmark_ea.verification.plots import _map_chrome
from benchmark_ea.verification.style import (
    BASELINE,
    CMAP_BIAS,
    HALF_WIDTH,
    INK2,
    NAN_COLOR,
    REF_LINE,
    apply_style,
    grid_y,
    panel_label,
    savefig,
)

MAM_MONTHS = [3, 4, 5]
OBS_LABELS = {"chirps": "CHIRPS", "tamsat": "TAMSAT"}
# Countries that get the extra MAM-anomaly-vs-normal delta map (CHIRPS only).
DELTA_COUNTRIES = ["Kenya", "Somalia"]
# Reference period for the MAM climatology (2018 is absent from the EA caches
# and is skipped automatically).
CLIM_YEARS = range(2000, 2021)

# Domain-aggregate row label written alongside the per-country rows.
DOMAIN_LABEL = "East Africa"
DOMAIN_COLOR = INK2
# Cells drier than this over the year are excluded from the fraction *field*
# (a near-zero annual denominator makes the MAM/annual ratio meaningless).
_MIN_ANNUAL_MM = 10.0
# A cell must have valid rainfall on at least this fraction of the year's days to
# count as land (matches the ≥99% validity convention used in verification.mask).
_VALID_FRAC = 0.99


# ── Data loading ──────────────────────────────────────────────────────────────

def load_obs_year(obs, year, cfg):
    """Load a full year of daily rainfall from ``obs`` on the 1° EA grid.

    Returns DataArray(time, lat, lon), mm/day, NaN over ocean."""
    if obs == "chirps":
        return chirps_io.load(f"{year}-01-01", f"{year}-12-31",
                              cfg.lat_vals, cfg.lon_vals, cfg.chirps_cache_dir,
                              download_missing=False)
    if obs == "tamsat":
        return tamsat_io.load(f"{year}-01-01", f"{year}-12-31",
                              cfg.lat_vals, cfg.lon_vals,
                              Path(cfg.data_dir) / "tamsat", download_missing=True)
    raise ValueError(f"obs must be one of {list(OBS_LABELS)}, got {obs!r}")


def mam_total_field(cfg, year, lat, lon):
    """MAM rainfall total (mm) for one year on the (lat, lon) grid, from CHIRPS.

    Only the MAM window is loaded/regridded. NaN over ocean (validity mask)."""
    da = chirps_io.load(f"{year}-03-01", f"{year}-05-31", lat, lon,
                        cfg.chirps_cache_dir, download_missing=False)
    valid = da.notnull().mean("time") >= _VALID_FRAC
    return da.sum("time").where(valid)


def mam_climatology_field(cfg, years, lat, lon, download_missing=True):
    """Mean MAM rainfall total (mm) over the reference ``years`` on the
    (lat, lon) grid, from the per-year EA CHIRPS caches (built on demand:
    ``load_year_ea`` downloads the native global, regrids, caches the small
    result, and deletes the ~1 GB global). The averaged field itself is cached
    as ``mam_clim_<lo>_<hi>_<gridhash>.nc`` so re-runs never re-download.

    Returns ((lat, lon) DataArray or None, list of years actually used)."""
    ghash = chirps_io._target_grid_hash(lat, lon)
    lo, hi = min(years), max(years)
    cache = Path(cfg.chirps_cache_dir) / f"mam_clim_{lo}_{hi}_{ghash}.nc"
    if cache.exists():
        da = xr.open_dataarray(cache)
        da.load()
        used = [int(y) for y in str(da.attrs.get("years_used", "")).split(",") if y]
        print(f"    climatology: reusing cached field {cache.name} "
              f"({len(used)} years)")
        return da, (used or list(years))

    totals, used = [], []
    for y in years:
        print(f"    climatology {y} … (downloads native CHIRPS if not cached)",
              flush=True)
        try:
            da = chirps_io.load_year_ea(y, lat, lon, cfg.chirps_cache_dir,
                                        download_missing=download_missing)
        except Exception as exc:                       # noqa: BLE001 (skip & continue)
            print(f"    climatology: {y} unavailable ({exc}) — skipped", flush=True)
            continue
        totals.append(da.sel(time=da.time.dt.month.isin(MAM_MONTHS)).sum("time"))
        used.append(y)
    if not totals:
        return None, []

    field = xr.concat(totals, dim="year").mean("year")
    field.attrs["years_used"] = ",".join(str(y) for y in used)
    field.to_netcdf(cache)
    print(f"    climatology: cached averaged field → {cache.name}")
    return field, used


# ── Stage 1: compute ──────────────────────────────────────────────────────────

def _weighted_mean(field, weights, cellmask):
    """Area-weighted mean of ``field`` over the cells selected by ``cellmask``.

    Returns np.nan when the selection is empty (e.g. a country too small to
    contain any 1° cell centre, like Rwanda/Burundi on the benchmark grid)."""
    w = weights.where(cellmask)
    denom = float(w.sum())
    if denom <= 0:
        return np.nan
    return float((field * w).sum() / denom)


def rainfall_fields(da):
    """Reduce daily rainfall to the MAM/annual total and MAM-fraction fields.

    Parameters
    ----------
    da : DataArray(time, lat, lon), mm/day, NaN over ocean — obs on the 1° grid.

    Returns
    -------
    dict with (lat, lon) DataArrays ``mam_total``/``annual_total``/``fraction``
    (mm, mm, %) plus scalars ``mam_days`` and ``valid`` (the land mask).
    """
    valid = da.notnull().mean("time") >= _VALID_FRAC            # (lat, lon) bool
    mam = da.sel(time=da.time.dt.month.isin(MAM_MONTHS))

    annual_total = da.sum("time").where(valid)
    mam_total = mam.sum("time").where(valid)
    fraction = (100.0 * mam_total / annual_total).where(annual_total >= _MIN_ANNUAL_MM)

    return dict(
        mam_total=mam_total,
        annual_total=annual_total,
        fraction=fraction,
        mam_days=int(mam.sizes["time"]),
        valid=valid,
    )


def country_rainfall_stats(fields, lat, lon):
    """Per-country (and whole-domain) area-weighted rainfall stats.

    Returns a DataFrame with one row per country in ``COUNTRIES`` order, plus a
    final ``DOMAIN_LABEL`` row aggregating every valid land cell in the domain.
    Columns: country, annual_total_mm, mam_total_mm, mam_fraction_pct,
    mam_mean_rate_mm_day, n_cells.
    """
    mam_total = fields["mam_total"]
    annual_total = fields["annual_total"]
    valid = fields["valid"]
    mam_days = fields["mam_days"]

    weights = area_weights(lat, lon)
    masks = country_masks(lat, lon)

    def _row(name, cellmask):
        mam = _weighted_mean(mam_total, weights, cellmask)
        annual = _weighted_mean(annual_total, weights, cellmask)
        frac = 100.0 * mam / annual if (annual and annual > 0) else np.nan
        return dict(
            country=name,
            annual_total_mm=annual,
            mam_total_mm=mam,
            mam_fraction_pct=frac,
            mam_mean_rate_mm_day=(mam / mam_days if np.isfinite(mam) else np.nan),
            n_cells=int((cellmask).sum()),
        )

    rows = []
    for name in COUNTRIES:
        mask = masks.get(name)
        cellmask = (mask & valid) if mask is not None else valid & False
        if mask is None:
            print(f"  warning: no Natural Earth polygon matched country {name!r}")
        rows.append(_row(name, cellmask))

    # Whole-domain aggregate over every valid land cell (independent of the
    # per-country polygons, so it also covers grid cells outside the 7 countries).
    rows.append(_row(DOMAIN_LABEL, valid))

    return pd.DataFrame(rows)


# ── Stage 2: figures (read the DataFrame; plain single-purpose figures) ────────

def _split_domain(df):
    """(per-country rows, domain row-or-None) from a stats DataFrame."""
    is_domain = df["country"] == DOMAIN_LABEL
    domain = df[is_domain].iloc[0] if is_domain.any() else None
    return df[~is_domain], domain


def plot_mam_fraction(df, out, obs_label, obs):
    """Bar chart: MAM rainfall as a % of the country's annual total, sorted
    descending, with a dashed reference line at the whole-domain fraction — the
    'which countries are most MAM-driven' figure."""
    countries, domain = _split_domain(df)
    countries = (countries.dropna(subset=["mam_fraction_pct"])
                 .sort_values("mam_fraction_pct", ascending=False))
    if countries.empty:
        return

    colors = [COUNTRIES.get(c, DOMAIN_COLOR) for c in countries["country"]]
    x = np.arange(len(countries))

    fig, ax = plt.subplots(figsize=(HALF_WIDTH + 1.6, 2.8))
    ax.bar(x, countries["mam_fraction_pct"], color=colors, width=0.72,
           edgecolor="white", linewidth=0.5)
    if domain is not None and np.isfinite(domain["mam_fraction_pct"]):
        ax.axhline(domain["mam_fraction_pct"], **REF_LINE)
        ax.text(len(countries) - 0.5, domain["mam_fraction_pct"],
                f"  East Africa mean {domain['mam_fraction_pct']:.0f}%",
                va="center", ha="left", fontsize=6.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(countries["country"], rotation=30, ha="right")
    ax.set_ylabel("MAM share of annual rainfall (%)")
    ax.set_ylim(bottom=0)
    grid_y(ax)
    panel_label(ax, "a")
    fig.suptitle(f"MAM rainfall as a share of the annual total, 2024 ({obs_label})")
    savefig(fig, out, f"regional_mam_fraction_{obs}")


def plot_rainfall_totals(df, out, obs_label, obs):
    """Grouped bars: MAM total vs full-year total rainfall per country (mm)."""
    countries, _ = _split_domain(df)
    countries = countries.dropna(subset=["annual_total_mm"])
    if countries.empty:
        return

    x = np.arange(len(countries))
    w = 0.38
    fig, ax = plt.subplots(figsize=(HALF_WIDTH + 1.6, 2.8))
    ax.bar(x - w / 2, countries["annual_total_mm"], w, label="Annual",
           color=BASELINE, edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, countries["mam_total_mm"], w, label="MAM",
           color="#1c5cab", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(countries["country"], rotation=30, ha="right")
    ax.set_ylabel("Rainfall total (mm)")
    ax.set_ylim(bottom=0)
    grid_y(ax)
    panel_label(ax, "a")
    ax.legend(loc="upper right")
    fig.suptitle(f"MAM vs annual rainfall total by country, 2024 ({obs_label})")
    savefig(fig, out, f"regional_rainfall_totals_{obs}")


def _map_figure(field, lat, lon, extent, cmap, norm, cbar_label, title, fname,
                out, extend="neither"):
    """One single-field cartopy map over ``extent``. ``extend`` adds the
    colorbar arrow(s) that flag a capped scale ("max"/"min"/"both")."""
    proj = ccrs.PlateCarree()
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(NAN_COLOR)

    fig, ax = plt.subplots(figsize=(HALF_WIDTH + 1.4, 4.4),
                           subplot_kw={"projection": proj})
    _map_chrome(ax, extent, proj, left_labels=True, bottom_labels=True)
    im = ax.pcolormesh(lon, lat, np.ma.masked_invalid(np.asarray(field)),
                       cmap=cm, norm=norm, transform=proj,
                       shading="nearest", zorder=1)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        fraction=0.05, pad=0.06, shrink=0.85, extend=extend)
    cbar.set_label(cbar_label, fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.4)
    fig.suptitle(title)
    savefig(fig, out, fname)


def _domain_extent(lat, lon):
    return [lon.min() - 0.5, lon.max() + 0.5, lat.min() - 0.5, lat.max() + 0.5]


def plot_rainfall_maps(fields, lat, lon, out, obs_label, obs):
    """The two spatial fields: MAM accumulation and MAM/annual fraction."""
    extent = _domain_extent(lat, lon)
    _map_figure(
        fields["mam_total"], lat, lon, extent, "YlGnBu", None,
        "MAM rainfall total (mm)",
        f"MAM rainfall accumulation, 2024 ({obs_label})",
        f"regional_mam_total_map_{obs}", out)
    _map_figure(
        fields["fraction"], lat, lon, extent, "PuBuGn", None,
        "MAM share of annual rainfall (%)",
        f"MAM share of annual rainfall, 2024 ({obs_label})",
        f"regional_mam_fraction_map_{obs}", out)


def plot_delta_map(mam_2024, mam_clim, country, masks, lat, lon, out,
                   obs, years_used):
    """MAM 2024 minus the reference-period MAM normal, masked and zoomed to one
    country — wetter-than-normal (green) vs drier-than-normal (brown)."""
    mask = masks.get(country)
    if mask is None:
        print(f"  delta: no polygon for {country} — skipped")
        return
    delta = (mam_2024 - mam_clim).where(mask)
    arr = np.asarray(delta)
    if not np.isfinite(arr).any():
        print(f"  delta: no valid cells for {country} — skipped")
        return

    rows, cols = np.where(np.asarray(mask))
    extent = [lon[cols].min() - 0.75, lon[cols].max() + 0.75,
              lat[rows].min() - 0.75, lat[rows].max() + 0.75]
    vmax = float(np.nanmax(np.abs(arr)))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    yr_txt = f"{min(years_used)}–{max(years_used)}"
    _map_figure(
        delta, lat, lon, extent, CMAP_BIAS, norm,
        f"MAM anomaly vs {yr_txt} normal (mm); green = wetter, brown = drier",
        f"{country}: MAM 2024 rainfall anomaly ({OBS_LABELS[obs]})",
        f"regional_delta_{country.lower()}_{obs}", out)


def plot_delta_map_domain(mam_2024, mam_clim, lat, lon, out, obs, years_used):
    """MAM 2024 minus the reference-period MAM normal over the WHOLE East Africa
    domain — same style as the per-country delta maps, just unmasked and at full
    extent (wetter-than-normal green, drier-than-normal brown)."""
    delta = mam_2024 - mam_clim
    arr = np.asarray(delta)
    if not np.isfinite(arr).any():
        print("  delta: no valid cells over the domain — skipped")
        return
    extent = _domain_extent(lat, lon)

    # Cap the symmetric scale at the 99th percentile of |anomaly| 
    vmax = float(np.percentile(np.abs(arr[np.isfinite(arr)]), 99))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    yr_txt = f"{min(years_used)}–{max(years_used)}"
    
    _map_figure(
        delta, lat, lon, extent, CMAP_BIAS, norm,
        f"MAM rainfall anomaly relative to the {yr_txt} years (mm), green = wetter, brown = drier",
        f"East Africa: MAM 2024 rainfall anomaly ({OBS_LABELS[obs]})",
        f"regional_delta_eastafrica_{obs}", out, extend="max")


def _cap_extend(finite, vmax):
    """Colorbar extend flag for a scale capped at ±vmax given the data range."""
    hi, lo = float(finite.max()), float(finite.min())
    if hi > vmax and lo < -vmax:
        return "both"
    return "max" if hi > vmax else "min" if lo < -vmax else "neither"


def plot_delta_pct_map_domain(mam_2024, mam_clim, lat, lon, out, obs, years_used,
                              min_normal_mm=25.0):
    """Percentage MAM rainfall anomaly over the whole domain:
    100·(2024 − normal)/normal at each cell — how much wetter/drier than the
    2000–2020 normal in relative terms. Same style as the absolute delta map
    (BrBG, 99th-pct cap + extend arrow). Cells whose normal MAM total is below
    ``min_normal_mm`` are masked: a near-zero denominator makes the percentage
    explode and become meaningless (the same guard the MAM-fraction field uses)."""
    pct = (100.0 * (mam_2024 - mam_clim) / mam_clim).where(mam_clim >= min_normal_mm)
    arr = np.asarray(pct)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        print("  delta%: no valid cells over the domain — skipped")
        return
    extent = _domain_extent(lat, lon)
    vmax = float(np.percentile(np.abs(finite), 99))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    yr_txt = f"{min(years_used)}–{max(years_used)}"
    _map_figure(
        pct, lat, lon, extent, CMAP_BIAS, norm,
        f"MAM rainfall anomaly relative to the {yr_txt} normal (%); "
        f"green = wetter, brown = drier",
        f"East Africa: MAM 2024 rainfall anomaly\n"
        f"relative to {yr_txt} normal (%)",
        f"regional_delta_pct_eastafrica_{obs}", out,
        extend=_cap_extend(finite, vmax))


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_regional_analysis(output_dir, data_dir, year, obs):
    cfg = BenchmarkConfig(data_dir=data_dir)
    out = os.path.join(output_dir, "regional")
    os.makedirs(out, exist_ok=True)
    label = OBS_LABELS[obs]

    print(f"\n=== {label} {year} ===")
    da = load_obs_year(obs, year, cfg)
    lat, lon = da.lat.values, da.lon.values
    print(f"  {label}  {dict(zip(da.dims, da.shape))}")

    fields = rainfall_fields(da)
    df = country_rainfall_stats(fields, lat, lon)

    csv_path = os.path.join(out, f"regional_rainfall_by_country_{obs}.csv")
    df.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"  saved → {os.path.basename(csv_path)}")

    plot_mam_fraction(df, out, label, obs)
    plot_rainfall_totals(df, out, label, obs)
    plot_rainfall_maps(fields, lat, lon, out, label, obs)
    return df


def build_delta_maps(output_dir, data_dir, year, res,
                     countries=DELTA_COUNTRIES, clim_years=CLIM_YEARS):
    """Kenya/Somalia MAM-anomaly-vs-normal delta maps at ``res``° (CHIRPS only).

    Both the ``year`` MAM total and the ``clim_years`` MAM climatology are built
    on the same finer grid so the anomaly carries genuine spatial detail (unlike
    upsampling the 1° fields, which would only interpolate). The climatology
    years are downloaded on demand and the averaged field is cached, so this is
    slow only on the first high-res run.
    """
    cfg = BenchmarkConfig(data_dir=data_dir, grid_res=res)
    out = os.path.join(output_dir, "regional")
    os.makedirs(out, exist_ok=True)
    lat, lon = cfg.lat_vals, cfg.lon_vals

    print(f"\n=== Delta maps @ {res}° (CHIRPS) ===")
    print(f"  building {year} MAM total @ {res}° …", flush=True)
    mam_current = mam_total_field(cfg, year, lat, lon)

    print(f"  building {min(clim_years)}–{max(clim_years)} MAM climatology "
          f"@ {res}° …", flush=True)
    clim, used = mam_climatology_field(cfg, clim_years, lat, lon,
                                       download_missing=True)
    if clim is None:
        print("  delta: no climatology years available — delta maps skipped")
        return
    print(f"  climatology from {len(used)} years ({min(used)}–{max(used)})")

    # Whole-domain anomaly (absolute mm and relative %), then the zoomed
    # per-country ones (same style).
    plot_delta_map_domain(mam_current, clim, lat, lon, out, "chirps", used)
    plot_delta_pct_map_domain(mam_current, clim, lat, lon, out, "chirps", used)
    masks = country_masks(lat, lon)
    for c in countries:
        plot_delta_map(mam_current, clim, c, masks, lat, lon, out, "chirps", used)


def main(argv):
    p = argparse.ArgumentParser(
        description="East Africa per-country rainfall climatology (CHIRPS/TAMSAT)")
    p.add_argument("--output-dir", default="./outputs_2024")
    p.add_argument("--data-dir", default="./data",
                   help="Root data dir; obs are read from <data-dir>/<obs>.")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--obs", nargs="+", default=["chirps", "tamsat"],
                   choices=list(OBS_LABELS),
                   help="Observation source(s) to run.")
    p.add_argument("--delta-res", type=float, default=0.25,
                   help="Grid resolution (°) for the Kenya/Somalia delta maps. "
                        "Finer than the 1° default re-downloads the climatology "
                        "years on first use (then caches). Ignored if 'chirps' "
                        "is not in --obs.")
    p.add_argument("--no-delta", action="store_true",
                   help="Skip the Kenya/Somalia delta maps.")
    args = p.parse_args(argv)

    apply_style()
    for obs in args.obs:
        run_regional_analysis(args.output_dir, args.data_dir, args.year, obs)

    # Delta maps are CHIRPS-only (the 2000–2020 EA climatology is CHIRPS) and
    # built once, at their own (finer) resolution.
    
    if "chirps" in args.obs and not args.no_delta:
        build_delta_maps(args.output_dir, args.data_dir, args.year,
                         args.delta_res)


if __name__ == "__main__":
    main(sys.argv[1:])

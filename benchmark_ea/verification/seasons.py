"""
East Africa seasonal stratification.

East Africa's rainfall is bimodal: the March-May "long rains" (MAM) and the
October-December "short rains" (OND), separated by two drier spells. Annual
averages mix these regimes together and can hide season-dependent skill (e.g.
a model that is strong in the well-observed long rains but weak in the
shorter, more convective short rains). Stratifying every score by the season
of the *valid* date — not the init date — lets each season's numbers reflect
only the days actually falling in it.
"""

import pandas as pd

SEASONS = ["annual", "MAM", "JJAS", "OND", "JF"]
# The four real seasons, excluding the "annual" full-period aggregate —
# for figures/loops that only make sense per-season (e.g. daily timeseries,
# where an "annual" version is what's being replaced, not one more bucket).
REAL_SEASONS = ["MAM", "JJAS", "OND", "JF"]

_MONTH_SEASON = {
    1: "JF", 2: "JF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJAS", 7: "JJAS", 8: "JJAS", 9: "JJAS",
    10: "OND", 11: "OND", 12: "OND",
}


def season_of(date) -> str:
    """Map a date (or anything with a ``.month`` attribute) to one of
    MAM / JJAS / OND / JF."""
    return _MONTH_SEASON[date.month]


def filter_by_season(init_dates, lead_day, season):
    """Filter ``init_dates`` to those whose *valid* date (init + lead_day)
    falls in ``season``. ``season`` of ``None`` or ``"annual"`` returns
    ``init_dates`` unchanged.

    For gatherers that take ``init_dates`` directly (the map-preserving
    ``analysis_io.gather_pairs`` used by ACC/SSR/CRPSS) rather than a
    ``season=`` kwarg, pre-filtering the init dates here is equivalent to
    filtering post-hoc on the valid date, and keeps season stratification
    consistent across every gatherer in the package.
    """
    if season is None or season == "annual":
        return init_dates
    keep = [d for d in init_dates
            if season_of((d + pd.Timedelta(days=lead_day)).date()) == season]
    return pd.DatetimeIndex(keep)


def filter_index_by_season(obj, season):
    """Subset a date/Timestamp-indexed Series/DataFrame to the rows whose
    index falls in ``season`` (e.g. the per-valid-date ``temporal`` frames
    from ``compute_temporal_metrics``). ``season == "annual"`` returns
    ``obj`` unchanged."""
    if season == "annual":
        return obj
    return obj[obj.index.map(season_of) == season]


def season_title(season: str, year: int = 2024) -> str:
    """Human-readable period for figure titles: 'MAM 2024' for the annual
    aggregate (the original default period this pipeline shipped with) or
    '<season> <year>' otherwise."""
    return f"MAM {year}" if season == "annual" else f"{season} {year}"

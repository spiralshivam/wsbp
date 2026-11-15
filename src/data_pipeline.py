"""Download NOAA GHCN-Daily station files and build station-month profiles."""
from __future__ import annotations

from pathlib import Path
import requests
import pandas as pd
import numpy as np

STATIONS = {
    "USW00094728": "New York Central Park", "USW00023174": "Los Angeles",
    "USW00012921": "Houston", "USW00024233": "Seattle",
    "USW00013874": "Atlanta", "USW00014732": "Boston",
    "USW00003927": "Dallas", "USW00012839": "Miami",
}
BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{}.dly"
ELEMENTS = {"TMAX", "TMIN", "PRCP"}


def download_station(station_id: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{station_id}.dly"
    if not path.exists():
        response = requests.get(BASE_URL.format(station_id), timeout=90)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def parse_dly(path: Path, start_year: int, end_year: int) -> pd.DataFrame:
    """Parse NOAA fixed-width GHCN-Daily records, retaining unflagged values."""
    records = []
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if len(line) < 269:
            continue
        station, year, month, element = line[:11], int(line[11:15]), int(line[15:17]), line[17:21]
        if not start_year <= year <= end_year or element not in ELEMENTS:
            continue
        for day in range(1, 32):
            offset = 21 + (day - 1) * 8
            value = int(line[offset:offset + 5])
            mflag, qflag, sflag = line[offset + 5:offset + 8]
            # NOAA M/S flags describe measurement/source; they do not by themselves
            # invalidate a reading. Q flags are the quality-control rejection flag.
            if value == -9999 or qflag.strip():
                continue
            try:
                date = pd.Timestamp(year, month, day)
            except ValueError:
                continue
            records.append((station, date, element, value / 10.0))
    if not records:
        return pd.DataFrame(columns=["station_id", "date", "element", "value"])
    return pd.DataFrame(records, columns=["station_id", "date", "element", "value"])


def prepare_profiles(raw_dir: Path, start_year: int, end_year: int,
                     station_ids: list[str] | None = None) -> pd.DataFrame:
    """Return monthly station profiles; measurement units: °C and mm."""
    observations = []
    for station_id in station_ids or list(STATIONS):
        frame = parse_dly(download_station(station_id, raw_dir), start_year, end_year)
        if not frame.empty:
            observations.append(frame)
    if not observations:
        raise ValueError("No valid observations found for the chosen stations and years.")
    daily = pd.concat(observations, ignore_index=True)
    wide = daily.pivot_table(index=["station_id", "date"], columns="element", values="value")
    wide = wide.rename_axis(columns=None).reset_index()
    if "TMAX" not in wide: wide["TMAX"] = np.nan
    if "TMIN" not in wide: wide["TMIN"] = np.nan
    if "PRCP" not in wide: wide["PRCP"] = np.nan
    wide["range_c"] = (wide.TMAX - wide.TMIN).clip(lower=0)
    wide["observed_elements"] = wide[["TMAX", "TMIN", "PRCP"]].notna().sum(axis=1)
    wide["year"] = wide.date.dt.year
    wide["month"] = wide.date.dt.month
    grouped = wide.groupby(["station_id", "year", "month"], as_index=False).agg(
        tmax_mean_c=("TMAX", "mean"), tmin_mean_c=("TMIN", "mean"),
        diurnal_range_c=("range_c", "mean"), precipitation_mm=("PRCP", "sum"),
        valid_days=("date", "nunique"), valid_measurements=("observed_elements", "sum"))
    days = pd.to_datetime(grouped[["year", "month"]].assign(day=1)).dt.days_in_month
    # Three eligible elements per day: TMAX, TMIN and PRCP.
    grouped["coverage"] = grouped.valid_measurements / (days * len(ELEMENTS))
    grouped["month_sin"] = np.sin(2 * np.pi * grouped.month / 12)
    grouped["month_cos"] = np.cos(2 * np.pi * grouped.month / 12)
    grouped["station_name"] = grouped.station_id.map(STATIONS).fillna(grouped.station_id)
    grouped["precipitation_log1p"] = np.log1p(grouped.precipitation_mm.clip(lower=0))
    return grouped


PROCESSED_NAME = "station_month_profiles.csv"


def load_profiles(root: Path, start_year: int, end_year: int, download: bool = False):
    """Return (profiles, source). Uses the frozen processed snapshot when it covers
    the requested years; otherwise downloads from NOAA (needs internet)."""
    processed = root / "data" / "processed" / PROCESSED_NAME
    if not download and processed.exists():
        frame = pd.read_csv(processed)
        if frame.year.min() <= start_year and frame.year.max() >= end_year:
            frame = frame[frame.year.between(start_year, end_year)].reset_index(drop=True)
            return frame, "processed snapshot"
    return prepare_profiles(root / "data" / "raw", start_year, end_year), "NOAA download"

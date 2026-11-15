"""REST API (Flask) for the weather-station clustering project.

Endpoints
  GET /api/health                      liveness check
  GET /api/meta                        stations, supported years, features, limits
  GET /api/analyze?start=2019&end=2024&k=2,3,4,5&distance=euclidean|manhattan

The API only reads the bundled processed snapshot (no outbound downloads), validates
every parameter, and returns JSON. The browser frontend in /frontend calls it.
"""
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
sys.path.insert(0, str(ROOT))

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from src.clustering import FEATURES, fit_configurations
from src.data_pipeline import STATIONS, load_profiles

MIN_YEAR, MAX_YEAR = 2019, 2024
ALLOWED_K = range(2, 7)
ALLOWED_DISTANCES = ("euclidean", "manhattan")

app = Flask(__name__)
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")  # set to the frontend URL in production


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def clean(value):
    """Make pandas/numpy values JSON-safe (NaN -> None)."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, float) and math.isnan(value):
            return None
    return value


def records(frame: pd.DataFrame):
    return [{c: clean(v) for c, v in row.items()} for row in frame.to_dict("records")]


@lru_cache(maxsize=64)
def run(start: int, end: int, ks: tuple, distance: str) -> dict:
    profiles, source = load_profiles(ROOT, start, end, download=False)
    clustered, metrics, stability, models = fit_configurations(profiles, ks, distance)
    groups = clustered.groupby("cluster").agg(
        profiles=("cluster", "size"), tmax_c=("tmax_mean_c", "mean"), tmin_c=("tmin_mean_c", "mean"),
        precipitation_mm=("precipitation_mm", "mean"), coverage=("coverage", "mean")).round(3).reset_index()
    counts = pd.crosstab(clustered.station_name, clustered.cluster)
    selected = metrics[metrics.selected].iloc[0]
    return {
        "source": source, "distance": distance, "profiles": int(len(clustered)),
        "selected_k": int(selected.k), "pca_variance_2d": round(float(selected.pca_variance_2d), 4),
        "stability": {k: round(v, 4) if isinstance(v, float) else v for k, v in stability.items()},
        "metrics": records(metrics.round(4)),
        "clusters": records(groups),
        "station_counts": {"stations": list(counts.index), "clusters": [int(c) for c in counts.columns],
                           "counts": counts.values.tolist()},
        "points": records(clustered[["station_name", "year", "month", "cluster", "pca_1", "pca_2"]].round(4)),
    }


def parse_args():
    try:
        start = int(request.args.get("start", MIN_YEAR))
        end = int(request.args.get("end", MAX_YEAR))
        ks = tuple(sorted({int(v) for v in request.args.get("k", "2,3,4,5").split(",") if v.strip()}))
    except ValueError:
        raise ValueError("start, end and k must be integers")
    distance = request.args.get("distance", "euclidean")
    if not (MIN_YEAR <= start <= end <= MAX_YEAR):
        raise ValueError(f"years must satisfy {MIN_YEAR} <= start <= end <= {MAX_YEAR}")
    if not ks or any(k not in ALLOWED_K for k in ks):
        raise ValueError("k must be a comma-separated list of integers between 2 and 6")
    if distance not in ALLOWED_DISTANCES:
        raise ValueError("distance must be euclidean or manhattan")
    return start, end, ks, distance


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.get("/")
def index():
    """Serve the browser app from the same origin as the API."""
    return send_from_directory(FRONTEND, "index.html")


@app.get("/<path:filename>")
def frontend_asset(filename):
    """Serve frontend files while preserving JSON 404s for unknown paths."""
    frontend_root = FRONTEND.resolve()
    candidate = (FRONTEND / filename).resolve()
    if candidate.is_relative_to(frontend_root) and candidate.is_file():
        return send_from_directory(FRONTEND, filename)
    return jsonify(error="not found"), 404


@app.get("/api/meta")
def meta():
    return jsonify(stations=STATIONS, min_year=MIN_YEAR, max_year=MAX_YEAR, features=FEATURES,
                   allowed_k=list(ALLOWED_K), distances=list(ALLOWED_DISTANCES))


@app.get("/api/analyze")
def analyze():
    try:
        start, end, ks, distance = parse_args()
        return jsonify(run(start, end, ks, distance))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.errorhandler(404)
def not_found(_):
    return jsonify(error="not found"), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

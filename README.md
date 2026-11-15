# Weather-Station Behaviour Pattern Grouping for Data Quality Assessment

**Problem Statement ID:** SIH26069 · **Course:** INT396 CA Project · **Section:** K3P24RD · **Course outcomes:** CO2, CO3

| Member | Roll No. | Registration No. |
|---|---|---|
| Shivam Misser | RK3P24RDB40 | 12420021 |
| Rishit Bajaj | RK3P24RDB55 | 12409400 |

## 1. Project overview

An unsupervised-learning project that groups **station-month weather profiles** built from NOAA GHCN-Daily observations (8 U.S. stations, 2019–2024, 576 profiles). It prepares daily data, builds interpretable monthly features, compares **Euclidean K-Means** and **Manhattan k-medians** for k = 2–5, compares **hierarchical clustering** (Ward, complete, average, single linkage), visualises with **PCA**, and validates with **WCSS, Silhouette, Davies–Bouldin and subsample stability (adjusted Rand index)**. The analysis is served by a **Flask REST API (backend)** and used by a **separate HTML/CSS/JavaScript web frontend**, which can be deployed independently.

Headline results (seed 42): both distances select **k = 2** (warm vs cool station-months). Euclidean K-Means: Silhouette 0.290, Davies–Bouldin 1.405, WCSS 2890.8, stability ARI 0.948. Manhattan k-medians: Silhouette 0.327, Davies–Bouldin 1.437, L1 objective 2253.4, stability ARI 0.941. The groups reflect climate/season, **not** faulty stations; see the report for limitations and responsible use.

## 2. Prerequisites

* Python **3.10 or later**
* No internet needed for the default run (a processed snapshot is bundled). Internet is needed only for `--download` or for years outside 2019–2024.

## 3. Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Execution

```bash
# Start the frontend and backend together (Linux/macOS)
./run.sh
# Open the URL printed by the script; press Ctrl+C to stop the server

# 1) Full analysis (uses data/processed/station_month_profiles.csv; ~3 s)
python src/run_analysis.py                       # defaults: 2019-2024, k = 2 3 4 5, both distances
python src/run_analysis.py --distance manhattan --clusters 2 3 4
python src/run_analysis.py --download            # re-download NOAA files (internet required)

# 2) Web application (frontend and backend are separate)
# Terminal 1 — Flask REST API:
python backend/app.py                            # API at http://localhost:5000
# Terminal 2 — static frontend:
python -m http.server 8000 --directory frontend   # UI at http://localhost:8000

# Demo shortcut: Flask can also serve the independent frontend folder
# python backend/app.py                            # open http://localhost:5000

# 3) Unit + API tests
python -m unittest discover tests

# 4) Rebuild the report and slides from outputs/
python src/build_submission.py
```

Outputs are written to `outputs/`: `validation_metrics.csv` (all configurations incl. hierarchical linkages), `station_month_clusters[_manhattan].csv`, `pca_clusters*.png`, `cluster_profiles*.png`, `station_composition*.png`, `dendrogram*.png`, `metrics_by_k.png`, `pca_loadings*.csv`, `cluster_station_counts*.csv` and `run_summary.txt`.

## 5. Data and setup

* **Source:** NOAA National Centers for Environmental Information, GHCN-Daily station `.dly` files  
  Product page: <https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily>  
  Format readme: <https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt>  
  File pattern: `https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/<STATION_ID>.dly`
* **Stations:** USW00094728 New York Central Park, USW00023174 Los Angeles, USW00012921 Houston, USW00024233 Seattle, USW00013874 Atlanta, USW00014732 Boston, USW00003927 Dallas, USW00012839 Miami.
* **Setup:** nothing to download for the default run. With `--download` the raw files are saved to `data/raw/` (git-ignored). The bundled `data/processed/station_month_profiles.csv` is the frozen snapshot used for the reported results; its SHA-256 is printed in `outputs/run_summary.txt`.
* **Features:** mean TMAX, mean TMIN, mean diurnal range, log(1+precipitation), observed-element coverage, month sin/cos. NOAA missing values (-9999) and quality-flagged readings are removed; missing feature cells are median-imputed; features are standardised; station identity is not a model feature.

## 6. Method notes

* Manhattan distance is implemented as **k-medians** (L1 objective), so its objective is not comparable with Euclidean WCSS.
* Model selection: exclude groups < 5% of profiles → keep Silhouette within 0.02 of the best → prefer higher subsample stability → lower Davies–Bouldin.
* PCA is used for two-dimensional display only; clustering runs in the full standardised feature space.

## 7. Project structure

```text
backend/app.py             Flask REST API (/api/health, /api/meta, /api/analyze)
frontend/                  static web UI (index.html, app.js, style.css, config.js)
render.yaml                Render blueprint for the backend
src/data_pipeline.py       NOAA download, parsing, feature preparation, snapshot loader
src/clustering.py          K-Means, k-medians, hierarchical comparison, validation, stability
src/run_analysis.py        reproducible CLI entry point and figures
src/build_submission.py    builds docs/Project_Report.pdf and docs/Project_Presentation.pptx
tests/test_pipeline.py     unit tests
data/processed/            bundled station-month profile snapshot
outputs/                   generated results and figures
docs/                      project report (PDF) and presentation (PPTX)
PROJECT_LINKS.txt          repository / deployment links
```

## 8. Deployment (separate backend and frontend)

1. **GitHub:** push this folder to a new public repository.
2. **Backend on Render:** *New + → Blueprint* (uses `render.yaml`) or *New Web Service* with build `pip install -r requirements.txt` and start `gunicorn backend.app:app --bind 0.0.0.0:$PORT --timeout 120`. Test `https://<your-service>.onrender.com/api/health`. Optionally set the env var `ALLOWED_ORIGIN` to your frontend URL.
3. **Frontend on Netlify:** set `window.API_BASE` in `frontend/config.js` to your Render URL, then drag the `frontend/` folder onto Netlify Drop (or connect the repo with publish directory `frontend`).
4. Put both URLs in `PROJECT_LINKS.txt`. The free Render service sleeps when idle, so the first request can take about a minute.

API: `GET /api/analyze?start=2019&end=2024&k=2,3,4,5&distance=euclidean|manhattan` returns metrics, clusters, PCA points and station counts as JSON; invalid parameters return HTTP 400.

## 9. Responsible use and limitations

Eight large-city U.S. stations are a small convenience sample, not representative of global or Indian (IMD) networks. Coverage is ≈100% almost everywhere, so the clusters mostly capture climate and season. Do not use the groups to penalise station operators or to make safety decisions without domain review.

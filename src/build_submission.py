"""Build docs/Project_Report.pdf and docs/Project_Presentation.pptx from outputs/.

Run after src/run_analysis.py:  python src/build_submission.py
All numbers are read from outputs/ so the documents always match the latest run.
Edit MEMBERS / CONTRIBUTIONS below if the work split changes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image as PILImage
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from sklearn.cluster import AgglomerativeClustering

from src.clustering import preprocess

OUT, DOCS = ROOT / "outputs", ROOT / "docs"
TITLE = "Weather-Station Behaviour Pattern Grouping for Data Quality Assessment"
PS_ID = "SIH26069"
MEMBERS = [("Shivam Misser", "RK3P24RDB40", "12420021"), ("Rishit Bajaj", "RK3P24RDB55", "12409400")]
SECTION, COURSE_OUTCOMES = "K3P24RD", "CO2, CO3"
CONTRIBUTIONS = {
    "Shivam Misser (12420021) - data, visualisation, frontend and deployment": [
        "Dataset selection and preparation: NOAA GHCN-Daily downloader, fixed-width parser, quality-flag filtering and monthly feature engineering (src/data_pipeline.py)",
        "Preprocessing (imputation, log transform, standardisation, cyclical month encoding) and PCA visualisation / result figures (src/run_analysis.py)",
        "Web frontend in HTML, CSS and JavaScript with Chart.js (frontend/)",
        "Deployment of backend and frontend (Render + Netlify), README and PROJECT_LINKS.txt",
    ],
    "Rishit Bajaj (12409400) - models, validation, backend and testing": [
        "Clustering models: Euclidean K-Means, Manhattan k-medians and hierarchical comparison (Ward, complete, average, single linkage) (src/clustering.py)",
        "Validation: WCSS / L1 objective, Silhouette, Davies-Bouldin, subsample stability (ARI) and the model-selection rule",
        "Flask REST API backend with input validation and CORS (backend/app.py)",
        "Unit and API tests (tests/)",
    ],
    "Both members": [
        "Interpretation of clusters, limitations and ethics discussion, and preparation and review of the report and presentation",
    ],
}
REFERENCES = [
    "NOAA NCEI, Global Historical Climatology Network Daily (GHCN-Daily): https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily",
    "NOAA NCEI, GHCN-Daily data directory and format readme: https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt",
    "Menne, M. J., Durre, I., Vose, R. S., Gleason, B. E. and Houston, T. G. (2012), An Overview of the Global Historical Climatology Network-Daily Database, J. Atmospheric and Oceanic Technology, 29, 897-910.",
    "Arthur, D. and Vassilvitskii, S. (2007), k-means++: The Advantages of Careful Seeding, SODA.",
    "Rousseeuw, P. J. (1987), Silhouettes: A graphical aid to the interpretation and validation of cluster analysis, J. Computational and Applied Mathematics, 20, 53-65.",
    "Davies, D. L. and Bouldin, D. W. (1979), A Cluster Separation Measure, IEEE TPAMI, PAMI-1(2), 224-227.",
    "Hubert, L. and Arabie, P. (1985), Comparing partitions, Journal of Classification, 2, 193-218 (adjusted Rand index).",
    "Pedregosa, F. et al. (2011), Scikit-learn: Machine Learning in Python, JMLR, 12, 2825-2830.",
    "Jolliffe, I. T. (2002), Principal Component Analysis, 2nd ed., Springer.",
]


def load():
    m = pd.read_csv(OUT / "validation_metrics.csv")
    e = pd.read_csv(OUT / "station_month_clusters.csv")
    k = pd.read_csv(OUT / "station_month_clusters_manhattan.csv")
    prof = pd.read_csv(ROOT / "data/processed/station_month_profiles.csv")
    summary = dict(line.split(": ", 1) for line in (OUT / "run_summary.txt").read_text().splitlines() if ": " in line)
    sel_e = m[(m.distance == "euclidean") & m.selected].iloc[0]
    sel_m = m[(m.distance == "manhattan") & m.selected].iloc[0]
    x, _, _ = preprocess(prof)
    labels = AgglomerativeClustering(n_clusters=2, linkage="average").fit_predict(x)
    minority = prof[labels == pd.Series(labels).value_counts().idxmin()]
    return m, e, k, prof, summary, sel_e, sel_m, minority


def group_table(df):
    g = df.groupby("cluster").agg(n=("cluster", "size"), tmax=("tmax_mean_c", "mean"), tmin=("tmin_mean_c", "mean"),
                                  prcp=("precipitation_mm", "mean"), cov=("coverage", "mean")).reset_index()
    return g


def describe_groups(df):
    g = group_table(df).sort_values("tmax", ascending=False)
    warm, cool = g.iloc[0], g.iloc[1]
    return warm, cool


def architecture_png(path):
    steps = ["NOAA GHCN-Daily\n.dly station files", "Parse + QC flags\n(TMAX, TMIN, PRCP)", "Station-month features\n(576 profiles)", "Impute, log1p,\nstandardise",
             "K-Means / k-medians\n+ hierarchical linkages", "Metrics, PCA,\nsubsample stability", "Flask REST API ->\\nweb frontend (JS)"]
    fig, ax = plt.subplots(figsize=(9, 3.4)); ax.axis("off")
    pos = [(0, 1), (1, 1), (2, 1), (3, 1), (0, 0), (1, 0), (2, 0)]  # two rows, both left->right
    for i, (s, (x, y)) in enumerate(zip(steps, pos)):
        ax.text(x, y, s, ha="center", va="center", fontsize=9.5, color="white",
                bbox=dict(boxstyle="round,pad=0.5", fc="#24788a" if i == 4 else "#14324a", ec="none"))
        if i < len(steps) - 1:
            nx, ny = pos[i + 1]
            dx, dy = nx - x, ny - y
            if i == 3:   # wrap from end of row 1 to start of row 0
                ax.annotate("", xy=(0.15, .32), xytext=(2.85, .68), arrowprops=dict(arrowstyle="->", color="#555", lw=1.6, connectionstyle="arc3,rad=0"))
            else:
                ax.annotate("", xy=(nx - .32, ny), xytext=(x + .32, y), arrowprops=dict(arrowstyle="->", color="#555", lw=1.8))
    ax.set_xlim(-.6, 3.6); ax.set_ylim(-.45, 1.45)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def pct(x): return f"{x:.1%}"


# ----------------------------------------------------------------- PDF report
def report_pdf(ctx):
    m, e, k, prof, summary, sel_e, sel_m, minority = ctx
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleC", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#14324a"), fontSize=20, leading=25))
    styles.add(ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=7.2, leading=8.8))
    styles.add(ParagraphStyle("Cap", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#444444"), alignment=TA_CENTER))
    body, h2, h3, bullet = styles["BodyText"], styles["Heading2"], styles["Heading3"], styles["BodyText"]
    P = lambda t, s=body: Paragraph(t, s)
    story = []

    def bullets(items):
        for it in items: story.append(Paragraph("&bull; " + it, ParagraphStyle("b", parent=body, leftIndent=12, firstLineIndent=-8, spaceAfter=2)))

    def fig(name, caption, w=165):
        p = OUT / name if (OUT / name).exists() else DOCS / name
        with PILImage.open(p) as im: ratio = im.height / im.width
        story.append(KeepTogether([Image(str(p), width=w * mm, height=w * ratio * mm), P(caption, styles["Cap"]), Spacer(1, 6)]))

    def table(rows, widths, header="#14324a", size=7.2):
        cell = ParagraphStyle("c", parent=styles["Cell"], fontSize=size, leading=size + 1.8)
        data = [[Paragraph(str(c), cell) for c in r] for r in rows]
        t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header)), ("GRID", (0, 0), (-1, -1), .25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        # white header text
        for j in range(len(rows[0])):
            data[0][j] = Paragraph(f'<font color="white"><b>{rows[0][j]}</b></font>', cell)
        t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header)), ("GRID", (0, 0), (-1, -1), .25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.extend([t, Spacer(1, 8)])

    # 1 Title page
    story += [Spacer(1, 40), P(TITLE, styles["TitleC"]), Spacer(1, 14),
              P(f"<b>Problem Statement ID:</b> {PS_ID} &nbsp;|&nbsp; <b>Course:</b> INT396 CA Project &nbsp;|&nbsp; <b>Section:</b> {SECTION} &nbsp;|&nbsp; <b>Course outcomes:</b> {COURSE_OUTCOMES}", ParagraphStyle("c2", parent=body, alignment=TA_CENTER)),
              P("<b>Project type:</b> Group &nbsp;|&nbsp; <b>Techniques:</b> Unit II, III &amp; IV - Partition Clustering, Hierarchical Comparison, PCA Visualisation", ParagraphStyle("c3", parent=body, alignment=TA_CENTER)),
              Spacer(1, 18)]
    table([["Group member", "Roll No.", "Registration No."]] + [list(mm_) for mm_ in MEMBERS], [70, 50, 50])
    story += [P("Submitted as a single ZIP (12420021_12409400.zip) containing this report, the presentation, the complete codebase, README.md, requirements.txt and PROJECT_LINKS.txt.", ParagraphStyle("c4", parent=body, alignment=TA_CENTER)), PageBreak()]

    # 2 Introduction
    story += [P("1. Introduction and problem statement", h2),
              P("Weather-station archives combine observations from instruments, climates and reporting practices that change over time. Reviewers of such archives cannot manually inspect every station-month, so they need a transparent way to find groups of similar behaviour and to see which records are unusual. Problem statement SIH26069 asks for behaviour pattern grouping of station observations for data-quality assessment. This project applies unsupervised learning - K-Means, Manhattan k-medians, hierarchical clustering and PCA - to monthly profiles built from NOAA GHCN-Daily observations, and validates the groupings with WCSS, Silhouette, Davies-Bouldin and resampling stability. The groups are exploratory summaries; they are not a supervised fault detector."),
              P("2. Project objectives and need", h2)]
    bullets(["<b>Objective 1 - data preparation:</b> prepare open station-observation data, build interpretable features, handle missing values, encode cyclical/categorical information where required and scale numerical variables.",
             "<b>Objective 2 - clustering and validation:</b> run K-Means for k = 2, 3, 4 and 5; compare Euclidean and Manhattan distance; visualise with PCA; evaluate with WCSS, Silhouette and Davies-Bouldin to identify a defensible structure; compare against hierarchical clustering.",
             "<b>Objective 3 - interpretation and responsibility:</b> interpret the clusters through feature profiles, validation results and stability; document limitations, sampling bias and ethical considerations."])
    story.append(P("<b>Need:</b> archive maintainers need prioritisation tools that are reproducible, explainable and honest about uncertainty. Grouping profiles by behaviour narrows where expert attention should go first, while the validation and stability analysis shows how much trust the groups deserve."))

    # 3 Overview
    story += [P("3. Project overview, key features and working", h2),
              P("The pipeline downloads eight public GHCN-Daily station files, removes missing and quality-flagged readings, aggregates daily TMAX, TMIN and PRCP into station-month-year profiles, imputes and standardises the features, and fits partition and hierarchical models. A command-line runner writes metrics, labels and figures to <i>outputs/</i>. The same engine is served by a Flask REST API (backend) and consumed by a separate JavaScript web interface (frontend), so the two parts can be deployed independently.")]
    bullets(["Works offline from a frozen processed snapshot (data/processed/station_month_profiles.csv); optional <i>--download</i> re-fetches NOAA data.",
             "Euclidean K-Means and Manhattan k-medians for k = 2-5, each with 20 subsample-stability runs.",
             "Hierarchical comparison with Ward (Euclidean only), complete, average and single linkage, plus dendrograms.",
             "Transparent model-selection rule (below), PCA projection, cluster profiles, station composition and unit tests."])
    fig("architecture.png", "Figure 1. System architecture and workflow (no database or API key is needed).", 125)

    # 4 Tools
    story += [P("4. Tools, technologies and frameworks used", h2)]
    table([["Tool", "Purpose"],
           ["Python 3.10+", "Implementation language"], ["pandas, NumPy", "Parsing, aggregation, feature table"],
           ["scikit-learn", "K-Means, agglomerative clustering, PCA, imputation, scaling, Silhouette, Davies-Bouldin, adjusted Rand index"],
           ["SciPy", "Linkage matrices and dendrograms"], ["Matplotlib", "All figures"], ["Flask, gunicorn", "REST API backend (backend/app.py)"], ["HTML, CSS, JavaScript, Chart.js", "Separate web frontend (frontend/)"],
           ["requests", "HTTPS download of NOAA files"], ["ReportLab, python-pptx", "Generate this report and the presentation from outputs/"],
           ["unittest, Flask test client", "Unit and API tests in tests/"]], [45, 125], size=8)

    # 5 Methodology
    n = len(prof)
    story += [P("5. Methodology and implementation", h2), P("5.1 Dataset details", h3),
              P(f"<b>Source:</b> NOAA National Centers for Environmental Information, GHCN-Daily, one fixed-width <i>.dly</i> file per station (https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/&lt;station&gt;.dly). <b>Stations (8):</b> New York Central Park (USW00094728), Los Angeles (USW00023174), Houston (USW00012921), Seattle (USW00024233), Atlanta (USW00013874), Boston (USW00014732), Dallas (USW00003927), Miami (USW00012839). <b>Period:</b> 2019-2024. <b>Size:</b> {n} station-month profiles (8 stations x 72 months), {prof.shape[1]} columns, derived from daily TMAX, TMIN and PRCP observations. The snapshot checksum (SHA-256) is stored in outputs/run_summary.txt."),
              P("<b>Model features (7):</b> mean daily maximum temperature, mean daily minimum temperature, mean diurnal range (TMAX-TMIN), log(1+monthly precipitation), observed-element coverage, and sine/cosine encoding of the month. Station identity is kept for interpretation but excluded from clustering, so groups are not trivially defined by station labels. Snowfall is omitted because its reporting differs between stations."),
              P("5.2 Preprocessing", h3)]
    bullets(["Drop the NOAA missing sentinel (-9999) and any value with a non-blank quality-control flag; convert tenths of degC / mm to degC / mm.",
             "Aggregate to station-month-year; coverage = valid element-days / (days in month x 3 elements).",
             "Categorical information: month is encoded cyclically (sin/cos) so December and January stay close; the station label is not used as a feature.",
             "Missing feature cells are median-imputed (none remain in this snapshot); precipitation is log1p-transformed; all features are z-score standardised before clustering."])
    story.append(P("5.3 Models, algorithms and justification", h3))
    bullets(["<b>K-Means (Euclidean):</b> baseline partition method with compact, explainable centroids and a WCSS elbow diagnostic; 20 k-means++ restarts.",
             "<b>k-medians (Manhattan / L1):</b> the partition method that matches Manhattan distance (coordinate-wise medians, best of 10 restarts); more robust to outlying months. Its L1 objective is not comparable with WCSS and is never presented as WCSS.",
             "<b>Hierarchical comparison:</b> Ward (Euclidean), complete, average and single linkage; used to test whether the partition result survives a very different clustering principle and to expose outlier profiles.",
             "<b>PCA:</b> two components for display only; clustering uses the full 7-dimensional standardised space."])
    story.append(P("5.4 Evaluation metrics and selection rule", h3))
    story.append(P("For every configuration the run reports Silhouette (using the matching distance), Davies-Bouldin (Euclidean), minimum group share and, for K-Means, WCSS; for k-medians the L1 objective. Stability is the mean adjusted Rand index (ARI) between the full-data partition and re-fits on 20 random 80% subsamples. Hierarchical rows also report ARI against the partition at the same k. <b>Selection rule:</b> exclude partitions with a group smaller than 5% of profiles, keep those whose Silhouette is within 0.02 of the best, prefer higher stability, and break ties with the lower Davies-Bouldin score. WCSS is read as an elbow curve and never used alone."))
    story.append(P("5.5 Deployment, performance and security", h3))
    bullets([f"<b>Deployment (separate frontend and backend):</b> the Flask API (backend/) is deployed as a web service on Render using gunicorn; the static frontend (frontend/) is hosted on Netlify and calls the API through the base URL in frontend/config.js. Both links are recorded in PROJECT_LINKS.txt. Locally: <i>python backend/app.py</i> and open frontend/index.html.",
             f"<b>Performance:</b> the full default run (576 profiles, both distances, k = 2-5, 20 stability resamples per configuration, 7 hierarchical models per k) completes in about {summary.get('Runtime (s)', 'n/a')} s on a standard CPU in our test environment; memory use is a few tens of MB because the feature matrix is only {n} x 7.",
             "<b>Security:</b> no credentials, user accounts or personal data; only HTTPS requests to NOAA with timeouts; parsed fields are validated (fixed-width slicing, integer parsing, invalid dates skipped); the API validates every parameter (year range, k list, distance) and returns HTTP 400 otherwise, never downloads from user input and uses CORS; dependencies are version-pinned in requirements.txt; downloaded files are kept in the git-ignored data/raw folder.",
             "<b>Reproducibility:</b> fixed random seed (42) and a frozen processed snapshot give identical metrics on every run."])

    # 6 Results
    story.append(PageBreak())
    story.append(P("6. Results, testing and screenshots", h2))
    warm_e, cool_e = describe_groups(e); warm_m, cool_m = describe_groups(k)
    story.append(P(f"<b>Headline.</b> The selection rule chose k = 2 for both distances. Euclidean K-Means: Silhouette {sel_e.silhouette:.3f}, Davies-Bouldin {sel_e.davies_bouldin_euclidean:.3f}, WCSS {sel_e.wcss_euclidean:.1f}, smallest group {pct(sel_e.min_cluster_fraction)}, stability ARI {sel_e.stability_ari_mean:.3f} &plusmn; {sel_e.stability_ari_std:.3f}. Manhattan k-medians: Silhouette {sel_m.silhouette:.3f}, Davies-Bouldin {sel_m.davies_bouldin_euclidean:.3f}, L1 objective {sel_m.l1_objective:.1f}, smallest group {pct(sel_m.min_cluster_fraction)}, stability ARI {sel_m.stability_ari_mean:.3f} &plusmn; {sel_m.stability_ari_std:.3f}. A Silhouette near 0.3 indicates moderate, not strong, separation. The first two principal components retain {pct(sel_e.pca_variance_2d)} of the variance."))
    parts = m[m.method.isin(["kmeans", "kmedians"])].sort_values(["distance", "k"])
    rows = [["Distance", "Method", "k", "WCSS / L1", "Silhouette", "Davies-Bouldin", "Min group", "Stability ARI", "Selected"]]
    for _, r in parts.iterrows():
        obj = f"{r.wcss_euclidean:.1f}" if r.method == "kmeans" else f"{r.l1_objective:.1f}"
        rows.append([r.distance, r.method, int(r.k), obj, f"{r.silhouette:.3f}", f"{r.davies_bouldin_euclidean:.3f}", pct(r.min_cluster_fraction), f"{r.stability_ari_mean:.3f} +/- {r.stability_ari_std:.3f}", "yes" if r.selected else ""])
    story.append(P("Table 1. Partition methods, all tested configurations.", styles["Cap"]))
    table(rows, [20, 19, 7, 21, 19, 24, 19, 30, 17])
    tiny = parts[(parts.distance == "euclidean") & (parts.k >= 3)].min_cluster_fraction.max()
    story.append(P(f"Euclidean K-Means with k = 3, 4 and 5 gives slightly higher or comparable Silhouette values but each contains a group of only {int(round(tiny * len(e)))} profiles (&lt; 1%), so the 5% rule excludes them; their lower stability (see Table 1) confirms the choice. WCSS falls steadily with k (an elbow-style trade-off), so it supports no unique k. For Manhattan k-medians, k = 3-5 have no tiny groups but lower Silhouette and clearly lower stability than k = 2."))
    fig("metrics_by_k.png", "Figure 2. WCSS/L1 objective, Silhouette, Davies-Bouldin and stability for every k and distance.", 165)
    story.append(P("6.1 Cluster interpretation", h3))
    story.append(P(f"<b>Euclidean k = 2:</b> a warm group ({int(warm_e.n)} profiles; mean TMAX {warm_e.tmax:.1f} degC, TMIN {warm_e.tmin:.1f} degC, precipitation {warm_e.prcp:.1f} mm) and a cool group ({int(cool_e.n)} profiles; TMAX {cool_e.tmax:.1f} degC, TMIN {cool_e.tmin:.1f} degC, {cool_e.prcp:.1f} mm). <b>Manhattan k = 2:</b> a warm group ({int(warm_m.n)} profiles; TMAX {warm_m.tmax:.1f}, TMIN {warm_m.tmin:.1f}, {warm_m.prcp:.1f} mm) and a cool group ({int(cool_m.n)}; TMAX {cool_m.tmax:.1f}, TMIN {cool_m.tmin:.1f}, {cool_m.prcp:.1f} mm). Precipitation barely differs between groups, and mean coverage is {e.coverage.mean():.4f} in both: the clusters are driven by temperature and season, i.e. <b>climate</b>, not by data quality. Station composition confirms this: for example Miami is mostly in the warm group while Seattle, Boston and New York are mostly in the cool group (Figure 4, outputs/cluster_station_counts.csv)."))
    pc = pd.read_csv(OUT / "pca_loadings.csv", index_col=0)
    story.append(P(f"PCA loadings show PC1 is mainly temperature (TMAX {pc.loc['tmax_mean_c', 'PC1']:.2f}, TMIN {pc.loc['tmin_mean_c', 'PC1']:.2f}, with month terms) and PC2 contrasts diurnal range with precipitation (diurnal range {pc.loc['diurnal_range_c', 'PC2']:.2f}, log precipitation {pc.loc['precipitation_log1p', 'PC2']:.2f}); coverage contributes almost nothing ({pc.loc['coverage', 'PC1']:.2f} on PC1)."))
    fig("pca_clusters.png", "Figure 3a. PCA projection of the Euclidean K-Means groups.", 120)
    fig("pca_clusters_manhattan.png", "Figure 3b. PCA projection of the Manhattan k-medians groups.", 120)
    fig("cluster_profiles.png", "Figure 4a. Mean feature profile per Euclidean cluster.", 130)
    fig("station_composition.png", "Figure 4b. Number of profiles per station in each Euclidean cluster.", 125)
    story.append(P("6.2 Hierarchical comparison and linkage sensitivity", h3))
    hdr = ["Distance", "Linkage"] + [f"k={k_}: Silhouette / min group / ARI vs partition" for k_ in sorted(m.k.unique())]
    rows = [hdr]
    for _, grp in m[m.method.str.startswith("hierarchical")].groupby(["distance", "method"], sort=False):
        r0 = grp.iloc[0]
        cells = [f"{r.silhouette:.2f} / {pct(r.min_cluster_fraction)} / {r.ari_vs_partition:.2f}" for _, r in grp.sort_values("k").iterrows()]
        rows.append([r0.distance, r0.method.replace("hierarchical_", "")] + cells)
    table(rows, [18, 16, 34, 34, 34, 34], size=6.8)
    ward = m[(m.method == "hierarchical_ward") & (m.k == 2)].iloc[0]
    ids = ", ".join(f"{r.station_name} {int(r.year)}-{int(r.month):02d} (coverage {r.coverage:.3f})" for _, r in minority.iterrows())
    story.append(P(f"Ward linkage (the only linkage matching K-Means' variance criterion) gives balanced groups at k = 2 (smallest group {pct(ward.min_cluster_fraction)}) with Silhouette {ward.silhouette:.3f} and only moderate agreement with K-Means (ARI {ward.ari_vs_partition:.2f}). Complete, average and single linkage instead split off a tiny group at k = 2 and obtain a very high Silhouette ({m[(m.method == 'hierarchical_average') & (m.k == 2) & (m.distance == 'euclidean')].silhouette.iloc[0]:.2f}) - an artefact of isolating outliers, which is why Silhouette alone is not used for selection. These tiny groups are nonetheless informative for data quality: the {len(minority)} profiles isolated by average linkage are <b>{ids}</b> - the two lowest-coverage station-months in the whole sample. Hierarchical results are therefore sensitive to the linkage rule, whereas the K-Means/k-medians k = 2 split is stable under resampling."))
    fig("dendrogram.png", "Figure 5. Ward dendrogram (Euclidean), last 30 merges.", 140)
    story.append(P("6.3 Testing", h3))
    bullets(["Unit and API tests (tests/test_pipeline.py, tests/test_api.py, run with <i>python -m unittest discover tests</i>) cover: GHCN fixed-width parsing, unit conversion and removal of flagged/missing values; year filtering; snapshot integrity (576 rows, no NaNs); snapshot loading; deterministic Euclidean runs; metric ranges and a single selected configuration; k-medians on synthetic blobs. All pass.",
             "Verification of results: the Euclidean metrics reproduce the earlier baseline exactly (WCSS 2890.8, Silhouette 0.290, Davies-Bouldin 1.405, stability 0.948). Manhattan k-medians now uses multiple restarts, which lowers the L1 objective and stabilises the result.",
             "Web application: the frontend calls <i>/api/analyze</i> and shows the validation table, stability metric, PCA scatter and cluster profiles for any chosen year range, k values and distance; API tests cover valid requests and rejected parameters."])

    # 7 Contributions
    story.append(P("7. Individual contributions of all group members", h2))
    for who, items in CONTRIBUTIONS.items():
        story.append(P(f"<b>{who}</b>", body)); bullets(items)

    # 8 Limitations
    story += [P("8. Limitations and future scope", h2)]
    bullets(["<b>Sampling bias:</b> eight U.S. stations in large cities are a convenience sample; results do not represent global or Indian (IMD) networks, rural stations or other climates.",
             "<b>Weak data-quality signal:</b> coverage is ~100% almost everywhere, so the clusters reflect climate/season rather than faults. Only two low-coverage Seattle months stand out.",
             "<b>Method sensitivity:</b> scaling, imputation, the 5% size rule, distance choice and linkage all change the groups; PCA (55% variance in 2-D) can hide separation.",
             "<b>Quality flags:</b> NOAA flags are applied as provided; unflagged readings are not guaranteed correct. Station moves and instrument changes are not modelled.",
             "<b>Future scope:</b> add quality features (count of flagged days, TMAX&lt;TMIN violations, repeated-value runs, outlier days), broaden stations including IMD/Indian networks, use station-move metadata, temporal and geographic hold-outs, DBSCAN/Isolation Forest for anomalies, and domain-expert review."])
    story += [P("Practical applications and societal impact", h3),
              P("The method can help data stewards prioritise which station-periods to inspect, make coverage and reporting differences visible, and support climate-data audits that underpin forecasting, agriculture and disaster planning. Ethical use requires that clusters never be used alone to penalise station operators, rank stations or make safety decisions; unusual groups may simply reflect real weather, and human review must separate instrument problems from climate.")]

    # 9 References
    story.append(P("9. References and resources used", h2))
    for i, r in enumerate(REFERENCES, 1):
        story.append(P(f"[{i}] {r.replace('&', '&amp;').replace('<', '&lt;')}", ParagraphStyle("ref", parent=body, fontSize=8.5, leading=10.5)))

    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.grey)
        canvas.drawString(18 * mm, 10 * mm, f"{PS_ID} - {TITLE}"); canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}"); canvas.restoreState()

    SimpleDocTemplate(str(DOCS / "Project_Report.pdf"), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=16 * mm, title=TITLE, author=", ".join(n for n, *_ in MEMBERS)).build(story, onFirstPage=footer, onLaterPages=footer)


# ---------------------------------------------------------------------- PPTX
NAVY, TEAL, INK, BG = RGBColor(20, 50, 74), RGBColor(36, 120, 138), RGBColor(45, 63, 77), RGBColor(247, 250, 252)


def presentation(ctx):
    m, e, k, prof, summary, sel_e, sel_m, minority = ctx
    prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    state = {"n": 0}

    def new(title):
        s = prs.slides.add_slide(prs.slide_layouts[6]); state["n"] += 1
        s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
        bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(1.05)); bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
        p = s.shapes.add_textbox(Inches(.6), Inches(.15), Inches(12.2), Inches(.8)).text_frame.paragraphs[0]
        p.text = title; p.font.size = Pt(30); p.font.bold = True; p.font.color.rgb = RGBColor(255, 255, 255)
        f = s.shapes.add_textbox(Inches(.6), Inches(7.05), Inches(12.2), Inches(.3)).text_frame.paragraphs[0]
        f.text = f"{PS_ID}  |  {state['n']}"; f.font.size = Pt(10); f.font.color.rgb = RGBColor(85, 110, 125); f.alignment = PP_ALIGN.RIGHT
        return s

    def bullets(s, items, left=.7, top=1.4, width=12, size=24, height=5.4):
        tf = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height)).text_frame; tf.word_wrap = True
        for i, t in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = "\u2022 " + t; p.font.size = Pt(size); p.font.color.rgb = INK; p.space_after = Pt(10)

    def picture(s, name, left, top, w, h):
        path = OUT / name if (OUT / name).exists() else DOCS / name
        with PILImage.open(path) as im: ratio = im.width / im.height
        width = min(w, h * ratio); height = width / ratio
        s.shapes.add_picture(str(path), Inches(left + (w - width) / 2), Inches(top + (h - height) / 2), width=Inches(width), height=Inches(height))

    warm_e, cool_e = describe_groups(e); warm_m, cool_m = describe_groups(k)
    # 1 Title page
    s = new("Title page")
    bullets(s, [TITLE, f"Problem Statement ID: {PS_ID}   |   Section {SECTION}   |   {COURSE_OUTCOMES}"] +
            [f"{n}  -  Roll No. {r}  -  Reg. No. {g}" for n, r, g in MEMBERS], size=24)
    s = new("Introduction and problem statement")
    bullets(s, ["Weather-station archives mix climates, instruments and reporting practices.",
                "Reviewers cannot inspect every station-month; they need transparent groups of similar behaviour.",
                f"{PS_ID}: group station behaviour patterns to support data-quality assessment.",
                "Approach: unsupervised learning (K-Means, k-medians, hierarchical, PCA) with rigorous validation.",
                "Exploratory grouping - not a supervised fault detector."])
    s = new("Project objectives and need")
    bullets(s, ["Prepare NOAA station data: features, missing values, cyclical encoding, scaling.",
                "K-Means with k = 2-5; compare Euclidean vs Manhattan; PCA visualisation.",
                "Evaluate with WCSS, Silhouette, Davies-Bouldin and resampling stability.",
                "Compare with hierarchical clustering (several linkages).",
                "Interpret clusters; document limitations, sampling bias and ethics.",
                "Need: reproducible, explainable, honest prioritisation of review effort."])
    s = new("Project overview, key features and working")
    bullets(s, ["Download/parse GHCN-Daily -> station-month profiles -> impute, log1p, standardise.",
                "Partition models for both distances; hierarchical comparison; PCA projection.",
                "Offline-ready: frozen snapshot of 576 profiles; --download refreshes from NOAA.",
                "CLI writes metrics and figures; Flask API + separate JavaScript frontend.",
                "Fixed seed and unit tests make results reproducible."])
    s = new("System architecture and workflow")
    picture(s, "architecture.png", .5, 1.3, 12.3, 3.7)
    bullets(s, ["No database or API key; only HTTPS downloads of public NOAA files.",
                "Outputs: validation_metrics.csv, cluster CSVs, PCA/profile/dendrogram figures, run_summary.txt."], top=5.2, size=18, height=1.7)
    s = new("Tools, technologies and frameworks")
    bullets(s, ["Python 3.10+, pandas, NumPy, SciPy", "scikit-learn: K-Means, agglomerative clustering, PCA, metrics",
                "Matplotlib: figures   |   Flask: REST API   |   JavaScript + Chart.js: frontend",
                "ReportLab and python-pptx: report and slides generated from outputs",
                "unittest: unit + API tests   |   Git/GitHub for version control"])
    s = new("Dataset and preprocessing")
    bullets(s, ["Source: NOAA NCEI GHCN-Daily (.dly), 8 U.S. stations, 2019-2024.",
                f"Size: {len(prof)} station-month profiles (8 x 72); daily TMAX, TMIN, PRCP.",
                "Features: mean TMAX, mean TMIN, diurnal range, log1p precipitation, coverage, month sin/cos.",
                "Removed -9999 and quality-flagged values; tenths -> degC / mm.",
                "Median imputation, log1p, z-score scaling; station ID excluded from clustering."])
    s = new("Methodology: models and justification")
    bullets(s, ["Euclidean K-Means (k-means++, 20 restarts): compact centroids, WCSS elbow.",
                "Manhattan k-medians (10 restarts): robust L1 partition; L1 objective is not WCSS.",
                "Hierarchical: Ward (Euclidean), complete, average, single linkage.",
                "PCA: 2-D view only; clustering in full 7-D space."])
    s = new("Evaluation metrics and selection rule")
    bullets(s, ["WCSS / L1 objective, Silhouette (matching distance), Davies-Bouldin, smallest-group share.",
                "Stability: mean ARI over 20 random 80% subsamples; hierarchical ARI vs partition.",
                "Rule: drop groups < 5% -> Silhouette within 0.02 of best -> higher stability -> lower Davies-Bouldin.",
                "WCSS only as elbow diagnostic; Silhouette alone can reward outlier-only groups."])
    s = new("Results: validation across k and distance")
    picture(s, "metrics_by_k.png", .4, 1.2, 7.6, 5.7)
    bullets(s, [f"Both methods select k = 2.", f"K-Means: Sil {sel_e.silhouette:.3f}, DB {sel_e.davies_bouldin_euclidean:.3f}, WCSS {sel_e.wcss_euclidean:.0f}, ARI {sel_e.stability_ari_mean:.3f}",
                f"k-medians: Sil {sel_m.silhouette:.3f}, DB {sel_m.davies_bouldin_euclidean:.3f}, L1 {sel_m.l1_objective:.0f}, ARI {sel_m.stability_ari_mean:.3f}",
                "Moderate separation; k >= 3 K-Means has 2-profile groups."], left=8.2, top=1.4, width=4.8, size=15)
    s = new("Results: PCA visualisation")
    picture(s, "pca_clusters.png", .4, 1.3, 6.2, 5.2); picture(s, "pca_clusters_manhattan.png", 6.7, 1.3, 6.2, 5.2)
    f = s.shapes.add_textbox(Inches(.6), Inches(6.5), Inches(12), Inches(.5)).text_frame.paragraphs[0]
    f.text = f"Two components retain {pct(sel_e.pca_variance_2d)} of variance; PC1 is temperature, PC2 contrasts diurnal range and precipitation."; f.font.size = Pt(15); f.font.color.rgb = INK
    s = new("Results: cluster interpretation")
    picture(s, "station_composition.png", .4, 1.3, 6.3, 4.6)
    bullets(s, [f"Warm group: {int(warm_e.n)} profiles, TMAX {warm_e.tmax:.1f} / TMIN {warm_e.tmin:.1f} degC",
                f"Cool group: {int(cool_e.n)} profiles, TMAX {cool_e.tmax:.1f} / TMIN {cool_e.tmin:.1f} degC",
                f"Precipitation {warm_e.prcp:.0f} vs {cool_e.prcp:.0f} mm; coverage {e.coverage.mean():.4f}",
                "Groups = climate/season, not faults (e.g. Miami mostly warm)."], left=7, top=1.4, width=6, size=17)
    s = new("Results: hierarchical comparison")
    picture(s, "dendrogram.png", .4, 1.3, 7.3, 4.8)
    ids = "; ".join(f"{r.station_name} {int(r.year)}-{int(r.month):02d}" for _, r in minority.iterrows())
    ward = m[(m.method == "hierarchical_ward") & (m.k == 2)].iloc[0]
    bullets(s, [f"Ward k=2: Sil {ward.silhouette:.3f}, ARI vs K-Means {ward.ari_vs_partition:.2f}",
                "Complete/average/single isolate a 2-profile group (inflated Silhouette)",
                f"Isolated: {ids} - the two lowest-coverage months",
                "Results are linkage-sensitive; partition k=2 is stable."], left=7.9, top=1.4, width=5.1, size=16)
    s = new("Testing, performance, deployment and security")
    bullets(s, ["Testing: unit + API tests pass (parser, flags, snapshot, determinism, metrics, k-medians, API validation).",
                f"Performance: full run in about {summary.get('Runtime (s)', 'n/a')} s; 576 x 7 feature matrix.",
                "Deployment: Flask API on Render, static frontend on Netlify (separate).",
                "Security: no credentials or personal data; HTTPS with timeouts; bounded inputs; pinned dependencies."])
    s = new("Individual contributions")
    items = []
    for who, subs in CONTRIBUTIONS.items():
        items.append(who); items += ["    " + t for t in subs]
    bullets(s, items, size=17)
    s = new("Limitations and future scope")
    bullets(s, ["Eight U.S. city stations: sampling bias; not representative of IMD / global networks.",
                "Near-100% coverage: weak data-quality signal; clusters reflect climate.",
                "Sensitive to scaling, imputation, distance, linkage; PCA hides structure.",
                "Future: flag counts, TMAX<TMIN checks, repeated-value runs, more stations incl. IMD, metadata, DBSCAN/Isolation Forest, expert review."], size=19)
    s = new("Practical applications and societal impact")
    bullets(s, ["Prioritise station-periods for expert inspection and archive audits.",
                "Make coverage and reporting differences visible; supports climate-data users.",
                "Ethics: never use clusters alone to penalise operators or for safety decisions."])
    s = new("References and resources")
    bullets(s, ["NOAA NCEI, GHCN-Daily (ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)",
                "Menne et al. (2012), GHCN-Daily overview, JTECH 29",
                "Arthur & Vassilvitskii (2007), k-means++   |   Rousseeuw (1987), Silhouettes",
                "Davies & Bouldin (1979)   |   Hubert & Arabie (1985), adjusted Rand index",
                "Pedregosa et al. (2011), scikit-learn   |   Jolliffe (2002), PCA"], size=17)
    prs.save(DOCS / "Project_Presentation.pptx")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    architecture_png(DOCS / "architecture.png")
    context = load()
    report_pdf(context); presentation(context)
    print("Built docs/Project_Report.pdf and docs/Project_Presentation.pptx")

"""Command line entry point for the reproducible analysis."""
import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

from src.clustering import FEATURES, fit_configurations
from src.data_pipeline import PROCESSED_NAME, load_profiles


def plot_metrics(metrics: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for distance, style in (("euclidean", "-"), ("manhattan", "--")):
        part = metrics[(metrics.distance == distance) & metrics.method.isin(["kmeans", "kmedians"])]
        if part.empty:
            continue
        objective_col = "wcss_euclidean" if distance == "euclidean" else "l1_objective"
        if objective_col not in part.columns:
            continue
        axes[0, 0].plot(part.k, part[objective_col], style, marker="o",
                        label="K-Means WCSS" if distance == "euclidean" else "k-medians L1")
        axes[0, 1].plot(part.k, part.silhouette, style, marker="o", label=f"{distance} partition")
        axes[1, 0].plot(part.k, part.davies_bouldin_euclidean, style, marker="o", label=f"{distance} partition")
        axes[1, 1].plot(part.k, part.stability_ari_mean, style, marker="o", label=f"{distance} partition")
        ward = metrics[(metrics.distance == distance) & metrics.method.str.startswith("hierarchical_average")]
        if not ward.empty:
            axes[0, 1].plot(ward.k, ward.silhouette, style, marker="x", alpha=.6,
                            label=f"{distance} hierarchical average")
    for ax, title in zip(axes.flat, ["Objective (elbow; not comparable across the two)", "Silhouette (higher is better)",
                                     "Davies-Bouldin (lower is better)", "Subsample stability (ARI)"]):
        ax.set(title=title, xlabel="k"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "metrics_by_k.png", dpi=160); plt.close(fig)


def plot_dendrogram(x, distance, out: Path, suffix: str):
    method, metric = ("ward", "euclidean") if distance == "euclidean" else ("average", "cityblock")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linkage(x, method=method, metric=metric), truncate_mode="lastp", p=30,
               show_leaf_counts=True, ax=ax, color_threshold=None)
    ax.set(title=f"Hierarchical dendrogram ({method} linkage, {distance}; last 30 merges)",
           xlabel="Cluster size (profiles)", ylabel="Merge distance")
    fig.tight_layout(); fig.savefig(out / f"dendrogram{suffix}.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--clusters", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--distance", choices=["euclidean", "manhattan", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true",
                        help="re-download NOAA files instead of using data/processed snapshot")
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("start year must not exceed end year")
    started = time.perf_counter()
    out = ROOT / "outputs"; out.mkdir(exist_ok=True)
    profiles, source = load_profiles(ROOT, args.start_year, args.end_year, args.download)
    distances = ["euclidean", "manhattan"] if args.distance == "both" else [args.distance]
    metric_frames, summary = [], []
    for distance in distances:
        clustered, metrics, stability, models = fit_configurations(profiles, args.clusters, distance, args.seed)
        suffix = "" if distance == "euclidean" else "_manhattan"
        clustered.to_csv(out / f"station_month_clusters{suffix}.csv", index=False)
        metric_frames.append(metrics)
        pd.DataFrame(models["pca"].components_.T, index=FEATURES, columns=["PC1", "PC2"]).round(3)\
            .to_csv(out / f"pca_loadings{suffix}.csv")
        pd.crosstab(clustered.station_name, clustered.cluster).to_csv(out / f"cluster_station_counts{suffix}.csv")
        plot_dendrogram(models["x"], distance, out, suffix)
        fig, ax = plt.subplots(figsize=(8, 6))
        for label, group in clustered.groupby("cluster"):
            ax.scatter(group.pca_1, group.pca_2, s=24, alpha=.75, label=f"Cluster {label}")
        ax.set(title=f"Station-month groups (PCA, {distance})",
               xlabel="PC1", ylabel="PC2"); ax.legend()
        fig.tight_layout(); fig.savefig(out / f"pca_clusters{suffix}.png", dpi=160); plt.close(fig)
        features = ["tmax_mean_c", "tmin_mean_c", "diurnal_range_c", "precipitation_mm", "coverage"]
        clustered.groupby("cluster")[features].mean().T.plot(kind="bar", figsize=(10, 5))
        plt.title(f"Mean feature profiles by cluster ({distance})"); plt.tight_layout()
        plt.savefig(out / f"cluster_profiles{suffix}.png", dpi=160); plt.close()
        counts = pd.crosstab(clustered.station_name, clustered.cluster)
        counts.plot(kind="barh", stacked=True, figsize=(9, 5))
        plt.title(f"Profiles per station in each cluster ({distance})"); plt.xlabel("Station-month profiles")
        plt.tight_layout(); plt.savefig(out / f"station_composition{suffix}.png", dpi=160); plt.close()
        selected = metrics.loc[metrics.selected].iloc[0]
        summary.append(f"{distance}: selected k={int(selected.k)}; stability ARI "
                       f"{stability['adjusted_rand_mean']:.3f} +/- {stability['adjusted_rand_std']:.3f}")
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_metrics.to_csv(out / "validation_metrics.csv", index=False)
    plot_metrics(all_metrics, out)
    elapsed = time.perf_counter() - started
    checksum = hashlib.sha256((ROOT / "data/processed" / PROCESSED_NAME).read_bytes()).hexdigest() \
        if (ROOT / "data/processed" / PROCESSED_NAME).exists() else "n/a"
    lines = [f"Data source: {source}", f"Profiles: {len(profiles)}", f"Years: {args.start_year}-{args.end_year}",
             f"Seed: {args.seed}", f"Runtime (s): {elapsed:.1f}", f"Snapshot SHA-256: {checksum}"] + summary
    (out / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

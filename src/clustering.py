"""Clustering algorithms and validation helpers.

Partition methods: Euclidean K-Means and Manhattan (L1) k-medians.
Hierarchical comparison: several linkage rules for each distance.
Validation: WCSS / L1 objective, Silhouette, Davies-Bouldin, subsample stability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["tmax_mean_c", "tmin_mean_c", "diurnal_range_c",
            "precipitation_log1p", "coverage", "month_sin", "month_cos"]

# Ward linkage is only defined for Euclidean distance.
LINKAGES = {"euclidean": ["ward", "complete", "average", "single"],
            "manhattan": ["complete", "average", "single"]}


def preprocess(profiles: pd.DataFrame):
    """Median-impute, then z-score the model features."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(profiles[FEATURES]))
    return x, imputer, scaler


def kmedians(x, k, seed=42, max_iter=100, n_init=10):
    """L1 k-medians (Manhattan objective); best of ``n_init`` random starts."""
    rng = np.random.default_rng(seed)
    best_labels, best_objective = None, np.inf
    for _ in range(n_init):
        centers = x[rng.choice(len(x), size=k, replace=False)].copy()
        labels = np.full(len(x), -1)
        for _ in range(max_iter):
            distances = np.abs(x[:, None, :] - centers[None, :, :]).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for j in range(k):
                members = x[labels == j]
                centers[j] = np.median(members, axis=0) if len(members) else x[rng.integers(len(x))]
        objective = float(np.abs(x - centers[labels]).sum())
        if objective < best_objective:
            best_labels, best_objective = labels.copy(), objective
    return best_labels, best_objective


def subsample_stability(x, k, baseline, seed=42, repeats=20, fraction=.8, distance="euclidean"):
    """Mean/std adjusted Rand index between the full fit and re-fits on 80% subsamples."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        idx = rng.choice(len(x), size=max(k + 1, int(fraction * len(x))), replace=False)
        repeat_seed = int(rng.integers(1_000_000))
        if distance == "euclidean":
            labels = KMeans(n_clusters=k, n_init=10, random_state=repeat_seed).fit_predict(x[idx])
        else:
            labels, _ = kmedians(x[idx], k, repeat_seed, n_init=3)
        scores.append(adjusted_rand_score(baseline[idx], labels))
    return {"adjusted_rand_mean": float(np.mean(scores)),
            "adjusted_rand_std": float(np.std(scores)), "runs": repeats}


def _scores(x, labels, distance):
    return {"silhouette": float(silhouette_score(x, labels, metric=distance)),
            "davies_bouldin_euclidean": float(davies_bouldin_score(x, labels)),
            "min_cluster_fraction": float(np.bincount(labels).min() / len(labels))}


def fit_configurations(profiles: pd.DataFrame, cluster_counts=(2, 3, 4, 5),
                       distance="euclidean", seed=42):
    """Fit every configuration, choose one by the documented rule, return results.

    Returns (clustered_frame, metrics_frame, selected_stability, artefacts).
    """
    if distance not in LINKAGES:
        raise ValueError("distance must be euclidean or manhattan")
    data = profiles.copy()
    x, imputer, scaler = preprocess(data)
    partition_name = "kmeans" if distance == "euclidean" else "kmedians"
    rows, labels_by_k, stability_by_k = [], {}, {}
    for k in cluster_counts:
        if k < 2 or k >= len(x):
            continue
        if distance == "euclidean":
            model = KMeans(n_clusters=k, n_init=20, random_state=seed)
            labels = model.fit_predict(x)
            objective = {"wcss_euclidean": float(model.inertia_)}
        else:
            labels, value = kmedians(x, k, seed)
            objective = {"l1_objective": value}
        labels_by_k[k] = labels
        stability_by_k[k] = subsample_stability(x, k, labels, seed, distance=distance)
        rows.append({"k": k, "distance": distance, "method": partition_name, **objective,
                     **_scores(x, labels, distance),
                     "stability_ari_mean": stability_by_k[k]["adjusted_rand_mean"],
                     "stability_ari_std": stability_by_k[k]["adjusted_rand_std"]})
        # Hierarchical comparison: several linkage rules, agreement with the partition.
        for linkage in LINKAGES[distance]:
            h_labels = AgglomerativeClustering(n_clusters=k, metric=distance,
                                               linkage=linkage).fit_predict(x)
            rows.append({"k": k, "distance": distance, "method": f"hierarchical_{linkage}",
                         **_scores(x, h_labels, distance),
                         "ari_vs_partition": float(adjusted_rand_score(labels, h_labels))})
    if not labels_by_k:
        raise ValueError("Need at least one valid cluster count and enough profiles.")

    # Selection rule: drop tiny groups (<5%), keep Silhouettes within 0.02 of the best,
    # prefer higher subsample stability, break ties with Davies-Bouldin.
    partition_rows = [r for r in rows if r["method"] == partition_name]
    eligible = [r for r in partition_rows if r["min_cluster_fraction"] >= 0.05] or partition_rows
    best_silhouette = max(r["silhouette"] for r in eligible)
    shortlist = [r for r in eligible if best_silhouette - r["silhouette"] <= 0.02]
    best_k = sorted(shortlist, key=lambda r: (-r["stability_ari_mean"],
                                              r["davies_bouldin_euclidean"]))[0]["k"]
    data["cluster"] = labels_by_k[best_k]
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(x)
    data["pca_1"], data["pca_2"] = coords[:, 0], coords[:, 1]
    metrics = pd.DataFrame(rows)
    metrics["selected"] = metrics.k.eq(best_k) & metrics.method.eq(partition_name)
    metrics["pca_variance_2d"] = float(pca.explained_variance_ratio_.sum())
    return data, metrics, stability_by_k[best_k], {
        "scaler": scaler, "imputer": imputer, "pca": pca, "features": FEATURES, "x": x}

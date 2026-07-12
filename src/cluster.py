"""
cluster.py
==========
Notebook 3 territory. Unsupervised segmentation.

Two algorithms, deliberately:
  K-Means -> hard assignment, spherical equal-variance assumption, fast.
  GMM     -> soft assignment, per-cluster variance, gives us a *probability*
             we can threshold on. `covariance_type='diag'` because full
             covariance on 57M rows is 6 x 4 x 4 parameters estimated over a
             matrix that will not fit in RAM, and our features are already
             roughly decorrelated after scaling.

IMPORTANT ON METRICS
--------------------
Clustering has no ground truth, so it has no accuracy / precision / recall.
It has:
  silhouette        [-1, 1]  higher better  - separation vs cohesion
  davies-bouldin    [0, inf) lower  better  - avg similarity to nearest cluster
  calinski-harabasz [0, inf) higher better  - between/within dispersion ratio
  BIC / AIC (GMM)   lower better             - likelihood penalised by params
Accuracy/precision/recall arrive in classify.py, where a *label* exists.
"""


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from . import config as C


def _subsample(X: np.ndarray, n: int, seed: int = C.RANDOM_STATE) -> np.ndarray:
    if len(X) <= n:
        return np.arange(len(X))
    rng = np.random.default_rng(seed)
    return rng.choice(len(X), size=n, replace=False)


def sweep_k(X: np.ndarray) -> pd.DataFrame:
    """
    Elbow + silhouette + DB + CH across K_RANGE.

    Silhouette is O(n^2) in memory, so it is scored on a SIL_SAMPLE subsample.
    Inertia is computed on the full matrix - it is cheap and the elbow location
    is sensitive to sample size.
    """
    idx = _subsample(X, C.SIL_SAMPLE)
    rows = []
    for k in C.K_RANGE:
        km = KMeans(n_clusters=k, random_state=C.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(X[idx], labels[idx])),
                "davies_bouldin": float(davies_bouldin_score(X[idx], labels[idx])),
                "calinski_harabasz": float(calinski_harabasz_score(X[idx], labels[idx])),
            }
        )
    return pd.DataFrame(rows)


def elbow_point(sweep: pd.DataFrame) -> int:
    """
    Kneedle-style elbow: the k whose inertia is furthest (perpendicular
    distance) from the straight line joining the first and last sweep points.
    Beats eyeballing the chart, and it is reproducible.
    """
    k = sweep["k"].to_numpy(dtype=float)
    y = sweep["inertia"].to_numpy(dtype=float)
    p1, p2 = np.array([k[0], y[0]]), np.array([k[-1], y[-1]])
    line = p2 - p1
    line = line / np.linalg.norm(line)
    dists = []
    for i in range(len(k)):
        v = np.array([k[i], y[i]]) - p1
        proj = np.dot(v, line) * line
        dists.append(np.linalg.norm(v - proj))
    return int(k[int(np.argmax(dists))])


def fit_kmeans(X: np.ndarray) -> tuple[KMeans, np.ndarray]:
    km = KMeans(**C.KMEANS_PARAMS)
    return km, km.fit_predict(X)


def fit_gmm(X: np.ndarray) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    gmm = GaussianMixture(**C.GMM_PARAMS)
    labels = gmm.fit_predict(X)
    proba = gmm.predict_proba(X)
    return gmm, labels, proba


def evaluate(X: np.ndarray, labels: np.ndarray, name: str) -> dict:
    idx = _subsample(X, C.SIL_SAMPLE)
    return {
        "model": name,
        "n_clusters": int(len(np.unique(labels))),
        "silhouette": float(silhouette_score(X[idx], labels[idx])),
        "davies_bouldin": float(davies_bouldin_score(X[idx], labels[idx])),
        "calinski_harabasz": float(calinski_harabasz_score(X[idx], labels[idx])),
    }


def profile_and_name(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, dict]:
    """
    Build the cluster profile, rank by average revenue, and attach the business
    segment names.

    THIS IS THE STEP MOST PEOPLE GET WRONG.
    Cluster ids from K-Means/GMM are arbitrary labels. Cluster 0 in one run is
    cluster 4 in the next. Hard-coding {0: "Question Mark", 1: "Loyal Core"}
    silently mislabels the entire customer base whenever the seed, the data, or
    the sklearn version changes. Naming must be derived from a stable, ordered
    business property. Here that property is mean revenue.
    """
    prof = (
        df.groupby(label_col)[C.CLUSTER_FEATURES + ["aon"]]
        .mean()
        .join(df.groupby(label_col).size().rename("n_customers"))
    )
    prof["pct_of_base"] = 100 * prof["n_customers"] / prof["n_customers"].sum()
    prof["revenue_rank"] = (
        prof["average_revenue"].rank(method="first", ascending=False).astype(int)
    )
    prof["segment"] = prof["revenue_rank"].map(C.SEGMENT_NAMES_BY_RANK)

    mapping = prof["segment"].to_dict()  # cluster_id -> segment name
    prof = prof.sort_values("revenue_rank")
    return prof, mapping

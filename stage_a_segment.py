"""
run_pipeline.py
===============
Executes every stage in order and writes:
  data/cleaned_data.csv
  outputs/*.csv
  outputs/metrics.json      <- everything the dashboard reads
  models/*.joblib
  figures/*.png
"""

import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import cluster, data, preprocess
from src import config as C

warnings.filterwarnings("ignore")
t0 = time.time()
M = {}  # everything worth persisting


def banner(msg):
    print(f"\n{'='*72}\n  {msg}\n{'='*72}")


# ============================================================ 1. LOAD
banner("STAGE 1  Data loading")
df = data.load_raw()
df = data.optimise_dtypes(df)
print(f"Raw shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"Memory:    {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
M["raw_rows"] = int(len(df))

# ============================================================ 2. PREPROCESS
banner("STAGE 2  Cleaning, AON filter, outliers, feature engineering")
df, report, caps = preprocess.run(df)
print(report.render())
print("\nOutlier fences (Tukey k=3.0):")
for c, v in caps.items():
    print(f"  {c:<24} [{v['low']:.2f}, {v['high']:.2f}]  capped={v['n_capped']:,}")
M["cleaning_report"] = dict(report)
M["outlier_caps"] = caps
M["clean_rows"] = int(len(df))
M["churn_base_rate"] = float(df["churn"].mean())
print(f"\nBase churn rate: {M['churn_base_rate']*100:.2f}%")

df.to_csv(C.CLEAN_PATH, index=False)

# ============================================================ 3. SCALE + SWEEP K
banner("STAGE 3  Scaling and choosing k")
X, scaler = preprocess.scale_for_clustering(df)
sweep = cluster.sweep_k(X)
print(sweep.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

elbow_k = cluster.elbow_point(sweep)
sil_k = int(sweep.loc[sweep.silhouette.idxmax(), "k"])
print(f"\nElbow (kneedle)   -> k = {elbow_k}")
print(f"Best silhouette   -> k = {sil_k}")
print(f"Business decision -> k = {C.N_CLUSTERS}  (locked: 6 named segments)")
M["k_sweep"] = sweep.to_dict("records")
M["elbow_k"] = elbow_k
M["silhouette_k"] = sil_k
M["chosen_k"] = C.N_CLUSTERS
sweep.to_csv(C.OUT_DIR / "k_sweep.csv", index=False)

# ---- figure: elbow + silhouette
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(sweep.k, sweep.inertia, "o-", color="#1f4e79", lw=2)
ax[0].axvline(elbow_k, ls="--", c="#c0392b", label=f"elbow k={elbow_k}")
ax[0].set(xlabel="k", ylabel="Inertia (WCSS)", title="Elbow method")
ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(sweep.k, sweep.silhouette, "o-", color="#1f4e79", lw=2)
ax[1].axvline(C.N_CLUSTERS, ls="--", c="#27ae60", label=f"chosen k={C.N_CLUSTERS}")
ax[1].set(xlabel="k", ylabel="Silhouette", title="Silhouette by k")
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(C.FIG_DIR / "elbow_silhouette.png", dpi=130); plt.close()

# ============================================================ 4. CLUSTER
banner("STAGE 4  K-Means vs Gaussian Mixture Model")
km, k_labels = cluster.fit_kmeans(X)
km_eval = cluster.evaluate(X, k_labels, "kmeans")
print("K-Means :", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in km_eval.items()})

gmm, g_labels, g_proba = cluster.fit_gmm(X)
gmm_eval = cluster.evaluate(X, g_labels, "gmm")
gmm_eval["bic"] = float(gmm.bic(X))
gmm_eval["aic"] = float(gmm.aic(X))
gmm_eval["log_likelihood"] = float(gmm.score(X) * len(X))
print("GMM     :", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in gmm_eval.items()})

M["kmeans_eval"] = km_eval
M["gmm_eval"] = gmm_eval

df["kmeans_cluster"] = k_labels
df["gmm_cluster"] = g_labels
df["gmm_confidence"] = g_proba.max(axis=1)

agree = float((pd.crosstab(k_labels, g_labels).max(axis=1).sum()) / len(df))
M["kmeans_gmm_agreement"] = agree
print(f"\nK-Means / GMM agreement (best 1-1 match): {agree*100:.2f}%")
print(f"Mean GMM assignment confidence:           {df.gmm_confidence.mean():.4f}")
print(f"Customers with confidence < 0.60:         {(df.gmm_confidence < 0.60).sum():,}")
M["mean_gmm_confidence"] = float(df.gmm_confidence.mean())
M["low_confidence_customers"] = int((df.gmm_confidence < 0.60).sum())

# ============================================================ 5. PROFILE + NAME
banner("STAGE 5  Cluster profiling and revenue-rank segment naming")
profile, cluster_to_segment = cluster.profile_and_name(df, "gmm_cluster")
df["segment"] = df["gmm_cluster"].map(cluster_to_segment)
print(profile.to_string(float_format=lambda v: f"{v:,.2f}"))
print(f"\nCluster id -> segment: {cluster_to_segment}")
profile.to_csv(C.OUT_DIR / "cluster_profile.csv")
M["cluster_to_segment"] = {int(k): v for k, v in cluster_to_segment.items()}
M["profile"] = profile.reset_index().to_dict("records")


# ---------------------------------------------------------------- checkpoint
import pickle

joblib.dump(scaler, C.MODEL_DIR / "scaler.joblib")
joblib.dump(gmm, C.MODEL_DIR / "gmm.joblib")
joblib.dump(km, C.MODEL_DIR / "kmeans.joblib")
df.to_pickle(C.DATA_DIR / "_ckpt_df.pkl")
with open(C.DATA_DIR / "_ckpt_meta.pkl", "wb") as f:
    pickle.dump({"M": M, "caps": caps, "cluster_to_segment": cluster_to_segment}, f)
print(f"\n[checkpoint written]  elapsed {time.time()-t0:.1f}s")

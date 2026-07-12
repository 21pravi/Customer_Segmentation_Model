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

import json
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from src import campaign, classify, cluster, data, preprocess
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

# ============================================================ 6. CHURN MODEL
banner("STAGE 6  Supervised churn model  (accuracy / precision / recall live here)")
Xc, yc, churn_features = classify.build_churn_matrix(df)
bundle = classify.train_churn_models(Xc, yc)
Xtr, Xte, ytr, yte = bundle["splits"]

print("Model comparison (threshold 0.50):")
for name, r in bundle["results"].items():
    print(f"  {name:<22} AUC={r['auc']:.4f}  acc={r['accuracy']:.4f}  f1={r['f1']:.4f}")
print(f"\nSelected: {bundle['best_name']}")
M["churn_model_comparison"] = {
    n: {"auc": r["auc"], "accuracy": r["accuracy"], "f1": r["f1"]}
    for n, r in bundle["results"].items()
}
M["churn_best_model"] = bundle["best_name"]

proba = bundle["best_proba"]
thr, pr_curve = classify.optimal_threshold(yte, proba)
print(f"F1-optimal threshold: {thr:.4f}  (default would be 0.5000)")

m_default = classify.full_metrics(yte, proba, 0.50)
m_tuned = classify.full_metrics(yte, proba, thr)

print("\n--- At threshold 0.50 (naive) ---")
for k in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "mcc"):
    print(f"  {k:<18} {m_default[k]:.4f}")
print("\n--- At F1-optimal threshold ---")
for k in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "mcc"):
    print(f"  {k:<18} {m_tuned[k]:.4f}")
cm = m_tuned["confusion_matrix"]
print("\nConfusion matrix @ tuned threshold:")
print("              pred_stay   pred_churn")
print(f"  act_stay    {cm['tn']:>9,}   {cm['fp']:>10,}")
print(f"  act_churn   {cm['fn']:>9,}   {cm['tp']:>10,}")

M["churn_metrics_default"] = m_default
M["churn_metrics_tuned"] = m_tuned
M["churn_threshold"] = thr

cvres = classify.cross_validate_churn(bundle["best_model"], Xc, yc)
print(
    f"\n{C.CV_FOLDS}-fold CV  ROC-AUC = "
    f"{cvres['roc_auc_mean']:.4f} +/- {cvres['roc_auc_std']:.4f}"
)
print(f"{C.CV_FOLDS}-fold CV  F1      = {cvres['f1_mean']:.4f} +/- {cvres['f1_std']:.4f}")
M["churn_cv"] = cvres

# feature importance
best = bundle["best_model"]
if hasattr(best, "feature_importances_"):
    imp = pd.Series(best.feature_importances_, index=churn_features).sort_values(ascending=False)
    print("\nTop 10 churn drivers:")
    print(imp.head(10).to_string(float_format=lambda v: f"{v:.4f}"))
    M["feature_importance"] = imp.head(15).to_dict()

# ROC curve figure
fpr, tpr, _ = roc_curve(yte, proba)
plt.figure(figsize=(5.5, 5))
plt.plot(fpr, tpr, lw=2.2, color="#1f4e79", label=f"AUC = {m_tuned['roc_auc']:.4f}")
plt.plot([0, 1], [0, 1], "--", c="#999")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("Churn model ROC"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(C.FIG_DIR / "roc_curve.png", dpi=130); plt.close()
M["roc_points"] = [
    {"fpr": float(a), "tpr": float(b)}
    for a, b in zip(fpr[:: max(1, len(fpr) // 120)], tpr[:: max(1, len(tpr) // 120)])
]
M["pr_curve"] = pr_curve.iloc[:: max(1, len(pr_curve) // 100)].to_dict("records")

# ============================================================ 7. SEGMENT SURROGATE
banner("STAGE 7  Segment surrogate classifier (fidelity to GMM)")
surro = classify.train_segment_surrogate(df)
print(f"Accuracy       {surro['accuracy']:.4f}")
print(f"Macro precision{surro['macro_precision']:>8.4f}")
print(f"Macro recall   {surro['macro_recall']:>8.4f}")
print(f"Macro F1       {surro['macro_f1']:>8.4f}")
print(f"Cohen's kappa  {surro['cohen_kappa']:>8.4f}")
print(f"\n{surro['report_text']}")
M["segment_surrogate"] = {
    k: v for k, v in surro.items() if k not in ("model", "report_text")
}

# ============================================================ 8. CAMPAIGN
banner("STAGE 8  UCG / UTG split and offer assignment")
df["churn_probability"] = best.predict_proba(Xc)[:, 1]
df["churn_prediction"] = (df.churn_probability >= thr).astype(int)

df = campaign.assign_ucg_utg(df)
bal = campaign.validate_ucg(df)
print("Randomisation balance check (|SMD| < 0.10 == balanced):")
print(bal.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
M["ucg_balance"] = bal.to_dict("records")
M["ucg_all_balanced"] = bool(bal.balanced.all())

counts = df.campaign_group.value_counts()
print(f"\nUTG: {counts.get('UTG',0):,}  ({100*counts.get('UTG',0)/len(df):.2f}%)")
print(f"UCG: {counts.get('UCG',0):,}  ({100*counts.get('UCG',0)/len(df):.2f}%)")
M["utg_count"] = int(counts.get("UTG", 0))
M["ucg_count"] = int(counts.get("UCG", 0))

df = campaign.recommend_offers(df)
summary = campaign.campaign_summary(df)
print("\nSegment summary:")
print(summary.to_string(float_format=lambda v: f"{v:,.2f}"))
summary.to_csv(C.OUT_DIR / "segment_summary.csv")
M["segment_summary"] = summary.reset_index().to_dict("records")
M["total_monthly_revenue"] = float(df.average_revenue.sum())
M["total_revenue_at_risk"] = float((df.average_revenue * df.churn_probability).sum())

# ============================================================ 9. PERSIST
banner("STAGE 9  Persisting models and outputs")
joblib.dump(scaler, C.MODEL_DIR / "scaler.joblib")
joblib.dump(gmm, C.MODEL_DIR / "gmm.joblib")
joblib.dump(km, C.MODEL_DIR / "kmeans.joblib")
joblib.dump(best, C.MODEL_DIR / "churn_model.joblib")
joblib.dump(surro["model"], C.MODEL_DIR / "segment_surrogate.joblib")
joblib.dump(
    {
        "cluster_to_segment": cluster_to_segment,
        "threshold": thr,
        "churn_features": churn_features,
        "caps": caps,
    },
    C.MODEL_DIR / "metadata.joblib",
)

export_cols = [
    "customer_id", "average_data_usage", "average_revenue", "average_dou",
    "pack_flag", "aon", "gmm_cluster", "segment", "gmm_confidence",
    "churn", "churn_probability", "churn_prediction",
    "campaign_group", "offer", "action", "priority",
]
df[export_cols].to_csv(C.OUT_DIR / "customer_segmentation_output.csv", index=False)
df[df.campaign_group == "UCG"][export_cols].to_csv(C.OUT_DIR / "ucg_holdout.csv", index=False)
df[df.campaign_group == "UTG"][export_cols].to_csv(C.OUT_DIR / "utg_campaign.csv", index=False)

# distribution for the dashboard
seg_order = [C.SEGMENT_NAMES_BY_RANK[r] for r in sorted(C.SEGMENT_NAMES_BY_RANK)]
M["segment_order"] = seg_order
M["runtime_seconds"] = round(time.time() - t0, 1)

with open(C.OUT_DIR / "metrics.json", "w") as f:
    json.dump(M, f, indent=2, default=str)

print(f"\nDone in {M['runtime_seconds']}s")
print(f"metrics.json written to {C.OUT_DIR/'metrics.json'}")

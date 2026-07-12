"""
stage_b_predict.py
==================
Resumes from the Stage A checkpoint: churn model, surrogate, campaign, exports.
Split from Stage A purely so each script finishes inside a single CPU budget.
"""
import json
import pickle
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from src import campaign, classify
from src import config as C

warnings.filterwarnings("ignore")
t0 = time.time()

def banner(msg):
    print(f"\n{'='*72}\n  {msg}\n{'='*72}")

df = pd.read_pickle(C.DATA_DIR / "_ckpt_df.pkl")
with open(C.DATA_DIR / "_ckpt_meta.pkl", "rb") as f:
    ck = pickle.load(f)
M, caps, cluster_to_segment = ck["M"], ck["caps"], ck["cluster_to_segment"]
print(f"Resumed checkpoint: {len(df):,} rows, {df.segment.nunique()} segments")

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
# scaler / gmm / kmeans were already persisted by stage_a_segment.py
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

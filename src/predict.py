"""
predict.py
==========
Notebook 6 territory. Scoring a customer who was not in the training base.

The artefacts a prediction needs, in order:
  scaler          -> put the new row on the same scale the GMM was fitted on
  gmm             -> soft-assign to a cluster, with a confidence
  cluster_to_seg  -> translate arbitrary cluster id into a business name
  churn_model     -> probability of leaving, at the tuned threshold
  threshold       -> the F1-optimal cut, not 0.5

Everything is loaded from `models/` so this file has zero dependency on
whether the training run is still in memory.
"""

import joblib
import numpy as np
import pandas as pd

from . import campaign
from . import config as C


class SegmentScorer:
    def __init__(self, model_dir=C.MODEL_DIR):
        self.scaler = joblib.load(model_dir / "scaler.joblib")
        self.gmm = joblib.load(model_dir / "gmm.joblib")
        self.churn_model = joblib.load(model_dir / "churn_model.joblib")
        bundle = joblib.load(model_dir / "metadata.joblib")
        self.cluster_to_segment = bundle["cluster_to_segment"]
        self.threshold = bundle["threshold"]
        self.churn_features = bundle["churn_features"]
        self.caps = bundle["caps"]

    # -------------------------------------------------- feature construction
    def _prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()

        # Apply the SAME outlier fences learned at training time. Recomputing
        # fences on the scoring batch would make the transform non-stationary.
        for col, cap in self.caps.items():
            df[col] = df[col].clip(cap["low"], cap["high"])

        df["average_dou"] = df["average_dou"].clip(0, 30)
        df.loc[df["average_revenue"] < 0, "average_revenue"] = 0.0

        eps = 1e-6
        df["revenue_per_gb"] = df["average_revenue"] / (df["average_data_usage"] + eps)
        df["usage_intensity"] = df["average_data_usage"] / (df["average_dou"] + eps)
        df["revenue_per_active_day"] = df["average_revenue"] / (df["average_dou"] + eps)
        df["is_dormant"] = (df["average_dou"] <= 5).astype(int)
        return df

    # -------------------------------------------------- public API
    def score(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        raw must contain: average_data_usage, average_revenue, average_dou,
                          pack_flag, aon
        Returns one row per input with segment, confidence, churn probability,
        churn flag, offer and priority.
        """
        eligible = raw["aon"] >= C.AON_MIN_DAYS

        df = self._prepare(raw)

        Xs = self.scaler.transform(df[C.CLUSTER_FEATURES].to_numpy(dtype=np.float64))
        cluster = self.gmm.predict(Xs)
        proba = self.gmm.predict_proba(Xs)

        df["cluster_id"] = cluster
        df["segment"] = [self.cluster_to_segment[c] for c in cluster]
        df["segment_confidence"] = proba.max(axis=1)

        # Second-best segment: tells you how borderline the assignment is.
        order = np.argsort(-proba, axis=1)
        df["runner_up_segment"] = [self.cluster_to_segment[order[i, 1]] for i in range(len(df))]
        df["runner_up_confidence"] = proba[np.arange(len(df)), order[:, 1]]

        # Churn matrix must have exactly the training columns, in order.
        seg_dummies = pd.get_dummies(df["segment"], prefix="seg", dtype="int8")
        Xc = pd.concat([df[C.CLUSTER_FEATURES + C.CHURN_EXTRA_FEATURES], seg_dummies], axis=1)
        Xc = Xc.reindex(columns=self.churn_features, fill_value=0)

        df["churn_probability"] = self.churn_model.predict_proba(Xc)[:, 1]
        df["churn_prediction"] = (df["churn_probability"] >= self.threshold).astype(int)
        df["risk_band"] = pd.cut(
            df["churn_probability"],
            bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
            labels=["Low", "Moderate", "High", "Critical"],
        )

        df["offer"] = df["segment"].map(lambda s: campaign.OFFER_BOOK[s][0])
        df["action"] = df["segment"].map(lambda s: campaign.OFFER_BOOK[s][1])

        # Honour the AON rule at inference, not just at training.
        df.loc[~eligible, ["segment", "offer", "action"]] = [
            "UNSEGMENTED (AON < 90d)", "Hold - insufficient tenure", "None",
        ]
        return df


def predict_customer(avg_data: float, avg_rev: float, avg_dou: float,
                     pack_flag: int, aon: int) -> dict:
    """Convenience single-row wrapper."""
    row = pd.DataFrame(
        [{
            "average_data_usage": avg_data,
            "average_revenue": avg_rev,
            "average_dou": avg_dou,
            "pack_flag": pack_flag,
            "aon": aon,
        }]
    )
    out = SegmentScorer().score(row).iloc[0]
    return {
        "segment": out["segment"],
        "segment_confidence": round(float(out["segment_confidence"]), 4),
        "runner_up_segment": out["runner_up_segment"],
        "churn_probability": round(float(out["churn_probability"]), 4),
        "churn_prediction": int(out["churn_prediction"]),
        "risk_band": str(out["risk_band"]),
        "offer": out["offer"],
    }

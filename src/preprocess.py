"""
preprocess.py
=============
Notebook 2 territory. Everything between "raw CSV" and "a matrix a model can eat".

Order matters and is not arbitrary:
  dedupe -> AON filter -> invalid values -> impute -> outlier cap -> engineer -> scale

Why this order:
  * Dedupe first so duplicated rows do not skew the medians used for imputation.
  * AON filter before imputation so we impute from the population we will
    actually model, not from tourists who joined last week.
  * Impute before outlier capping because the IQR fences need complete columns.
  * Engineer after capping so ratios are not built on 1400 GB nonsense.
  * Scale last, fitted on train only when a classifier is downstream.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as C


class CleaningReport(dict):
    """Dict that also prints like an audit log. Every drop is accounted for."""

    def render(self) -> str:
        w = max(len(k) for k in self) + 2
        return "\n".join(f"  {k:<{w}} {v:>12,}" for k, v in self.items())


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    rep = CleaningReport()
    rep["rows_in"] = len(df)

    # ---- 1. duplicates -------------------------------------------------
    # customer_id is a surrogate key from the generator, so dedupe on the
    # behavioural payload, not the id.
    payload = [c for c in df.columns if c not in ("customer_id", "true_population")]
    before = len(df)
    df = df.drop_duplicates(subset=payload, keep="first").reset_index(drop=True)
    rep["dropped_duplicates"] = before - len(df)

    # ---- 2. AON filter -------------------------------------------------
    # The single most important business rule in the pipeline. Sub-90-day
    # subscribers have not established a usage baseline; including them creates
    # a spurious "low usage" cluster that is really just "new".
    before = len(df)
    df = df[df["aon"] >= C.AON_MIN_DAYS].reset_index(drop=True)
    rep["dropped_by_aon_filter"] = before - len(df)

    # ---- 3. invalid values --------------------------------------------
    # Negative revenue is a credit note, not a subscriber behaviour. Floor at 0
    # rather than dropping: the customer is real, the adjustment is an artefact.
    neg = int((df["average_revenue"] < 0).sum())
    df.loc[df["average_revenue"] < 0, "average_revenue"] = 0.0
    rep["negative_revenue_floored"] = neg

    # dou is days out of a 30-day window. Anything outside is a pipeline bug.
    bad_dou = int(((df["average_dou"] < 0) | (df["average_dou"] > 30)).sum())
    df["average_dou"] = df["average_dou"].clip(0, 30)
    rep["dou_clipped_to_0_30"] = bad_dou

    # ---- 4. missing values --------------------------------------------
    # Median, not mean: both columns are right-skewed, and the mean is dragged
    # by the corporate-SIM tail we have not capped yet at this point.
    for col in ("average_data_usage", "average_revenue"):
        n_miss = int(df[col].isna().sum())
        if n_miss:
            df[col] = df[col].fillna(df[col].median())
        rep[f"imputed_{col}"] = n_miss

    return df, rep


def cap_outliers(df: pd.DataFrame, cols: list[str], k: float = 3.0) -> tuple[pd.DataFrame, dict]:
    """
    Winsorise at Tukey fences with k=3.0 (the 'far out' fence, not the usual 1.5).

    k=1.5 would clip ~2-3% of a right-skewed telecom usage column, which throws
    away the genuinely heavy users we most want to find. k=3.0 clips only the
    corporate/IoT SIMs. We cap rather than drop because these are paying
    customers who belong in the segment counts, just not at their raw magnitude.
    """
    caps = {}
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        df[col] = df[col].clip(lo, hi)
        caps[col] = {"low": float(lo), "high": float(hi), "n_capped": n}
    return df, caps


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratios that encode business meaning the raw columns cannot.

    revenue_per_gb        - monetisation efficiency. A Sleeping Giant has high
                            data and low revenue_per_gb: they consume without
                            paying proportionally. That is an upsell target.
    usage_intensity       - GB per active day. Separates "heavy on few days"
                            from "light every day".
    revenue_per_active_day- ARPU normalised by engagement.
    is_dormant            - hard flag, dou <= 5 days in the month.
    """
    eps = 1e-6
    df["revenue_per_gb"] = df["average_revenue"] / (df["average_data_usage"] + eps)
    df["usage_intensity"] = df["average_data_usage"] / (df["average_dou"] + eps)
    df["revenue_per_active_day"] = df["average_revenue"] / (df["average_dou"] + eps)
    df["is_dormant"] = (df["average_dou"] <= 5).astype("int8")

    # Ratios with a near-zero denominator explode. Cap at the 99.5th pct.
    for col in ("revenue_per_gb", "usage_intensity", "revenue_per_active_day"):
        hi = df[col].quantile(0.995)
        df[col] = df[col].clip(upper=hi).astype("float32")
    return df


def scale_for_clustering(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    StandardScaler over the four clustering features.

    Note on pack_flag: it is binary, and standardising a binary column is
    defensible here because K-Means uses Euclidean distance and we want the
    flag to carry roughly one feature's worth of weight, not to be drowned by
    revenue's raw scale (hundreds) or amplified relative to dou (tens).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[C.CLUSTER_FEATURES].to_numpy(dtype=np.float64))
    return X, scaler


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport, dict]:
    df, rep = clean(df)
    df, caps = cap_outliers(df, ["average_data_usage", "average_revenue"])
    df = engineer(df)
    rep["rows_out"] = len(df)
    return df, rep, caps

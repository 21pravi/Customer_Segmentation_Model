"""
campaign.py
===========
Notebook 5 territory. Turning segments into a campaign.

=========================================================================
A CORRECTION TO THE PREVIOUS IMPLEMENTATION - PLEASE READ
=========================================================================
The earlier code assigned the control group like this:

    df = df.sort_values("gmm_probability", ascending=False)
    n = int(len(df) * 0.03)
    df["campaign"] = "UTG"
    df.iloc[:n, ...] = "UCG"          # <-- top 3% by cluster confidence

That is not a control group. That is the 3% of customers the GMM was *most
certain about* - i.e. the ones sitting closest to their cluster centroid, which
in a revenue-driven segmentation means a systematically different (and usually
higher-value, more archetypal) population than the 97% they are meant to be
compared against.

The whole purpose of a Universal Control Group is to be an unbiased
counterfactual: "what would these customers have done if we had left them
alone?" Selection on any variable correlated with the outcome destroys that.
You would measure the campaign's lift against a control that was never
comparable, and every incrementality number downstream would be wrong -
typically biased so the campaign looks worse than it is, because your control
group over-represents your best customers.

CORRECT: sample the UCG *at random*, stratified by segment so every segment
contributes ~3% of its own members. Stratification preserves segment mix
without introducing selection on outcome. Randomisation inside each stratum is
what makes the comparison causal.
=========================================================================
"""

import numpy as np
import pandas as pd

from . import config as C


def assign_ucg_utg(df: pd.DataFrame, seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    """
    Stratified random 3% holdout.

    Stratifying on segment guarantees the control mirrors the treatment on the
    one dimension we know drives the outcome. Within a stratum, assignment is
    a coin flip - no ordering, no confidence, no revenue.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["campaign_group"] = "UTG"

    ucg_idx = []
    for seg, grp in df.groupby("segment", sort=False):
        n_ucg = int(round(len(grp) * C.UCG_FRACTION))
        if n_ucg == 0 and len(grp) > 0:
            n_ucg = 1  # never leave a segment without a control
        picked = rng.choice(grp.index.to_numpy(), size=n_ucg, replace=False)
        ucg_idx.append(picked)

    ucg_idx = np.concatenate(ucg_idx)
    df.loc[ucg_idx, "campaign_group"] = "UCG"
    return df


def validate_ucg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balance check. If randomisation worked, UCG and UTG means should be within
    noise of each other on every feature. A standardised mean difference (SMD)
    above 0.10 on any covariate is the conventional red flag.

    Run this every single time. A control group you did not validate is a
    control group you cannot trust.
    """
    rows = []
    for col in C.CLUSTER_FEATURES + ["aon", "churn"]:
        t = df.loc[df.campaign_group == "UTG", col].astype(float)
        c = df.loc[df.campaign_group == "UCG", col].astype(float)
        pooled_sd = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
        smd = (t.mean() - c.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append(
            {
                "feature": col,
                "utg_mean": float(t.mean()),
                "ucg_mean": float(c.mean()),
                "abs_smd": float(abs(smd)),
                "balanced": bool(abs(smd) < 0.10),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- offers
OFFER_BOOK = {
    "Premium Power":     ("Priority 5G + concierge retention call", "Retain",  "Very High"),
    "Loyal Core":        ("Loyalty tier upgrade, 12-month lock-in",  "Retain",  "High"),
    "Sleeping Giant":    ("Data-to-revenue upsell: unlimited plan",  "Upsell",  "High"),
    "Deal Seeker":       ("Margin-safe bundle, no discount stacking","Sustain", "Medium"),
    "Question Mark":     ("Onboarding nudge + 3-day trial pack",     "Activate","Medium"),
    "Comfortably Numb":  ("Low-cost reactivation SMS, no spend",     "Reactivate","Low"),
}


def recommend_offers(df: pd.DataFrame, churn_col: str = "churn_probability") -> pd.DataFrame:
    """
    Offer + priority. Only UTG customers get contacted; the UCG is left
    deliberately untouched - that is the entire point of holding them out.

    Priority escalates when churn risk is high, because a Deal Seeker about to
    leave is worth more attention than a Deal Seeker who is not.
    """
    df = df.copy()
    df["offer"] = df["segment"].map(lambda s: OFFER_BOOK[s][0])
    df["action"] = df["segment"].map(lambda s: OFFER_BOOK[s][1])
    df["base_priority"] = df["segment"].map(lambda s: OFFER_BOOK[s][2])

    if churn_col in df.columns:
        escalate = df[churn_col] >= 0.60
        df.loc[escalate & (df.base_priority == "Medium"), "base_priority"] = "High"
        df.loc[escalate & (df.base_priority == "Low"), "base_priority"] = "Medium"

    # Control group is never contacted.
    df.loc[df.campaign_group == "UCG", ["offer", "action"]] = ["HOLDOUT - no contact", "None"]
    df.loc[df.campaign_group == "UCG", "base_priority"] = "N/A"
    return df.rename(columns={"base_priority": "priority"})


def campaign_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-segment rollup: who they are, how many, what they're worth, risk."""
    g = df.groupby("segment", sort=False)
    out = pd.DataFrame(
        {
            "customers": g.size(),
            "pct_of_base": 100 * g.size() / len(df),
            "avg_revenue": g["average_revenue"].mean(),
            "avg_data_gb": g["average_data_usage"].mean(),
            "avg_dou": g["average_dou"].mean(),
            "avg_aon_days": g["aon"].mean(),
            "pack_penetration": 100 * g["pack_flag"].mean(),
            "actual_churn_rate": 100 * g["churn"].mean(),
            "monthly_revenue_inr": g["average_revenue"].sum(),
        }
    )
    if "churn_probability" in df.columns:
        out["mean_churn_risk"] = g["churn_probability"].mean()
        # Revenue standing behind customers the model flags as high risk.
        out["revenue_at_risk_inr"] = g.apply(
            lambda x: float((x["average_revenue"] * x["churn_probability"]).sum()),
            include_groups=False,
        )
    out["ucg_count"] = df[df.campaign_group == "UCG"].groupby("segment", sort=False).size()
    out["utg_count"] = df[df.campaign_group == "UTG"].groupby("segment", sort=False).size()
    return out.sort_values("avg_revenue", ascending=False)

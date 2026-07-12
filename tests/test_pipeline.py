"""
Fast invariant tests. Deliberately small (n=6,000) so CI finishes in seconds
while still exercising every stage of the pipeline.

These test *invariants*, not exact numbers. Asserting silhouette == 0.4156 would
break on any sklearn point release; asserting that the AON filter removes every
subscriber below the threshold will not.
"""

import numpy as np
import pandas as pd
import pytest

from src import campaign, classify, cluster, data, preprocess
from src import config as C

N = 6_000


@pytest.fixture(scope="module")
def raw():
    return data.generate_synthetic(N)


@pytest.fixture(scope="module")
def clean(raw):
    df, report, caps = preprocess.run(raw.copy())
    return df, report, caps


# ------------------------------------------------------------------ data
def test_synthetic_has_required_schema(raw):
    required = {
        "customer_id", "average_data_usage", "average_revenue",
        "average_dou", "pack_flag", "aon", "churn",
    }
    assert required <= set(raw.columns)


def test_churn_is_binary_and_not_degenerate(raw):
    assert set(raw["churn"].unique()) <= {0, 1}
    rate = raw["churn"].mean()
    assert 0.10 < rate < 0.50, f"churn base rate {rate:.3f} is implausible"


def test_dirt_was_actually_injected(raw):
    """If the generator stops producing defects, preprocessing is untested."""
    assert raw["average_data_usage"].isna().any()
    assert (raw["average_revenue"] < 0).any()
    assert raw.duplicated(subset=["average_revenue", "average_dou", "aon"]).any()


# ------------------------------------------------------------ preprocessing
def test_aon_filter_removes_all_short_tenure(clean):
    df, _, _ = clean
    assert (df["aon"] >= C.AON_MIN_DAYS).all()


def test_no_missing_values_survive(clean):
    df, _, _ = clean
    assert not df[C.CLUSTER_FEATURES].isna().any().any()


def test_no_negative_revenue_survives(clean):
    df, _, _ = clean
    assert (df["average_revenue"] >= 0).all()


def test_dou_within_valid_range(clean):
    df, _, _ = clean
    assert df["average_dou"].between(0, 30).all()


def test_outlier_caps_are_enforced(clean):
    df, _, caps = clean
    for col, cap in caps.items():
        assert df[col].max() <= cap["high"] + 1e-6
        assert df[col].min() >= cap["low"] - 1e-6


def test_cleaning_report_reconciles(clean):
    """Every row that left the base must be accounted for."""
    df, rep, _ = clean
    expected = rep["rows_in"] - rep["dropped_duplicates"] - rep["dropped_by_aon_filter"]
    assert rep["rows_out"] == expected == len(df)


def test_engineered_features_are_finite(clean):
    df, _, _ = clean
    for col in ("revenue_per_gb", "usage_intensity", "revenue_per_active_day"):
        assert np.isfinite(df[col]).all(), f"{col} contains inf/nan"


# ---------------------------------------------------------------- clustering
def test_elbow_point_is_in_sweep_range():
    sweep = pd.DataFrame(
        {"k": list(C.K_RANGE), "inertia": [100, 55, 30, 24, 21, 19, 18, 17, 16]}
    )
    k = cluster.elbow_point(sweep)
    assert k in list(C.K_RANGE)


def test_segments_named_by_revenue_rank(clean):
    """
    The core invariant: rank 1 must be the highest-revenue cluster, always.
    This is what protects against cluster-id permutation across runs.
    """
    df, _, _ = clean
    X, _ = preprocess.scale_for_clustering(df)
    _, labels = cluster.fit_kmeans(X)
    df = df.assign(gmm_cluster=labels)

    profile, mapping = cluster.profile_and_name(df, "gmm_cluster")

    assert len(mapping) == C.N_CLUSTERS
    assert set(mapping.values()) == set(C.SEGMENT_NAMES_BY_RANK.values())

    # profile is sorted by revenue_rank; revenue must decrease monotonically
    revenues = profile["average_revenue"].to_numpy()
    assert (np.diff(revenues) <= 0).all(), "rank order does not track revenue"
    assert profile.iloc[0]["segment"] == C.SEGMENT_NAMES_BY_RANK[1]


# ---------------------------------------------------------------- classifier
def test_full_metrics_confusion_matrix_sums_to_n():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    proba = rng.random(1000)
    m = classify.full_metrics(y, proba, 0.5)
    cm = m["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 1000


def test_threshold_extremes_behave():
    """At t->0 everything is flagged; at t->1 nothing is."""
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.4, 0.6, 0.9])

    lo = classify.full_metrics(y, proba, 0.0)
    assert lo["recall"] == 1.0
    assert lo["confusion_matrix"]["tn"] == 0

    hi = classify.full_metrics(y, proba, 1.01)
    assert hi["recall"] == 0.0
    assert hi["confusion_matrix"]["tp"] == 0
    # and accuracy collapses to the majority-class baseline
    assert hi["accuracy"] == pytest.approx(0.5)


def test_optimal_threshold_beats_default_on_f1():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    proba = np.clip(y * 0.35 + rng.normal(0.35, 0.2, 2000), 0, 1)
    thr, curve = classify.optimal_threshold(y, proba)
    assert 0.0 < thr < 1.0
    best = classify.full_metrics(y, proba, thr)["f1"]
    default = classify.full_metrics(y, proba, 0.5)["f1"]
    assert best >= default - 1e-9


# ---------------------------------------------------------------- campaign
def test_ucg_is_three_percent_and_stratified(clean):
    df, _, _ = clean
    X, _ = preprocess.scale_for_clustering(df)
    _, labels = cluster.fit_kmeans(X)
    df = df.assign(gmm_cluster=labels)
    _, mapping = cluster.profile_and_name(df, "gmm_cluster")
    df["segment"] = df["gmm_cluster"].map(mapping)

    out = campaign.assign_ucg_utg(df)
    share = (out.campaign_group == "UCG").mean()
    assert abs(share - C.UCG_FRACTION) < 0.005

    # every segment contributes ~3% of its own members, and none is left
    # without a control
    for seg, grp in out.groupby("segment"):
        n_ucg = (grp.campaign_group == "UCG").sum()
        assert n_ucg >= 1
        assert abs(n_ucg / len(grp) - C.UCG_FRACTION) < 0.02


def test_ucg_is_balanced_on_covariates(clean):
    """
    The control group must not differ systematically from the treatment group.
    A holdout selected on cluster confidence or revenue would fail this.
    """
    df, _, _ = clean
    X, _ = preprocess.scale_for_clustering(df)
    _, labels = cluster.fit_kmeans(X)
    df = df.assign(gmm_cluster=labels)
    _, mapping = cluster.profile_and_name(df, "gmm_cluster")
    df["segment"] = df["gmm_cluster"].map(mapping)

    out = campaign.assign_ucg_utg(df)
    bal = campaign.validate_ucg(out)
    # n=6k is small, so allow the conventional 0.10 threshold some slack
    assert (bal["abs_smd"] < 0.20).all(), bal.to_string()


def test_every_segment_has_an_offer():
    assert set(campaign.OFFER_BOOK) == set(C.SEGMENT_NAMES_BY_RANK.values())

"""
data.py
=======
Notebook 1 territory: get raw data on disk, then load it.

Two modes:
  * USE_SYNTHETIC=True  -> fabricate a telecom subscriber base with six latent
                           behavioural populations and a churn process that
                           depends on behaviour (so a classifier has real signal
                           to find, but nowhere near perfect separation).
  * USE_SYNTHETIC=False -> read the real CSV and rename columns via COLUMN_MAP.

The synthetic generator deliberately injects the messiness you get in
production: missing values, negative revenue (credit adjustments), absurd
outliers (corporate SIMs), and duplicate rows. The preprocessing module has to
earn its keep.
"""

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------
# Latent population definitions.
# Each tuple: (share, data_gb_mu, data_gb_sd, rev_mu, rev_sd, dou_mu, dou_sd,
#              pack_prob, aon_mu, aon_sd, churn_logit_bias)
# Values are on the *natural* scale; revenue in INR/month, data in GB/month,
# dou = days-of-usage out of 30, aon = age-on-network in days.
# --------------------------------------------------------------------------
POPULATIONS = [
    # name              share  dataμ  dataσ  revμ   revσ  douμ douσ pack   aonμ   aonσ  bias
    ("premium_power",   0.08,  24.0,  6.0,   780.0, 150.0, 28.0, 2.0, 0.95, 1900,  600, -2.2),
    ("loyal_core",      0.17,  11.0,  3.0,   430.0,  80.0, 26.0, 3.0, 0.88, 2600,  700, -2.8),
    ("sleeping_giant",  0.13,  19.0,  5.0,   310.0,  70.0, 25.0, 4.0, 0.80,  900,  350, -0.6),
    ("deal_seeker",     0.22,   8.5,  2.5,   210.0,  45.0, 22.0, 5.0, 0.99,  700,  300, -0.1),
    ("question_mark",   0.19,   5.0,  2.2,   150.0,  50.0, 13.0, 6.0, 0.45,  260,  120,  0.9),
    ("comfortably_numb",0.21,   1.6,  0.9,    75.0,  30.0,  9.0, 5.0, 0.18, 2200,  800, -0.9),
]


def _sample_populations(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw n subscribers from the six latent populations."""
    shares = np.array([p[1] for p in POPULATIONS])
    shares = shares / shares.sum()
    counts = rng.multinomial(n, shares)

    frames = []
    for (name, _, dmu, dsd, rmu, rsd, umu, usd, pack, amu, asd, bias), cnt in zip(
        POPULATIONS, counts
    ):
        if cnt == 0:
            continue
        # Lognormal-ish for usage/revenue: right-skewed, strictly positive.
        data_gb = rng.normal(dmu, dsd, cnt).clip(0.0, None)
        revenue = rng.normal(rmu, rsd, cnt)
        dou = rng.normal(umu, usd, cnt).clip(0, 30).round()
        pack_flag = (rng.random(cnt) < pack).astype(int)
        aon = rng.normal(amu, asd, cnt).clip(1, None).round()

        frames.append(
            pd.DataFrame(
                {
                    "true_population": name,
                    "average_data_usage": data_gb,
                    "average_revenue": revenue,
                    "average_dou": dou,
                    "pack_flag": pack_flag,
                    "aon": aon,
                    "_churn_bias": bias,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1.0, random_state=C.RANDOM_STATE).reset_index(drop=True)


def _simulate_churn(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """
    Churn as a logistic function of behaviour + population bias + noise.

    Signal, in plain terms:
      - low days-of-usage  -> churn up   (disengagement is the strongest signal)
      - short tenure       -> churn up   (no switching cost yet)
      - no active pack     -> churn up
      - high revenue       -> churn down (invested customers stay)
    The noise term is large on purpose. A churn model that hits 0.99 AUC is a
    model with a leak, not a good model.
    """
    z = (
        df["_churn_bias"].to_numpy()
        - 0.085 * df["average_dou"].to_numpy()
        - 0.00042 * df["aon"].to_numpy()
        - 0.0021 * df["average_revenue"].to_numpy()
        - 0.62 * df["pack_flag"].to_numpy()
        + 0.019 * df["average_data_usage"].to_numpy()
        + 1.95  # intercept tuned to land the base rate near 22%, which is where
                # Indian prepaid telecom churn actually sits. A 43% base rate
                # makes every metric look better than it has any right to.
        + rng.normal(0, 1.15, len(df))  # irreducible noise -> caps achievable AUC
    )
    p = 1.0 / (1.0 + np.exp(-z))
    return (rng.random(len(df)) < p).astype(int)


def _inject_dirt(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Make the data look like it came out of a real billing warehouse."""
    n = len(df)

    # 1. Missing values (MCAR) in the two hardest-to-instrument fields.
    for col, frac in [("average_data_usage", 0.021), ("average_revenue", 0.014)]:
        idx = rng.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # 2. Credit adjustments -> genuinely negative revenue for ~0.4% of rows.
    idx = rng.choice(n, size=int(n * 0.004), replace=False)
    df.loc[idx, "average_revenue"] = -rng.uniform(5, 120, len(idx))

    # 3. Corporate / IoT SIMs -> extreme data usage outliers (~0.15%).
    idx = rng.choice(n, size=int(n * 0.0015), replace=False)
    df.loc[idx, "average_data_usage"] = rng.uniform(300, 1400, len(idx))

    # 4. Duplicate rows from a bad warehouse join (~0.5%).
    dupes = df.sample(frac=0.005, random_state=C.RANDOM_STATE)
    df = pd.concat([df, dupes], ignore_index=True)

    return df


def generate_synthetic(n: int = C.N_CUSTOMERS) -> pd.DataFrame:
    rng = np.random.default_rng(C.RANDOM_STATE)
    df = _sample_populations(n, rng)
    df["churn"] = _simulate_churn(df, rng)
    df = df.drop(columns=["_churn_bias"])
    df = _inject_dirt(df, rng)
    df.insert(0, "customer_id", [f"MSISDN{i:09d}" for i in range(len(df))])
    return df


def build_raw() -> pd.DataFrame:
    """Write raw_customers.csv to disk and return it."""
    df = generate_synthetic()
    df.to_csv(C.RAW_PATH, index=False)
    return df


def load_raw() -> pd.DataFrame:
    """
    Load whatever the configured source is.
    Real mode renames columns through COLUMN_MAP and validates the result.
    """
    if C.USE_SYNTHETIC:
        if not C.RAW_PATH.exists():
            return build_raw()
        return pd.read_csv(C.RAW_PATH)

    df = pd.read_csv(C.RAW_PATH)
    inverse = {v: k for k, v in C.COLUMN_MAP.items()}
    df = df.rename(columns=inverse)

    required = set(C.COLUMN_MAP.keys()) - {"customer_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Real CSV is missing required columns after COLUMN_MAP: {sorted(missing)}. "
            f"Edit COLUMN_MAP in config.py to point at the right source names."
        )
    return df


def optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Memory optimisation. Matters little at 200k rows; matters enormously at the
    57M rows of the real project, where naive float64 costs ~3.7 GB for four
    feature columns alone. float32 halves that; int8 for flags saves another 8x.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in ("pack_flag", "churn"):
        if col in df.columns:
            df[col] = df[col].astype("int8")
    if "aon" in df.columns:
        df["aon"] = df["aon"].astype("int32")
    return df

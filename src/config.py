"""
config.py
=========
Single source of truth for every tunable in the pipeline.

TO USE THE REAL KAGGLE DATA
---------------------------
1. Download `telecom_churn.csv` from
   https://www.kaggle.com/datasets/suraj520/telecom-churn-dataset
2. Drop it in `data/raw/`.
3. Set USE_SYNTHETIC = False below and map your real column names in
   COLUMN_MAP. Nothing else in the codebase needs to change.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models"

for _d in (DATA_DIR, OUT_DIR, FIG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_PATH = DATA_DIR / "raw_customers.csv"
CLEAN_PATH = DATA_DIR / "cleaned_data.csv"

# ---------------------------------------------------------------- data source
USE_SYNTHETIC = True          # <-- flip to False once you have the real CSV
N_CUSTOMERS = 200_000         # synthetic row count (real project: 57,390,987)
RANDOM_STATE = 42

# Map: canonical_name -> name in YOUR csv. Identity by default.
COLUMN_MAP = {
    "customer_id": "customer_id",
    "average_data_usage": "average_data_usage",
    "average_revenue": "average_revenue",
    "average_dou": "average_dou",
    "pack_flag": "pack_flag",
    "aon": "aon",
    "churn": "churn",
}

# ---------------------------------------------------------------- business rules
AON_MIN_DAYS = 90             # AON filter: drop customers newer than 90 days
                              # (their usage has not stabilised, so they would
                              #  pollute the cluster centroids)

CLUSTER_FEATURES = [
    "average_data_usage",
    "average_revenue",
    "average_dou",
    "pack_flag",
]

# Engineered features added on top of the four above, used ONLY by the
# supervised churn model (not by the clustering, to keep segments interpretable).
CHURN_EXTRA_FEATURES = [
    "aon",
    "revenue_per_gb",
    "usage_intensity",
    "revenue_per_active_day",
    "is_dormant",
]

# ---------------------------------------------------------------- clustering
K_RANGE = range(2, 11)        # elbow / silhouette sweep
N_CLUSTERS = 6                # locked at 6 per the original project
SIL_SAMPLE = 10_000           # silhouette is O(n^2) in time AND memory. Scores are
                              # stable from ~10k up (0.4153 -> 0.4107 across 5k->20k),
                              # so subsampling costs accuracy we cannot measure.

GMM_PARAMS = dict(
    n_components=N_CLUSTERS,
    covariance_type="diag",
    init_params="kmeans",
    reg_covar=1e-5,
    n_init=5,
    random_state=RANDOM_STATE,
)

KMEANS_PARAMS = dict(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10,
)

# ---------------------------------------------------------------- campaign
UCG_FRACTION = 0.03           # Universal Control Group -> 3%
UTG_FRACTION = 0.97           # Universal Target Group  -> 97%

# Segment names are assigned by REVENUE RANK, not by raw cluster id.
# Cluster ids are arbitrary and change between runs; ranks are stable.
# rank 1 = highest average revenue.
SEGMENT_NAMES_BY_RANK = {
    1: "Premium Power",       # top spend, top usage - protect at all costs
    2: "Loyal Core",          # loyal, long tenure, healthy spend
    3: "Sleeping Giant",      # heavy data, revenue lags usage - upsell
    4: "Deal Seeker",         # pack-driven, price sensitive, margin-thin
    5: "Question Mark",       # young tenure, erratic - could go either way
    6: "Comfortably Numb",    # low usage, low spend, sticky but dormant
}

# ---------------------------------------------------------------- classifier
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_SUBSAMPLE = 60_000         # CV on a subsample: 5 refits of a forest over the
                              # full base is minutes of CPU for a variance estimate
                              # that is already tight at 60k.

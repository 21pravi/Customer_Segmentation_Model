"""
predict_samples.py
==================
Ten hand-built subscribers, none of which were in the training base, run through
the full scoring path: scale -> GMM soft-assign -> segment -> churn -> offer.

The cases are chosen to probe the boundaries, not to flatter the model:
  1-6   one clean archetype per segment
  7     a borderline customer sitting between two clusters (low confidence)
  8     a brand-new subscriber -> must be refused by the AON rule
  9     a corporate/IoT SIM -> must be caught by the outlier fence
 10     a credit-adjustment customer with negative revenue -> must be floored
"""

import json

import pandas as pd

from src import config as C
from src.predict import SegmentScorer

pd.set_option("display.width", 200, "display.max_columns", 40)

CASES = [
    # id,                     data_gb, revenue, dou, pack, aon,   why
    ("HIGH_VALUE_WHALE",         26.0,   840.0,  29,   1,  2100, "textbook Premium Power"),
    ("LOYAL_LONG_TENURE",        14.0,   380.0,  26,   1,  2400, "textbook Loyal Core"),
    ("HEAVY_DATA_LOW_SPEND",     19.5,   300.0,  26,   0,  1500, "Sleeping Giant: upsell target"),
    ("PACK_HOPPER",               8.0,   200.0,  22,   1,   650, "textbook Deal Seeker"),
    ("ERRATIC_LOW_ENGAGE",        3.2,   115.0,  10,   1,  1000, "textbook Question Mark"),
    ("DORMANT_STICKY",            1.4,    80.0,   8,   0,  2300, "textbook Comfortably Numb"),
    ("BORDERLINE_MIDFIELD",      11.0,   285.0,  23,   1,  1200, "sits between two clusters"),
    ("BRAND_NEW_SIM",             9.0,   250.0,  20,   1,    30, "AON=30d -> must be refused"),
    ("CORPORATE_IOT_SIM",       950.0,   410.0,  30,   1,  1800, "extreme usage -> fence must cap"),
    ("CREDIT_ADJUSTED",           6.0,   -45.0,  18,   1,   800, "negative revenue -> must floor"),
]

raw = pd.DataFrame(
    [
        {
            "case_id": cid,
            "average_data_usage": d,
            "average_revenue": r,
            "average_dou": u,
            "pack_flag": p,
            "aon": a,
            "rationale": why,
        }
        for cid, d, r, u, p, a, why in CASES
    ]
)

scorer = SegmentScorer()
scored = scorer.score(raw.drop(columns=["case_id", "rationale"]))
scored.insert(0, "case_id", raw["case_id"].to_numpy())
scored["rationale"] = raw["rationale"].to_numpy()

cols = [
    "case_id", "segment", "segment_confidence", "runner_up_segment",
    "runner_up_confidence", "churn_probability", "churn_prediction",
    "risk_band", "offer",
]

print("=" * 118)
print("  SAMPLE TEST CASES  -  full scoring path on ten unseen subscribers")
print("=" * 118)
print(scored[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n" + "=" * 118)
print("  EDGE CASE VERIFICATION")
print("=" * 118)
new = scored[scored.case_id == "BRAND_NEW_SIM"].iloc[0]
print(f"  BRAND_NEW_SIM      aon=30  -> segment = '{new['segment']}'  (AON rule fired: "
      f"{'PASS' if 'UNSEGMENTED' in new['segment'] else 'FAIL'})")

iot_in = raw[raw.case_id == "CORPORATE_IOT_SIM"].average_data_usage.iloc[0]
iot_out = scored[scored.case_id == "CORPORATE_IOT_SIM"].average_data_usage.iloc[0]
cap = scorer.caps["average_data_usage"]["high"]
print(f"  CORPORATE_IOT_SIM  data {iot_in:.1f} GB -> capped to {iot_out:.2f} GB at the "
      f"training fence {cap:.2f}  ({'PASS' if abs(iot_out-cap) < 1e-6 else 'FAIL'})")

cr_out = scored[scored.case_id == "CREDIT_ADJUSTED"].average_revenue.iloc[0]
print(f"  CREDIT_ADJUSTED    revenue -45.0 -> floored to {cr_out:.2f}  "
      f"({'PASS' if cr_out == 0.0 else 'FAIL'})")

bl = scored[scored.case_id == "BORDERLINE_MIDFIELD"].iloc[0]
print(f"  BORDERLINE_MIDFIELD  '{bl['segment']}' @ {bl['segment_confidence']:.4f} vs "
      f"runner-up '{bl['runner_up_segment']}' @ {bl['runner_up_confidence']:.4f}")
gap = bl["segment_confidence"] - bl["runner_up_confidence"]
verdict = "review manually" if bl["segment_confidence"] < 0.80 else "assignment is safe"
print(f"                       confidence gap = {gap:.4f} -> {verdict}")

scored[cols + ["rationale"]].to_csv(C.OUT_DIR / "sample_predictions.csv", index=False)

# feed the dashboard
with open(C.OUT_DIR / "sample_predictions.json", "w") as f:
    json.dump(scored[cols].to_dict("records"), f, indent=2, default=str)

print("\nWritten -> outputs/sample_predictions.csv")

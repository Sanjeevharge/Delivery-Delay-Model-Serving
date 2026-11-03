"""Prints one valid example order as JSON (taken from the saved test stream)."""
import json, sys
import pandas as pd
from u2.common import FEATS
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "artifacts/test_stream.csv.gz")
row = d[FEATS].iloc[0].to_dict()
row = {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}
for k in ("purchase_hour", "purchase_dow", "purchase_month", "n_items", "n_sellers", "same_state", "n_payments"):
    if row[k] is not None: row[k] = int(row[k])
print(json.dumps(row))

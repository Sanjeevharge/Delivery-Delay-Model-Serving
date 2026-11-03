#!/usr/bin/env python
"""
Delivery-delay prediction on the Olist Brazilian e-commerce dataset.

Business question
-----------------
At the moment an order is placed (and payment approved), which orders are likely
to MISS their promised delivery date, so operations can intervene early?

Usage
-----
    python delivery_delay.py --data data --out outputs
    python delivery_delay.py --data data_sample --out outputs_sample   # synthetic smoke test

Input : the Olist CSV files (Kaggle: olistbr/brazilian-ecommerce) in --data
Output: --out/report.md (all results + findings), PNG plots, CSV tables and
        test_predictions.csv (a late-risk score for every test order).
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC

import lightgbm as lgb

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
SELLER_PRIOR_STRENGTH = 10  # pseudo-orders used to smooth a seller's historical late rate

FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}
OPTIONAL_FILES = {
    "reviews": "olist_order_reviews_dataset.csv",
    "translation": "product_category_name_translation.csv",
}
ORDER_TS = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
            "order_delivered_customer_date", "order_estimated_delivery_date"]

# Everything here is only known AFTER delivery (or after the order's fate is settled) -> excluded.
LEAKAGE_COLUMNS = {
    "order_delivered_customer_date": "The outcome itself (actual delivery timestamp).",
    "order_delivered_carrier_date": "Carrier hand-over happens days after purchase and partly reveals the outcome.",
    "order_status": "Final status (delivered / canceled / unavailable) is only known at the end of the order.",
    "delivery_days, delay_days": "Derived from the delivery timestamp - used for EDA only, never as features.",
    "reviews table (review_score, comments, review dates)": "Written after delivery and strongly driven by lateness.",
}
KNOWN_AT_PREDICTION_TIME = ("order_purchase_timestamp, order_approved_at (minutes/hours after purchase), "
                            "order_estimated_delivery_date (the promise shown to the customer), "
                            "items / products / freight / payment / customer & seller location, "
                            "seller history from orders delivered BEFORE this purchase.")

NUM_FEATURES = ["purchase_hour", "purchase_dow", "purchase_month", "promised_days", "approval_hours",
                "shipping_limit_days", "n_items", "n_sellers", "price_total", "freight_total",
                "freight_ratio", "weight_total_g", "volume_total_cm3", "distance_km", "same_state",
                "installments", "n_payments", "seller_hist_n", "seller_hist_late_rate"]
CAT_FEATURES = ["customer_state", "seller_state", "category", "payment_type"]

REGIONS = {"North": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
           "Northeast": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
           "Center-West": ["DF", "GO", "MT", "MS"],
           "Southeast": ["ES", "MG", "RJ", "SP"],
           "South": ["PR", "RS", "SC"]}
STATE_TO_REGION = {s: r for r, states in REGIONS.items() for s in states}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ----------------------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(f"[delivery-delay] {msg}", flush=True)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    d = df.reset_index()
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def score(model, X) -> np.ndarray:
    """Probability if the model has one, otherwise a ranking score (SVM)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def rank_metrics(y, s) -> dict:
    return {"PR_AUC": average_precision_score(y, s), "ROC_AUC": roc_auc_score(y, s)}


def threshold_metrics(y, p, t, c_fp, c_fn) -> dict:
    flag = p >= t
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    return {"threshold": t, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "flagged_pct": flag.mean() * 100,
            "false_alarms_per_1000": fp / len(y) * 1000, "missed_delays_per_1000": fn / len(y) * 1000,
            "cost_per_1000": (c_fp * fp + c_fn * fn) / len(y) * 1000}


# ----------------------------------------------------------------------------- 1. load
def load_data(data_dir: Path) -> dict:
    missing = [f for f in FILES.values() if not (data_dir / f).exists()]
    if missing:
        raise SystemExit(f"Missing files in {data_dir.resolve()}:\n  " + "\n  ".join(missing)
                         + "\nDownload: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce")
    d = {k: pd.read_csv(data_dir / f) for k, f in FILES.items()}
    for k, f in OPTIONAL_FILES.items():
        d[k] = pd.read_csv(data_dir / f) if (data_dir / f).exists() else None
    for c in ORDER_TS:
        d["orders"][c] = pd.to_datetime(d["orders"][c], errors="coerce")
    d["items"]["shipping_limit_date"] = pd.to_datetime(d["items"]["shipping_limit_date"], errors="coerce")
    return d


# ----------------------------------------------------------------------------- 2. data quality
def data_quality_report(d: dict):
    rows = []
    for name, df in d.items():
        if df is None:
            continue
        miss = df.isna().mean()
        rows.append({"table": name, "rows": len(df), "columns": df.shape[1],
                     "duplicate_rows": int(df.duplicated().sum()),
                     "missing_cells_pct": float(df.isna().values.mean() * 100),
                     "columns_with_missing": ", ".join(f"{c} ({v:.1%})" for c, v in miss[miss > 0].items()) or "-"})
    tables = pd.DataFrame(rows).set_index("table")

    o = d["orders"]
    pur = o.order_purchase_timestamp
    checks = {
        "approved before purchase": o.order_approved_at < pur,
        "handed to carrier before purchase": o.order_delivered_carrier_date < pur,
        "delivered before purchase": o.order_delivered_customer_date < pur,
        "delivered before carrier hand-over": o.order_delivered_customer_date < o.order_delivered_carrier_date,
        "estimated date before purchase": o.order_estimated_delivery_date < pur,
        "status 'delivered' but no delivery date": (o.order_status == "delivered") & o.order_delivered_customer_date.isna(),
        "delivery date but status not 'delivered'": (o.order_status != "delivered") & o.order_delivered_customer_date.notna(),
        "orders without items": ~o.order_id.isin(d["items"].order_id),
        "orders without payment rows": ~o.order_id.isin(d["payments"].order_id),
        "duplicate order_id": o.order_id.duplicated(),
    }
    ts = pd.DataFrame({"rows": {k: int(v.sum()) for k, v in checks.items()}})
    ts["pct_of_orders"] = ts["rows"] / len(o) * 100
    return tables, ts


# ----------------------------------------------------------------------------- 3. joins + target
def build_dataset(d: dict):
    o = d["orders"].copy()
    status = o.order_status.value_counts().rename("orders").to_frame()
    funnel = {"all orders": len(o)}

    impossible = ((o.order_delivered_customer_date < o.order_purchase_timestamp)
                  | (o.order_approved_at < o.order_purchase_timestamp)
                  | (o.order_estimated_delivery_date < o.order_purchase_timestamp))
    o = o[~impossible]
    funnel["minus impossible timestamps"] = len(o)

    o = o[(o.order_status == "delivered") & o.order_delivered_customer_date.notna()]
    funnel["delivered and with a delivery date"] = len(o)

    # Right-censoring: orders bought just before the data ends are only present if they were
    # delivered quickly, which would bias the late rate down. Drop the last 45 days.
    cutoff = o.order_delivered_customer_date.max() - pd.Timedelta(days=45)
    start = pd.Timestamp("2017-01-01")  # 2016 has only a few hundred orders
    o = o[(o.order_purchase_timestamp >= start) & (o.order_purchase_timestamp <= cutoff)].copy()
    funnel[f"purchased between {start.date()} and {cutoff.date()}"] = len(o)

    # TARGET: delivered (calendar day) later than the promised (estimated) day.
    o["late"] = (o.order_delivered_customer_date.dt.normalize()
                 > o.order_estimated_delivery_date.dt.normalize()).astype(int)
    o["delivery_days"] = (o.order_delivered_customer_date - o.order_purchase_timestamp).dt.total_seconds() / 86400
    o["delay_days"] = (o.order_delivered_customer_date - o.order_estimated_delivery_date).dt.total_seconds() / 86400

    # ---- items + products (order level)
    it = d["items"].merge(d["products"], on="product_id", how="left")
    cat = it["product_category_name"]
    if d.get("translation") is not None:
        tr = d["translation"].set_index("product_category_name")["product_category_name_english"]
        cat = cat.map(tr).fillna(cat)
    it["category"] = cat.fillna("unknown")
    it["volume_cm3"] = it.product_length_cm * it.product_height_cm * it.product_width_cm
    it = it.sort_values(["order_id", "order_item_id"])
    agg = it.groupby("order_id").agg(
        n_items=("order_item_id", "count"), n_sellers=("seller_id", "nunique"),
        price_total=("price", "sum"), freight_total=("freight_value", "sum"),
        weight_total_g=("product_weight_g", "sum"), n_w=("product_weight_g", "count"),
        volume_total_cm3=("volume_cm3", "sum"), n_v=("volume_cm3", "count"),
        shipping_limit=("shipping_limit_date", "max"),
        seller_id=("seller_id", "first"), category=("category", "first"))
    agg.loc[agg.n_w == 0, "weight_total_g"] = np.nan
    agg.loc[agg.n_v == 0, "volume_total_cm3"] = np.nan
    agg = agg.drop(columns=["n_w", "n_v"])

    # ---- payments (type of the largest payment)
    pay = (d["payments"].sort_values(["order_id", "payment_value"], ascending=[True, False])
           .groupby("order_id").agg(payment_type=("payment_type", "first"),
                                    installments=("payment_installments", "max"),
                                    n_payments=("payment_sequential", "count")))

    # ---- geolocation: one lat/lng per zip prefix (raw file has many rows per prefix + outliers)
    g = d["geolocation"]
    g = g[g.geolocation_lat.between(-34, 6) & g.geolocation_lng.between(-74, -34)]
    geo = g.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()

    cust = d["customers"][["customer_id", "customer_zip_code_prefix", "customer_state"]]
    sell = d["sellers"][["seller_id", "seller_zip_code_prefix", "seller_state"]]
    n_before = len(o)
    df = (o.merge(agg, left_on="order_id", right_index=True, how="left")
           .merge(pay, left_on="order_id", right_index=True, how="left")
           .merge(cust, on="customer_id", how="left")
           .merge(sell, on="seller_id", how="left")
           .merge(geo.rename(columns={"geolocation_lat": "cust_lat", "geolocation_lng": "cust_lng"}),
                  left_on="customer_zip_code_prefix", right_index=True, how="left")
           .merge(geo.rename(columns={"geolocation_lat": "sell_lat", "geolocation_lng": "sell_lng"}),
                  left_on="seller_zip_code_prefix", right_index=True, how="left"))
    assert len(df) == n_before, "a join duplicated rows"

    # ---- features known at purchase / approval time
    ts = df.order_purchase_timestamp
    df["purchase_hour"] = ts.dt.hour
    df["purchase_dow"] = ts.dt.dayofweek
    df["purchase_month"] = ts.dt.month
    df["promised_days"] = (df.order_estimated_delivery_date - ts).dt.total_seconds() / 86400
    df["approval_hours"] = (df.order_approved_at - ts).dt.total_seconds() / 3600
    df["shipping_limit_days"] = (df.shipping_limit - ts).dt.total_seconds() / 86400
    df["freight_ratio"] = df.freight_total / (df.price_total + 1e-6)
    df["distance_km"] = haversine_km(df.cust_lat, df.cust_lng, df.sell_lat, df.sell_lng)
    df["same_state"] = (df.customer_state == df.seller_state).astype(int)
    df["customer_region"] = df.customer_state.map(STATE_TO_REGION).fillna("unknown")
    return df, status, pd.Series(funnel, name="orders").to_frame()


def add_seller_history(df: pd.DataFrame, prior: float) -> pd.DataFrame:
    """Seller late rate using ONLY orders that were delivered before this order was purchased."""
    ev = (df.loc[df.seller_id.notna(), ["seller_id", "order_delivered_customer_date", "late"]]
          .sort_values("order_delivered_customer_date").copy())
    ev["cum_n"] = ev.groupby("seller_id").cumcount() + 1
    ev["cum_late"] = ev.groupby("seller_id")["late"].cumsum()
    ev = ev.rename(columns={"order_delivered_customer_date": "known_at"})[["seller_id", "known_at", "cum_n", "cum_late"]]
    q = df.loc[df.seller_id.notna(), ["order_id", "seller_id", "order_purchase_timestamp"]].sort_values("order_purchase_timestamp")
    m = pd.merge_asof(q, ev, left_on="order_purchase_timestamp", right_on="known_at", by="seller_id",
                      direction="backward", allow_exact_matches=False)
    m["seller_hist_n"] = m.cum_n.fillna(0)
    m["seller_hist_late_rate"] = (m.cum_late.fillna(0) + prior * SELLER_PRIOR_STRENGTH) / (m.seller_hist_n + SELLER_PRIOR_STRENGTH)
    df = df.merge(m[["order_id", "seller_hist_n", "seller_hist_late_rate"]], on="order_id", how="left")
    df["seller_hist_n"] = df.seller_hist_n.fillna(0)
    df["seller_hist_late_rate"] = df.seller_hist_late_rate.fillna(prior)
    return df


def time_split(df: pd.DataFrame):
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    t1, t2 = df.order_purchase_timestamp.quantile([0.70, 0.85])
    df["split"] = np.where(df.order_purchase_timestamp < t1, "train",
                           np.where(df.order_purchase_timestamp < t2, "val", "test"))
    return df, t1, t2


# ----------------------------------------------------------------------------- 4. EDA
def eda(train: pd.DataFrame, out: Path):
    f, plots = [], []
    lr = train.late.mean()
    f.append(f"**{lr:.1%}** of delivered orders in the training period arrived after the promised date.")
    f.append(f"Median delivery takes **{train.delivery_days.median():.1f} days** (90th pct "
             f"{train.delivery_days.quantile(.9):.1f}) while the median promise is **{train.promised_days.median():.1f} days** "
             "- promises are padded, so lateness is a tail event, not the typical case.")
    late = train[train.late == 1]
    if len(late):
        f.append(f"When an order is late it is late by a median **{late.delay_days.median():.1f} days**.")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(train.delivery_days.clip(upper=60), bins=60, color="#4C72B0")
    ax[0].set(title="Actual delivery time (days, clipped at 60)", xlabel="days", ylabel="orders")
    ax[1].hist(train.delay_days.clip(-40, 40), bins=80, color="#55A868")
    ax[1].axvline(0, color="red", ls="--")
    ax[1].set(title="Delivered minus promised (days)", xlabel="> 0 means late")
    plots.append(save(fig, out / "eda_1_delivery_time.png"))

    # seasonality
    mon = train.groupby(train.order_purchase_timestamp.dt.to_period("M")).late.agg(["mean", "size"])
    wd = train.groupby("purchase_dow").late.mean()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    x = mon.index.astype(str)
    ax[0].bar(x, mon["size"], color="lightgray"); ax[0].set_ylabel("orders"); ax[0].tick_params(axis="x", rotation=90)
    ax2 = ax[0].twinx(); ax2.plot(x, mon["mean"], color="crimson", marker="o"); ax2.set_ylabel("late rate")
    ax[0].set_title("Monthly volume (bars) and late rate (line)")
    ax[1].bar([DAYS[i] for i in wd.index], wd.values, color="#8172B2"); ax[1].set_title("Late rate by purchase weekday")
    plots.append(save(fig, out / "eda_2_seasonality.png"))
    big = mon[mon["size"] >= max(30, mon["size"].median() * 0.2)]
    if len(big) > 1:
        f.append(f"**Seasonality:** late rate peaks for orders bought in **{big['mean'].idxmax()}** "
                 f"({big['mean'].max():.1%}) vs a low of {big['mean'].min():.1%} in {big['mean'].idxmin()} - "
                 "peak periods (Black Friday / holidays, carnival) overload the logistics network.")

    # region & distance
    st = train.groupby("customer_state").late.agg(["mean", "size"])
    st = st[st["size"] >= 30].sort_values("mean")
    dq = train.groupby(pd.qcut(train.distance_km, 5, duplicates="drop"), observed=True).late.mean()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].barh(st.index, st["mean"], color="#C44E52"); ax[0].axvline(lr, color="k", ls="--")
    ax[0].set_title("Late rate by customer state (>=30 orders)")
    ax[1].bar([f"{iv.left:.0f}-{iv.right:.0f}" for iv in dq.index], dq.values, color="#DD8452")
    ax[1].set(title="Late rate by seller-customer distance (km quintile)"); ax[1].tick_params(axis="x", rotation=30)
    plots.append(save(fig, out / "eda_3_region_distance.png"))
    if len(st) >= 4:
        worst = ", ".join(f"{s} ({v:.1%})" for s, v in st["mean"].tail(3)[::-1].items())
        best = ", ".join(f"{s} ({v:.1%})" for s, v in st["mean"].head(3).items())
        f.append(f"**Region:** worst destination states are {worst}; best are {best}.")
    if len(dq) > 1:
        f.append(f"**Distance:** late rate goes from {dq.iloc[0]:.1%} in the nearest distance quintile "
                 f"to {dq.iloc[-1]:.1%} in the farthest.")

    # category
    ca = train.groupby("category").late.agg(["mean", "size"]).sort_values("size", ascending=False).head(15).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ca.index, ca["mean"], color="#64B5CD"); ax.axvline(lr, color="k", ls="--")
    ax.set_title("Late rate - top 15 categories by volume")
    plots.append(save(fig, out / "eda_4_category.png"))
    if len(ca) >= 3:
        f.append(f"**Category:** among the 15 largest categories, **{ca.index[-1]}** is late most often "
                 f"({ca['mean'].iloc[-1]:.1%}) and **{ca.index[0]}** least ({ca['mean'].iloc[0]:.1%}).")

    # seller
    se = train.groupby("seller_id").late.agg(["mean", "size", "sum"])
    se30 = se[se["size"] >= 30]
    if len(se30) >= 5:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(se30["mean"], bins=30, color="#937860"); ax.axvline(lr, color="k", ls="--")
        ax.set(title="Late rate of sellers with >= 30 orders", xlabel="seller late rate", ylabel="sellers")
        plots.append(save(fig, out / "eda_5_sellers.png"))
        top = se30.sort_values("mean", ascending=False)
        k = max(1, len(top) // 10)
        share = top["sum"].head(k).sum() / max(se30["sum"].sum(), 1)
        f.append(f"**Sellers:** of {len(se30)} sellers with >= 30 orders, {(se30['mean'] > 2 * lr).mean():.0%} are late "
                 f"more than twice the average; the worst 10% of them cause {share:.0%} of those sellers' late orders "
                 "-> seller history is a strong signal.")
    return f, plots


# ----------------------------------------------------------------------------- 5. models
def linear_prep(num, cat):
    t = []
    if num:
        t.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                                   ("sc", StandardScaler())]), num))
    if cat:
        t.append(("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                                   ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30))]), cat))
    return ColumnTransformer(t, verbose_feature_names_out=False)


def tree_prep(num, cat):
    t = []
    if num:
        t.append(("num", SimpleImputer(strategy="median", keep_empty_features=True), num))
    if cat:
        t.append(("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                                   ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat))
    return ColumnTransformer(t, verbose_feature_names_out=False)


def make_lgbm(num, cat, pos_weight):
    return Pipeline([("prep", tree_prep(num, cat)),
                     ("clf", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                                min_child_samples=50, subsample=0.8, subsample_freq=1,
                                                colsample_bytree=0.8, scale_pos_weight=pos_weight,
                                                random_state=RANDOM_STATE, verbose=-1))])


def make_models(num, cat, pos_weight, with_svm=True):
    m = {
        "LogisticRegression": Pipeline([("prep", linear_prep(num, cat)),
                                        ("clf", LogisticRegression(class_weight="balanced", max_iter=3000))]),
        "RandomForest": Pipeline([("prep", tree_prep(num, cat)),
                                  ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                                 class_weight="balanced_subsample", n_jobs=-1,
                                                                 random_state=RANDOM_STATE))]),
        "LightGBM": make_lgbm(num, cat, pos_weight),
    }
    if with_svm:
        m["LinearSVM"] = Pipeline([("prep", linear_prep(num, cat)),
                                   ("clf", LinearSVC(C=0.05, class_weight="balanced", dual=False, max_iter=5000))])
    return m


def shap_explain(pipe, X: pd.DataFrame, out: Path):
    import shap
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    Xt = pd.DataFrame(prep.transform(X), columns=names)
    sample = Xt.sample(min(3000, len(Xt)), random_state=RANDOM_STATE)
    sv = shap.TreeExplainer(clf).shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
    rows = []
    for c in imp.index[:12]:
        if c in CAT_FEATURES:
            direction = "categorical (effect depends on the level)"
        else:
            x, s = sample[c].values.astype(float), sv[:, names.index(c)]
            r = np.corrcoef(x, s)[0, 1] if np.nanstd(x) > 0 else np.nan
            direction = ("higher value -> higher late risk" if r > 0.1 else
                         "higher value -> lower late risk" if r < -0.1 else "non-linear / weak")
        rows.append({"feature": c, "mean_abs_shap": imp[c], "direction": direction})
    shap.summary_plot(sv, sample, show=False, max_display=15)
    plt.gcf().set_size_inches(9, 6)
    save(plt.gcf(), out / "shap_summary.png")
    return pd.DataFrame(rows).set_index("feature")


def group_report(g: pd.DataFrame) -> pd.Series:
    y, f, p = g.late.values, g.flag.values, g.prob.values
    tp = int(((f == 1) & (y == 1)).sum())
    return pd.Series({"orders": len(g), "late_rate": y.mean(), "flagged_rate": f.mean(),
                      "recall": tp / y.sum() if y.sum() else np.nan,
                      "precision": tp / f.sum() if f.sum() else np.nan,
                      "PR_AUC": average_precision_score(y, p) if 0 < y.sum() < len(y) else np.nan,
                      "missed_delays": int(((f == 0) & (y == 1)).sum()),
                      "false_alarms": int(((f == 1) & (y == 0)).sum())})


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="folder with the Olist CSV files")
    ap.add_argument("--out", default="outputs", help="folder for report, plots and predictions")
    ap.add_argument("--cost-false-alarm", type=float, default=1.0, help="cost of flagging an order that would be on time")
    ap.add_argument("--cost-missed-delay", type=float, default=5.0, help="cost of NOT flagging an order that ends up late")
    ap.add_argument("--no-svm", action="store_true")
    ap.add_argument("--no-shap", action="store_true")
    a = ap.parse_args()
    c_fp, c_fn = a.cost_false_alarm, a.cost_missed_delay
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R = []  # report lines

    log(f"loading CSVs from {Path(a.data).resolve()}")
    d = load_data(Path(a.data))

    log("data-quality report")
    dq_tables, dq_ts = data_quality_report(d)
    dq_tables.to_csv(out / "data_quality_tables.csv"); dq_ts.to_csv(out / "data_quality_timestamps.csv")

    log("joining tables, defining target, engineering features")
    df, status, funnel = build_dataset(d)
    df, t1, t2 = time_split(df)
    prior = df.loc[df.split == "train", "late"].mean()
    df = add_seller_history(df, prior)
    train, val, test = (df[df.split == s].copy() for s in ("train", "val", "test"))
    FEATS = NUM_FEATURES + CAT_FEATURES
    Xtr, Xva, Xte = train[FEATS], val[FEATS], test[FEATS]
    ytr, yva, yte = train.late.values, val.late.values, test.late.values
    pos_weight = (1 - ytr.mean()) / ytr.mean()
    log(f"train {len(train):,} | val {len(val):,} | test {len(test):,} | late rate train {ytr.mean():.1%}")

    log("EDA (training period only)")
    findings, eda_plots = eda(train, out)

    # ---- baseline
    log("baseline + models")
    state_rate = train.groupby("customer_state").late.mean()
    base = {s: part.customer_state.map(state_rate).fillna(prior).values for s, part in (("val", val), ("test", test))}
    comp = {"Baseline: late rate of destination state": {**{f"val_{k}": v for k, v in rank_metrics(yva, base["val"]).items()},
                                                          **{f"test_{k}": v for k, v in rank_metrics(yte, base["test"]).items()},
                                                          "fit_seconds": 0.0}}
    fitted = {}
    for name, m in make_models(NUM_FEATURES, CAT_FEATURES, pos_weight, not a.no_svm).items():
        s = time.time(); m.fit(Xtr, ytr); fit_s = time.time() - s
        fitted[name] = m
        comp[name] = {**{f"val_{k}": v for k, v in rank_metrics(yva, score(m, Xva)).items()},
                      **{f"test_{k}": v for k, v in rank_metrics(yte, score(m, Xte)).items()},
                      "fit_seconds": fit_s}
        log(f"  {name:20s} val PR-AUC {comp[name]['val_PR_AUC']:.3f}  ({fit_s:.1f}s)")

    # ---- feature selection
    log("feature selection: L1 + permutation importance")
    l1 = Pipeline([("prep", linear_prep(NUM_FEATURES, CAT_FEATURES)),
                   ("clf", LogisticRegression(penalty="l1", solver="liblinear", C=0.02,
                                              class_weight="balanced", max_iter=3000))]).fit(Xtr, ytr)
    l1_names = l1.named_steps["prep"].get_feature_names_out()
    l1_coef = l1.named_steps["clf"].coef_[0]
    l1_dropped = [n for n, c in zip(l1_names, l1_coef) if c == 0]
    comp["LogisticRegression + L1 selection"] = {
        **{f"val_{k}": v for k, v in rank_metrics(yva, score(l1, Xva)).items()},
        **{f"test_{k}": v for k, v in rank_metrics(yte, score(l1, Xte)).items()}, "fit_seconds": np.nan}

    pi = permutation_importance(fitted["LightGBM"], Xva, yva, scoring="average_precision",
                                n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    perm = pd.DataFrame({"importance_mean": pi.importances_mean, "importance_std": pi.importances_std},
                        index=FEATS).sort_values("importance_mean", ascending=False)
    perm.to_csv(out / "permutation_importance.csv")
    keep = perm.index[perm.importance_mean > 0].tolist()
    sel_num = [c for c in NUM_FEATURES if c in keep]; sel_cat = [c for c in CAT_FEATURES if c in keep]
    lgb_sel = make_lgbm(sel_num, sel_cat, pos_weight).fit(train[sel_num + sel_cat], ytr)
    fitted["LightGBM + permutation selection"] = lgb_sel
    comp["LightGBM + permutation selection"] = {
        **{f"val_{k}": v for k, v in rank_metrics(yva, score(lgb_sel, val[sel_num + sel_cat])).items()},
        **{f"test_{k}": v for k, v in rank_metrics(yte, score(lgb_sel, test[sel_num + sel_cat])).items()},
        "fit_seconds": np.nan}
    comp = pd.DataFrame(comp).T
    comp.to_csv(out / "model_comparison.csv")

    # ---- pick final model on VALIDATION PR-AUC (test is only reported, never used to choose)
    candidates = {k: v for k, v in fitted.items()}
    feats_of = {k: FEATS for k in candidates}; feats_of["LightGBM + permutation selection"] = sel_num + sel_cat
    best = max(candidates, key=lambda k: comp.loc[k, "val_PR_AUC"])
    final, ffeats = candidates[best], feats_of[best]
    log(f"final model: {best}")

    # ---- calibration (isotonic, fit on validation) + threshold by cost on validation
    s_va, s_te = score(final, val[ffeats]), score(final, test[ffeats])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s_va, yva)
    p_va, p_te = iso.predict(s_va), iso.predict(s_te)
    has_proba = hasattr(final, "predict_proba")
    calib = pd.DataFrame({
        "Brier (lower is better)": [brier_score_loss(yte, np.clip(s_te, 0, 1)) if has_proba else np.nan,
                                    brier_score_loss(yte, p_te)],
        "mean predicted prob": [s_te.mean() if has_proba else np.nan, p_te.mean()],
        "actual late rate": [yte.mean(), yte.mean()]}, index=["raw (class-weighted) model", "isotonic-calibrated"])

    grid = np.round(np.arange(0.02, 0.96, 0.01), 2)
    val_cost = [threshold_metrics(yva, p_va, t, c_fp, c_fn)["cost_per_1000"] for t in grid]
    thr = float(grid[int(np.argmin(val_cost))])
    test_curve = pd.DataFrame([threshold_metrics(yte, p_te, t, c_fp, c_fn) for t in grid]).set_index("threshold")
    op = threshold_metrics(yte, p_te, thr, c_fp, c_fn)
    do_nothing = c_fn * yte.sum() / len(yte) * 1000
    flag_all = c_fp * (1 - yte).sum() / len(yte) * 1000
    final_metrics = {**rank_metrics(yte, p_te), **op}

    # ---- plots: PR curves, calibration, cost curve
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, m in fitted.items():
        pr, rc, _ = precision_recall_curve(yte, score(m, test[feats_of[k]]))
        ax.plot(rc, pr, label=f"{k} (AP {comp.loc[k, 'test_PR_AUC']:.3f})", lw=2 if k == best else 1)
    pr, rc, _ = precision_recall_curve(yte, base["test"])
    ax.plot(rc, pr, "k--", label=f"state baseline (AP {comp.iloc[0]['test_PR_AUC']:.3f})")
    ax.axhline(yte.mean(), color="gray", ls=":", label="random")
    ax.set(xlabel="recall", ylabel="precision", title="Precision-recall on the TEST period"); ax.legend(fontsize=8)
    pr_plot = save(fig, out / "model_pr_curves.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    if has_proba:
        fx, fy = calibration_curve(yte, np.clip(s_te, 0, 1), n_bins=10, strategy="quantile")
        ax.plot(fy, fx, "o-", label="raw (class-weighted)")
    fx, fy = calibration_curve(yte, p_te, n_bins=10, strategy="quantile")
    ax.plot(fy, fx, "o-", label="calibrated")
    ax.plot([0, 1], [0, 1], "k:"); ax.set(xlabel="predicted probability", ylabel="observed late rate",
                                          title="Reliability diagram (test)"); ax.legend()
    cal_plot = save(fig, out / "calibration.png")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(test_curve.index, test_curve.cost_per_1000, lw=2, label="model")
    ax[0].axhline(do_nothing, color="r", ls="--", label="do nothing (flag none)")
    ax[0].axhline(flag_all, color="g", ls="--", label="flag every order")
    ax[0].axvline(thr, color="k", ls=":", label=f"chosen threshold {thr:.2f}")
    ax[0].set(xlabel="probability threshold", ylabel=f"cost per 1000 orders (FA={c_fp:g}, miss={c_fn:g})",
              title="Cost curve (test)"); ax[0].legend(fontsize=8)
    ax[1].plot(test_curve.false_alarms_per_1000, test_curve.missed_delays_per_1000, lw=2)
    ax[1].scatter([op["false_alarms_per_1000"]], [op["missed_delays_per_1000"]], color="k", zorder=5, label="chosen")
    ax[1].set(xlabel="false alarms per 1000 orders", ylabel="missed delays per 1000 orders",
              title="Trade-off: false alarms vs missed delays"); ax[1].legend()
    cost_plot = save(fig, out / "cost_curve.png")

    # ---- error analysis
    log("error analysis + SHAP")
    te = test.copy(); te["prob"] = p_te; te["flag"] = (p_te >= thr).astype(int)
    by_region = te.groupby("customer_region").apply(group_report).sort_values("orders", ascending=False)
    by_state = te.groupby("customer_state").apply(group_report)
    by_state = by_state[by_state.orders >= 50].sort_values("orders", ascending=False)
    te["seller_group"] = pd.cut(te.seller_hist_n, [-1, 0, 9, 49, np.inf],
                                labels=["new (0 past orders)", "small (1-9)", "medium (10-49)", "large (50+)"])
    by_seller = te.groupby("seller_group", observed=True).apply(group_report)
    for nm, t in (("region", by_region), ("state", by_state), ("seller_group", by_seller)):
        t.to_csv(out / f"error_by_{nm}.csv")
    te[["order_id", "order_purchase_timestamp", "customer_state", "seller_id", "category",
        "prob", "flag", "late"]].sort_values("prob", ascending=False).to_csv(out / "test_predictions.csv", index=False)

    shap_tbl, shap_note = None, ""
    if not a.no_shap:
        try:
            shap_model = final if best.startswith("LightGBM") else fitted["LightGBM"]
            shap_feats = ffeats if best.startswith("LightGBM") else FEATS
            shap_tbl = shap_explain(shap_model, test[shap_feats], out)
            shap_note = "" if best.startswith("LightGBM") else f" (computed on LightGBM; final model is {best})"
        except Exception as e:  # shap is optional
            shap_note = f" - SHAP failed: {e}"

    # ---- report
    R += ["# Delivery-delay prediction - Olist", "",
          f"_Generated in {time.time() - t0:.0f}s from `{Path(a.data).resolve()}`_", "",
          "## 1. Data-quality report", "", md_table(dq_tables, "{:.2f}"), "",
          "Timestamp / consistency checks on `orders`:", "", md_table(dq_ts, "{:.2f}"), "",
          "## 2. Target and population", "",
          "**Target:** `late = 1` if the order's delivery **calendar day** is after `order_estimated_delivery_date` "
          "(the promise shown to the customer at checkout).", "",
          "**Cancelled / unavailable / undelivered orders are excluded** from training and evaluation: their "
          "outcome is not a delivery, so 'late vs on time' is undefined, and labelling them late would teach the "
          "model to predict cancellations. They are reported here so the exclusion is visible:", "",
          md_table(status), "", "Row funnel:", "", md_table(funnel), "",
          "Orders bought in the final 45 days are dropped because only the fast ones have a delivery date "
          "(right-censoring would make the late rate look too low).", "",
          "## 3. Leakage audit", "", "Excluded because they are only known after delivery:", "",
          *[f"- `{k}` - {v}" for k, v in LEAKAGE_COLUMNS.items()], "",
          f"Used (known when the order is placed/approved): {KNOWN_AT_PREDICTION_TIME}", "",
          "## 4. EDA findings (training period only)", "", *[f"- {x}" for x in findings], "",
          *[f"![]({p})" for p in eda_plots], "",
          "## 5. Split", "",
          f"Chronological by purchase date - train < {t1.date()} <= validation < {t2.date()} <= test. "
          f"Sizes: {len(train):,} / {len(val):,} / {len(test):,}. Late rate: "
          f"{ytr.mean():.1%} / {yva.mean():.1%} / {yte.mean():.1%}. No random split.", "",
          "## 6. Baseline vs models (same split)", "",
          "Imbalance handled with class weights (`balanced`, `scale_pos_weight`). Model chosen on **validation** PR-AUC; "
          "test columns are reported for transparency only.", "",
          md_table(comp), "", f"![]({pr_plot})", "",
          "## 7. Feature selection (before / after)", "",
          f"- **L1 logistic regression** (C=0.02) zeroed {len(l1_dropped)} of {len(l1_names)} encoded columns. "
          f"Val PR-AUC {comp.loc['LogisticRegression', 'val_PR_AUC']:.3f} -> "
          f"{comp.loc['LogisticRegression + L1 selection', 'val_PR_AUC']:.3f}.",
          f"- **Permutation importance** on LightGBM (validation, PR-AUC drop) kept {len(keep)} of {len(FEATS)} features. "
          f"Val PR-AUC {comp.loc['LightGBM', 'val_PR_AUC']:.3f} -> {comp.loc['LightGBM + permutation selection', 'val_PR_AUC']:.3f}. "
          f"Dropped: {', '.join(c for c in FEATS if c not in keep) or 'none'}.", "",
          md_table(perm.head(15), "{:.4f}"), "",
          f"## 8. Final model: **{best}** - calibration and operating point", "",
          md_table(calib, "{:.4f}"), "", f"![]({cal_plot})", "",
          f"Costs: false alarm = {c_fp:g}, missed delay = {c_fn:g}. Threshold chosen on validation by minimum cost: "
          f"**{thr:.2f}** (theory for calibrated probabilities: {c_fp / (c_fp + c_fn):.2f}).", "",
          "Test-period results at that threshold:", "",
          md_table(pd.Series(final_metrics, name="value").to_frame(), "{:.3f}"), "",
          f"Cost per 1000 orders: model **{op['cost_per_1000']:.1f}** vs do-nothing {do_nothing:.1f} vs flag-everything {flag_all:.1f}.", "",
          f"![]({cost_plot})", "",
          "## 9. Error analysis (test)", "", "By region:", "", md_table(by_region), "",
          "By seller group (past delivered orders at purchase time):", "", md_table(by_seller), "",
          "By state (>= 50 test orders):", "", md_table(by_state), "",
          f"## 10. SHAP - top drivers{shap_note}", ""]
    if shap_tbl is not None:
        R += [md_table(shap_tbl, "{:.4f}"), "", "![](shap_summary.png)", ""]
    (out / "report.md").write_text("\n".join(R), encoding="utf-8")

    log(f"done in {time.time() - t0:.0f}s")
    print("\n================ SUMMARY ================")
    print(comp[["val_PR_AUC", "test_PR_AUC", "test_ROC_AUC"]].round(3).to_string())
    print(f"\nFinal model: {best} | threshold {thr:.2f}")
    print(f"Test: PR-AUC {final_metrics['PR_AUC']:.3f}  ROC-AUC {final_metrics['ROC_AUC']:.3f}  "
          f"precision {op['precision']:.2f}  recall {op['recall']:.2f}  flagged {op['flagged_pct']:.1f}%")
    print(f"Cost/1000 orders: model {op['cost_per_1000']:.1f} | do nothing {do_nothing:.1f} | flag all {flag_all:.1f}")
    print(f"\nFull report: {(out / 'report.md').resolve()}")


if __name__ == "__main__":
    main()


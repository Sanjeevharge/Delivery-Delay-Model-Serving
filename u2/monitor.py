"""Monitoring job: feature drift (PSI, KS), prediction drift, data quality, rolling PR-AUC on delayed labels.

The test stream is replayed as consecutive batches of 'production' traffic.
  --scenario clean : traffic as-is (should raise no / few alerts)
  --scenario drift : SYNTHETIC drift injected from batch `drift_start_batch` (see inject_drift)

    python -m u2.monitor --artifacts artifacts_sample --scenario drift --out monitor_out
"""
import argparse, json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score

from u2.common import CAT_FEATURES, FEATS, NUM_FEATURES, predict_prob

EPS = 1e-4


def psi_numeric(ref, cur, bins=10):
    ref, cur = pd.Series(ref).dropna().values, pd.Series(cur).dropna().values
    if len(ref) == 0 or len(cur) == 0:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref, cur, min_share=0.03):
    ref, cur = pd.Series(ref).fillna("missing").astype(str), pd.Series(cur).fillna("missing").astype(str)
    share = ref.value_counts(normalize=True)
    keep = set(share[share >= min_share].index)
    f = lambda s: s.where(s.isin(keep), "__other__")
    r = f(ref).value_counts(normalize=True)
    c = f(cur).value_counts(normalize=True).reindex(r.index.union(f(cur).unique()), fill_value=0)
    r = r.reindex(c.index, fill_value=0)
    r, c = np.clip(r.values, EPS, None), np.clip(c.values, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def inject_drift(b: pd.DataFrame, rng) -> pd.DataFrame:
    """SYNTHETIC drift: longer routes, shorter promises, pricier freight, unseen categories,
    missing weights, a worse seller mix, and concept drift (labels partly re-drawn at 3x the base rate)."""
    b = b.copy()
    b["distance_km"] *= 1.8
    b["promised_days"] *= 0.6
    b["freight_total"] *= 1.5
    b["seller_hist_late_rate"] = (b["seller_hist_late_rate"] + 0.10).clip(0, 1)
    m = rng.random(len(b)) < 0.20
    b.loc[m, "category"] = "new_category_x"
    b.loc[rng.random(len(b)) < 0.15, "weight_total_g"] = np.nan
    redraw = rng.random(len(b)) < 0.30
    b.loc[redraw, "late"] = (rng.random(redraw.sum()) < min(0.5, 3 * b["late"].mean() + 0.05)).astype(int)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--config", default="configs/monitoring.json")
    ap.add_argument("--scenario", choices=["clean", "drift"], default="clean")
    ap.add_argument("--out", default="monitor_out")
    a = ap.parse_args()
    cfg = json.load(open(a.config)); A = cfg["alerts"]
    art, out = Path(a.artifacts), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    bundle = joblib.load(art / "model_bundle.joblib")
    ref = pd.read_csv(art / "reference_train.csv.gz")
    stream = pd.read_csv(art / "test_stream.csv.gz")
    batches = [stream.iloc[ix] for ix in np.array_split(np.arange(len(stream)), cfg["n_batches"])]
    floor = A["pr_auc_floor_ratio"] * bundle["ref_val_pr_auc"]
    keyf = cfg["key_features"]
    lo_hi = {c: (r[0] - 0.1 * (r[1] - r[0]), r[1] + 0.1 * (r[1] - r[0])) for c, r in bundle["num_ranges"].items()}

    rows, alerts, pool = [], [], []
    for i, b in enumerate(batches):
        b = b.reset_index(drop=True)
        drifted = a.scenario == "drift" and i >= cfg["drift_start_batch"]
        if drifted:
            b = inject_drift(b, rng)
        b["prob"] = predict_prob(bundle, b)
        r = {"batch": i, "n": len(b), "drifted": int(drifted), "flag_rate": float((b.prob >= bundle["thr"]).mean()),
             "mean_prob": float(b.prob.mean())}

        # feature drift
        for c in keyf:
            if c in NUM_FEATURES:
                r[f"psi_{c}"] = psi_numeric(ref[c], b[c])
                ks = ks_2samp(ref[c].dropna(), b[c].dropna())
                r[f"ks_{c}"], r[f"ksp_{c}"] = float(ks.statistic), float(ks.pvalue)
            else:
                r[f"psi_{c}"] = psi_categorical(ref[c], b[c])
        # prediction drift
        r["psi_prediction"] = psi_numeric(bundle["ref_val_prob"], b.prob)
        # data quality
        r["max_null_increase"] = max(b[c].isna().mean() - bundle["ref_null_rate"][c] for c in FEATS)
        rc = [c for c in NUM_FEATURES if c not in cfg.get("range_check_exclude", [])]
        oor = np.mean([((b[c] < lo_hi[c][0]) | (b[c] > lo_hi[c][1])).mean() for c in rc])
        r["out_of_range_rate"] = float(oor)
        known = {c: set(bundle["categories"][c]) for c in CAT_FEATURES}
        r["unseen_cat_rate"] = float(max((~b[c].astype(str).isin(known[c]) & b[c].notna()).mean() for c in CAT_FEATURES))

        # alert rules
        fired = []
        for c in keyf:
            if r[f"psi_{c}"] > A["psi_feature"]:
                fired.append(f"PSI {c} {r[f'psi_{c}']:.2f} > {A['psi_feature']}")
            if c in NUM_FEATURES and r[f"ks_{c}"] > A["ks_stat"] and r[f"ksp_{c}"] < A["ks_pvalue"]:
                fired.append(f"KS {c} stat {r[f'ks_{c}']:.2f}")
        if r["psi_prediction"] > A["psi_prediction"]:
            fired.append(f"prediction PSI {r['psi_prediction']:.2f} > {A['psi_prediction']}")
        if r["max_null_increase"] > A["null_rate_increase"]:
            fired.append(f"null rate +{r['max_null_increase']:.2f}")
        if r["out_of_range_rate"] > A["out_of_range_rate"]:
            fired.append(f"out-of-range {r['out_of_range_rate']:.3f}")
        if r["unseen_cat_rate"] > A["unseen_category_rate"]:
            fired.append(f"unseen category {r['unseen_cat_rate']:.3f}")

        # delayed labels: batch j's labels become available after batch j + lag
        pool.append(b[["prob", "late"]])
        avail = pool[: max(0, i + 1 - cfg["label_lag_batches"])][-cfg["rolling_window_batches"]:]
        r["rolling_pr_auc"] = np.nan
        if avail:
            lab = pd.concat(avail)
            if lab.late.sum() >= cfg.get("min_positives_for_pr_auc", 15):
                r["rolling_pr_auc"] = float(average_precision_score(lab.late, lab.prob))
                if r["rolling_pr_auc"] < floor:
                    fired.append(f"rolling PR-AUC {r['rolling_pr_auc']:.3f} < floor {floor:.3f}")
        r["n_alerts"] = len(fired)
        alerts += [{"batch": i, "drifted": int(drifted), "rule": f} for f in fired]
        rows.append(r)

    res = pd.DataFrame(rows)
    res.to_csv(out / f"monitor_{a.scenario}.csv", index=False)
    json.dump(alerts, open(out / f"alerts_{a.scenario}.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for c in keyf:
        ax[0].plot(res.batch, res[f"psi_{c}"], label=c)
    ax[0].axhline(A["psi_feature"], color="r", ls="--"); ax[0].set(title="PSI per key feature", xlabel="batch"); ax[0].legend(fontsize=6)
    ax[1].plot(res.batch, res.rolling_pr_auc, "o-"); ax[1].axhline(floor, color="r", ls="--", label="floor")
    ax[1].set(title="Rolling PR-AUC (delayed labels)", xlabel="batch"); ax[1].legend()
    ax[2].bar(res.batch, res.n_alerts); ax[2].set(title="Alerts fired per batch", xlabel="batch")
    for x in ax:
        if a.scenario == "drift": x.axvline(cfg["drift_start_batch"] - 0.5, color="k", ls=":")
    fig.tight_layout(); fig.savefig(out / f"monitor_{a.scenario}.png", dpi=110)

    pre = [x for x in alerts if not x["drifted"]]
    post = [x for x in alerts if x["drifted"]]
    print(f"scenario={a.scenario} batches={len(res)} rows/batch~{int(res.n.mean())} PR-AUC floor={floor:.3f}")
    print(f"alerts BEFORE drift (false alarms): {len(pre)} in {len({x['batch'] for x in pre})} batches")
    if a.scenario == "drift":
        if post:
            first = min(x["batch"] for x in post)
            print(f"first alert at batch {first} = {first - cfg['drift_start_batch'] + 1} batch(es) after drift start")
            print("rules fired at that batch:", [x["rule"] for x in post if x["batch"] == first])
        else:
            print("NO ALERT fired after drift start")
    print("outputs:", out.resolve())


if __name__ == "__main__":
    main()

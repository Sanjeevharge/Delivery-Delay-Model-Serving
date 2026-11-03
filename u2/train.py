"""U2 training: re-uses the U1 pipeline code, logs every model to MLflow, saves a serving bundle.

    python -m u2.train --data data_sample --out artifacts_sample
"""
import argparse, json, platform, sys, time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression

import delivery_delay as dd
from u2.common import CAT_FEATURES, FEATS, NUM_FEATURES, predict_prob, raw_score

SEED = dd.RANDOM_STATE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--experiment", default="udaan-delivery-delay")
    ap.add_argument("--cost-false-alarm", type=float, default=1.0)
    ap.add_argument("--cost-missed-delay", type=float, default=5.0)
    ap.add_argument("--no-svm", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(a.experiment)

    d = dd.load_data(Path(a.data))
    df, _, _ = dd.build_dataset(d)
    df, t1, t2 = dd.time_split(df)
    prior = df.loc[df.split == "train", "late"].mean()
    df = dd.add_seller_history(df, prior)
    train, val, test = (df[df.split == s].copy() for s in ("train", "val", "test"))
    ytr, yva, yte = train.late.values, val.late.values, test.late.values
    pos_weight = (1 - ytr.mean()) / ytr.mean()

    results, fitted = {}, {}
    with mlflow.start_run(run_name="final-selection") as parent:
        mlflow.log_params({"seed": SEED, "n_train": len(train), "n_val": len(val), "n_test": len(test),
                           "split_t1": str(t1.date()), "split_t2": str(t2.date()),
                           "cost_false_alarm": a.cost_false_alarm, "cost_missed_delay": a.cost_missed_delay,
                           "python": platform.python_version(), "sklearn": sklearn.__version__})
        for name, m in dd.make_models(NUM_FEATURES, CAT_FEATURES, pos_weight, not a.no_svm).items():
            with mlflow.start_run(run_name=name, nested=True):
                m.fit(train[FEATS], ytr)
                rv = dd.rank_metrics(yva, raw_score(m, val[FEATS]))
                rt = dd.rank_metrics(yte, raw_score(m, test[FEATS]))
                mlflow.log_params({"model": name, "seed": SEED, **{k: str(v)[:250] for k, v in
                                   m.named_steps["clf"].get_params().items()
                                   if k in ("C", "n_estimators", "learning_rate", "num_leaves",
                                            "min_samples_leaf", "class_weight", "scale_pos_weight")}})
                mlflow.log_metrics({"val_PR_AUC": rv["PR_AUC"], "val_ROC_AUC": rv["ROC_AUC"],
                                    "test_PR_AUC": rt["PR_AUC"], "test_ROC_AUC": rt["ROC_AUC"]})
            fitted[name] = m
            results[name] = {"val_PR_AUC": rv["PR_AUC"], "val_ROC_AUC": rv["ROC_AUC"],
                             "test_PR_AUC": rt["PR_AUC"], "test_ROC_AUC": rt["ROC_AUC"]}
            print(f"  {name:20s} val PR-AUC {rv['PR_AUC']:.4f}")

        best = max(results, key=lambda k: results[k]["val_PR_AUC"])  # chosen on validation only
        final = fitted[best]
        s_va, s_te = raw_score(final, val[FEATS]), raw_score(final, test[FEATS])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s_va, yva)
        p_va, p_te = iso.predict(s_va), iso.predict(s_te)
        grid = np.round(np.arange(0.02, 0.96, 0.01), 2)
        costs = [dd.threshold_metrics(yva, p_va, t, a.cost_false_alarm, a.cost_missed_delay)["cost_per_1000"]
                 for t in grid]
        thr = float(grid[int(np.argmin(costs))])
        op = dd.threshold_metrics(yte, p_te, thr, a.cost_false_alarm, a.cost_missed_delay)
        fm = {**dd.rank_metrics(yte, p_te), **op}

        ref = train[NUM_FEATURES + CAT_FEATURES].sample(min(50000, len(train)), random_state=SEED)
        bundle = {"model": final, "iso": iso, "thr": thr, "feats": FEATS, "model_name": best,
                  "run_id": parent.info.run_id, "num_features": NUM_FEATURES, "cat_features": CAT_FEATURES,
                  "ref_val_prob": p_va, "ref_val_pr_auc": float(dd.rank_metrics(yva, p_va)["PR_AUC"]),
                  "ref_late_rate": float(ytr.mean()),
                  "categories": {c: sorted(train[c].dropna().astype(str).unique().tolist()) for c in CAT_FEATURES},
                  "num_ranges": {c: [float(train[c].min()), float(train[c].max())] for c in NUM_FEATURES},
                  "ref_null_rate": {c: float(train[c].isna().mean()) for c in FEATS}}
        joblib.dump(bundle, out / "model_bundle.joblib")
        ref.to_csv(out / "reference_train.csv.gz", index=False)
        te = test[["order_purchase_timestamp"] + FEATS + ["late"]].sort_values("order_purchase_timestamp")
        te.to_csv(out / "test_stream.csv.gz", index=False)

        row = {"final_model": best, "threshold": thr, "test_PR_AUC": fm["PR_AUC"], "test_ROC_AUC": fm["ROC_AUC"],
               "precision": op["precision"], "recall": op["recall"], "flagged_pct": op["flagged_pct"],
               "cost_per_1000": op["cost_per_1000"]}
        # results.csv has no timing columns -> byte-identical across reruns with the same seed
        pd.DataFrame(results).T.round(6).to_csv(out / "results_models.csv")
        pd.Series(row).to_frame("value").round(6).to_csv(out / "results_final.csv")
        mlflow.log_metrics({"final_test_PR_AUC": fm["PR_AUC"], "final_precision": op["precision"],
                            "final_recall": op["recall"], "threshold": thr})
        mlflow.log_params({"final_model": best})
        mlflow.log_artifacts(str(out))
        print("\nFINAL:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
        print("run_id:", parent.info.run_id)


if __name__ == "__main__":
    main()

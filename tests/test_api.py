import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.isotonic import IsotonicRegression

import delivery_delay as dd
from u2.common import CAT_FEATURES, FEATS, NUM_FEATURES

ORDER = {"purchase_hour": 10, "purchase_dow": 2, "purchase_month": 5, "promised_days": 20.0,
         "approval_hours": 1.0, "shipping_limit_days": 4.0, "n_items": 1, "n_sellers": 1,
         "price_total": 50.0, "freight_total": 12.0, "freight_ratio": 0.24, "weight_total_g": 500.0,
         "volume_total_cm3": 4000.0, "distance_km": 800.0, "same_state": 0, "installments": 1,
         "n_payments": 1, "seller_hist_n": 20, "seller_hist_late_rate": 0.08, "customer_state": "RJ",
         "seller_state": "SP", "category": "toys", "payment_type": "credit_card"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({c: rng.random(n) * 10 for c in NUM_FEATURES})
    for c, vals in {"customer_state": ["RJ", "SP"], "seller_state": ["SP", "MG"],
                    "category": ["toys", "auto"], "payment_type": ["credit_card", "boleto"]}.items():
        X[c] = rng.choice(vals, n)
    y = (X.distance_km + rng.normal(size=n) > 6).astype(int)
    m = dd.make_models(NUM_FEATURES, CAT_FEATURES, 5.0, with_svm=False)["LogisticRegression"].fit(X[FEATS], y)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(m.predict_proba(X[FEATS])[:, 1], y)
    p = tmp_path_factory.mktemp("m") / "bundle.joblib"
    joblib.dump({"model": m, "iso": iso, "thr": 0.3, "feats": FEATS, "model_name": "LogisticRegression",
                 "run_id": "testrun12345"}, p)
    import u2.app as app_module
    app_module.MODEL_PATH = str(p)
    app_module.LOG_PATH = tmp_path_factory.mktemp("l") / "pred.jsonl"
    app_module.STATE["bundle"] = None
    with TestClient(app_module.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_probability(client):
    r = client.post("/predict", json=ORDER)
    assert r.status_code == 200
    j = r.json()
    assert 0.0 <= j["late_probability"] <= 1.0 and isinstance(j["flag"], bool)


def test_predict_accepts_missing_optional_fields(client):
    o = {**ORDER, "distance_km": None, "weight_total_g": None}
    assert client.post("/predict", json=o).status_code == 200


def test_predict_rejects_invalid_input(client):
    assert client.post("/predict", json={**ORDER, "purchase_hour": 99}).status_code == 422
    assert client.post("/predict", json={**ORDER, "distance_km": -5}).status_code == 422
    bad = {k: v for k, v in ORDER.items() if k != "category"}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_predict(client):
    r = client.post("/batch_predict", json={"orders": [ORDER, ORDER, ORDER]})
    assert r.status_code == 200 and len(r.json()) == 3
    assert client.post("/batch_predict", json={"orders": []}).status_code == 422


def test_unseen_category_does_not_crash(client):
    assert client.post("/predict", json={**ORDER, "category": "never_seen_before"}).status_code == 200

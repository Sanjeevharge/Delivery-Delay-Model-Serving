import numpy as np
import pandas as pd
import delivery_delay as dd
from u2.monitor import psi_numeric, psi_categorical


def test_haversine_known_distance():
    # Sao Paulo -> Rio de Janeiro is ~360 km
    d = dd.haversine_km(-23.55, -46.63, -22.91, -43.17)
    assert 340 < float(d) < 380
    assert float(dd.haversine_km(1, 1, 1, 1)) == 0.0


def test_time_split_is_chronological_and_disjoint():
    ts = pd.date_range("2017-01-01", periods=1000, freq="h")
    df = pd.DataFrame({"order_purchase_timestamp": np.random.default_rng(0).permutation(ts), "late": 0})
    out, t1, t2 = dd.time_split(df)
    tr, va, te = (out[out.split == s].order_purchase_timestamp for s in ("train", "val", "test"))
    assert tr.max() < va.min() and va.max() < te.min()
    assert len(tr) + len(va) + len(te) == 1000


def test_seller_history_uses_only_past_deliveries():
    t = pd.Timestamp
    df = pd.DataFrame({
        "order_id": ["a", "b", "c"], "seller_id": ["s1"] * 3,
        "order_purchase_timestamp": [t("2017-01-01"), t("2017-01-05"), t("2017-01-20")],
        "order_delivered_customer_date": [t("2017-01-10"), t("2017-01-12"), t("2017-01-25")],
        "late": [1, 0, 1]})
    out = dd.add_seller_history(df, prior=0.1).set_index("order_id")
    assert out.loc["a", "seller_hist_n"] == 0          # nothing delivered before purchase
    assert out.loc["b", "seller_hist_n"] == 0          # 'a' delivered Jan 10, after b was bought
    assert out.loc["c", "seller_hist_n"] == 2          # a and b delivered before Jan 20; own outcome excluded


def test_psi_zero_for_same_and_large_for_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi_numeric(x, rng.normal(size=5000)) < 0.05
    assert psi_numeric(x, rng.normal(loc=2, size=5000)) > 0.2
    c = pd.Series(["a"] * 500 + ["b"] * 500)
    assert psi_categorical(c, c) < 1e-6
    assert psi_categorical(c, pd.Series(["a"] * 950 + ["b"] * 50)) > 0.2

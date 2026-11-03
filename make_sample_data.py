"""
Generate a SYNTHETIC dataset with the exact Olist file names and columns.
Only for smoke-testing delivery_delay.py when the real Kaggle data is not available.
Numbers produced on this data mean nothing about the real business.

    python make_sample_data.py --out data_sample --orders 12000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATES = {"SP": (-23.5, -46.6, 0.42), "RJ": (-22.9, -43.2, 0.13), "MG": (-19.9, -43.9, 0.12),
          "RS": (-30.0, -51.2, 0.055), "PR": (-25.4, -49.3, 0.05), "SC": (-27.6, -48.5, 0.037),
          "BA": (-12.97, -38.5, 0.034), "DF": (-15.8, -47.9, 0.022), "GO": (-16.7, -49.3, 0.02),
          "PE": (-8.05, -34.9, 0.017), "CE": (-3.7, -38.5, 0.013), "MA": (-2.5, -44.3, 0.008),
          "PA": (-1.45, -48.5, 0.01), "ES": (-20.3, -40.3, 0.02)}
CATS = {"cama_mesa_banho": ("bed_bath_table", 0.5), "beleza_saude": ("health_beauty", 0.0),
        "esporte_lazer": ("sports_leisure", 0.3), "moveis_decoracao": ("furniture_decor", 1.5),
        "informatica_acessorios": ("computers_accessories", 0.5), "utilidades_domesticas": ("housewares", 0.3),
        "relogios_presentes": ("watches_gifts", 0.0), "telefonia": ("telephony", 0.2),
        "ferramentas_jardim": ("garden_tools", 0.8), "automotivo": ("auto", 0.4),
        "brinquedos": ("toys", 0.2), "moveis_escritorio": ("office_furniture", 3.0)}
STATE_EFFECT = {"RJ": 2.5, "BA": 2.0, "MA": 3.5, "PA": 3.0, "CE": 2.0, "PE": 1.5, "ES": 1.0}
FMT = "%Y-%m-%d %H:%M:%S"


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_sample")
    ap.add_argument("--orders", type=int, default=12000)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = list(STATES); w = np.array([STATES[s][2] for s in names]); w /= w.sum()

    # zip prefixes + geolocation (several jittered rows per prefix, plus exact duplicates)
    z = []
    for i, s in enumerate(names):
        lat, lng, _ = STATES[s]
        for j in range(25):
            z.append((10000 + i * 100 + j, s, lat + rng.normal(0, .6), lng + rng.normal(0, .6)))
    zips = pd.DataFrame(z, columns=["zip", "state", "lat", "lng"])
    g = zips.loc[zips.index.repeat(3)].reset_index(drop=True)
    geo = pd.DataFrame({"geolocation_zip_code_prefix": g.zip, "geolocation_lat": g.lat + rng.normal(0, .02, len(g)),
                        "geolocation_lng": g.lng + rng.normal(0, .02, len(g)), "geolocation_city": "cidade",
                        "geolocation_state": g.state})
    geo = pd.concat([geo, geo.head(40)], ignore_index=True)
    geo.to_csv(out / "olist_geolocation_dataset.csv", index=False)
    zl = zips.set_index("zip")

    def pick_zip(states):
        by = {s: zips.zip.values[zips.state.values == s] for s in names}
        return np.array([rng.choice(by[s]) for s in states])

    # sellers
    ns = 250
    sp = w.copy(); sp[names.index("SP")] += 1.0; sp /= sp.sum()
    s_state = rng.choice(names, ns, p=sp)
    sellers = pd.DataFrame({"seller_id": [f"s{i:04d}" for i in range(ns)], "seller_zip_code_prefix": pick_zip(s_state),
                            "seller_city": "cidade", "seller_state": s_state})
    s_delay = rng.gamma(1.5, 1.0, ns); s_delay[rng.random(ns) < .1] += 6
    s_pop = rng.pareto(1.2, ns) + 1; s_pop /= s_pop.sum()
    sellers.to_csv(out / "olist_sellers_dataset.csv", index=False)

    # products
    npd = 800
    pcat = rng.choice(list(CATS), npd)
    weight = rng.lognormal(6.5, 1.1, npd).round()
    products = pd.DataFrame({"product_id": [f"p{i:05d}" for i in range(npd)],
                             "product_category_name": np.where(rng.random(npd) < .02, None, pcat),
                             "product_name_lenght": rng.integers(20, 60, npd),
                             "product_description_lenght": rng.integers(100, 3000, npd),
                             "product_photos_qty": rng.integers(1, 6, npd), "product_weight_g": weight,
                             "product_length_cm": rng.integers(10, 80, npd).astype(float),
                             "product_height_cm": rng.integers(2, 60, npd).astype(float),
                             "product_width_cm": rng.integers(10, 60, npd).astype(float)})
    products.loc[products.sample(5, random_state=1).index, ["product_weight_g", "product_length_cm"]] = np.nan
    products.to_csv(out / "olist_products_dataset.csv", index=False)
    pd.DataFrame({"product_category_name": list(CATS), "product_category_name_english": [v[0] for v in CATS.values()]}
                 ).to_csv(out / "product_category_name_translation.csv", index=False)

    # orders
    n = a.orders
    start, end = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-09-30")
    purchase = pd.Series(start + pd.to_timedelta(np.sqrt(rng.random(n)) * (end - start).total_seconds(), unit="s"))
    purchase = purchase.sort_values().reset_index(drop=True).dt.floor("s")
    c_state = rng.choice(names, n, p=w); c_zip = pick_zip(c_state)
    sidx = rng.choice(ns, n, p=s_pop); s_zip = sellers.seller_zip_code_prefix.values[sidx]
    pidx = rng.integers(0, npd, n)
    dist = hav(zl.loc[c_zip, "lat"].values, zl.loc[c_zip, "lng"].values, zl.loc[s_zip, "lat"].values, zl.loc[s_zip, "lng"].values)
    month = purchase.dt.month.values
    season = np.where(np.isin(month, [11, 12]), 3.5, np.where(np.isin(month, [2, 3]), 2.0, 0.0))
    st_eff = pd.Series(STATE_EFFECT).reindex(c_state).fillna(0).values
    cat_eff = np.array([CATS[c][1] for c in pcat[pidx]])
    weekend = (purchase.dt.dayofweek.values >= 5) * 1.0
    promised = np.round(rng.normal(21, 4, n) + dist / 300).clip(5, 60)
    actual = (2 + dist / 250 + s_delay[sidx] + season + st_eff + cat_eff + weekend
              + weight[pidx] / 8000 + rng.gamma(2.0, 2.5, n))
    approved = purchase + pd.to_timedelta(rng.exponential(10, n), unit="h")
    delivered = purchase + pd.to_timedelta(actual, unit="D")
    carrier = approved + pd.to_timedelta(np.minimum(rng.gamma(2, 1.2, n), actual * .5), unit="D")
    estimated = (purchase + pd.to_timedelta(promised, unit="D")).dt.normalize()
    status = rng.choice(["delivered", "canceled", "unavailable", "shipped", "invoiced"], n,
                        p=[.965, .012, .008, .01, .005]).astype(object)
    censored = delivered > pd.Timestamp("2018-10-17")
    status[censored & (status == "delivered")] = "shipped"
    delivered = delivered.where(status == "delivered")
    carrier = carrier.where(np.isin(status, ["delivered", "shipped"]))
    approved = approved.where(rng.random(n) > .002)
    bad = rng.choice(n, 8, replace=False)
    delivered.iloc[bad[:4]] = purchase.iloc[bad[:4]] - pd.Timedelta(days=2)
    approved.iloc[bad[4:]] = purchase.iloc[bad[4:]] - pd.Timedelta(days=1)

    oid = [f"o{i:06d}" for i in range(n)]; cid = [f"c{i:06d}" for i in range(n)]
    pd.DataFrame({"order_id": oid, "customer_id": cid, "order_status": status,
                  "order_purchase_timestamp": purchase, "order_approved_at": approved,
                  "order_delivered_carrier_date": carrier, "order_delivered_customer_date": delivered,
                  "order_estimated_delivery_date": estimated}).to_csv(out / "olist_orders_dataset.csv", index=False, date_format=FMT)
    pd.DataFrame({"customer_id": cid, "customer_unique_id": [f"u{i:06d}" for i in rng.integers(0, int(n * .95), n)],
                  "customer_zip_code_prefix": c_zip, "customer_city": "cidade", "customer_state": c_state}
                 ).to_csv(out / "olist_customers_dataset.csv", index=False)

    # items
    k = rng.choice([1, 1, 1, 1, 1, 1, 1, 2, 2, 3], n)
    rows = np.repeat(np.arange(n), k)
    item_no = np.concatenate([np.arange(1, x + 1) for x in k])
    prod = np.where(item_no == 1, pidx[rows], rng.integers(0, npd, len(rows)))
    lim = approved.fillna(purchase).iloc[rows].values + np.timedelta64(6, "D")
    price = rng.lognormal(4, .8, len(rows)).round(2)
    freight = (8 + dist[rows] / 150 + weight[prod] / 1000 * 2 + rng.normal(0, 2, len(rows))).clip(0).round(2)
    pd.DataFrame({"order_id": np.array(oid)[rows], "order_item_id": item_no, "product_id": products.product_id.values[prod],
                  "seller_id": sellers.seller_id.values[sidx[rows]], "shipping_limit_date": pd.to_datetime(lim),
                  "price": price, "freight_value": freight}).to_csv(out / "olist_order_items_dataset.csv", index=False, date_format=FMT)

    # payments
    total = pd.Series(price + freight).groupby(rows).sum().values
    ptype = rng.choice(["credit_card", "boleto", "voucher", "debit_card"], n, p=[.74, .19, .05, .02])
    pay = pd.DataFrame({"order_id": oid, "payment_sequential": 1, "payment_type": ptype,
                        "payment_installments": np.where(ptype == "credit_card", rng.integers(1, 11, n), 1),
                        "payment_value": total.round(2)})
    extra = pay.sample(frac=.03, random_state=2).assign(payment_sequential=2, payment_type="voucher", payment_installments=1, payment_value=10.0)
    pd.concat([pay, extra]).to_csv(out / "olist_order_payments_dataset.csv", index=False)

    # reviews (post-delivery -> leakage; generated only so the audit has something to point at)
    late = (delivered.dt.normalize() > estimated).fillna(False).values
    pd.DataFrame({"review_id": [f"r{i:06d}" for i in range(n)], "order_id": oid,
                  "review_score": np.where(late, rng.integers(1, 4, n), rng.integers(3, 6, n)),
                  "review_comment_title": None, "review_comment_message": None,
                  "review_creation_date": (delivered + pd.Timedelta(days=1)).dt.normalize(),
                  "review_answer_timestamp": delivered + pd.Timedelta(days=2)}
                 ).to_csv(out / "olist_order_reviews_dataset.csv", index=False, date_format=FMT)
    print(f"synthetic Olist-format data written to {out.resolve()} ({n} orders, late share among delivered "
          f"{late[status == 'delivered'].mean():.1%})")


if __name__ == "__main__":
    main()


"""Fixed-rate load test. Reports p50/p95/p99 latency and errors.

    python -m u2.loadtest --url http://localhost:8000/predict --rate 50 --seconds 30
"""
import argparse, asyncio, json, time
import numpy as np
import httpx
import pandas as pd
from u2.common import FEATS


def payloads(path, n=500):
    d = pd.read_csv(path).sample(n, random_state=42)[FEATS]
    out = []
    for r in d.to_dict("records"):
        r = {k: (None if pd.isna(v) else v) for k, v in r.items()}
        for k in ("purchase_hour", "purchase_dow", "purchase_month", "n_items", "n_sellers", "same_state", "n_payments"):
            if r[k] is not None: r[k] = int(r[k])
        out.append(r)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/predict")
    ap.add_argument("--stream", default="artifacts/test_stream.csv.gz")
    ap.add_argument("--rate", type=float, default=50, help="requests per second")
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--out", default="artifacts/loadtest.json")
    a = ap.parse_args()
    bodies = payloads(a.stream)
    lat, errs = [], 0

    async with httpx.AsyncClient(timeout=10, limits=httpx.Limits(max_connections=200)) as c:
        async def one(i):
            nonlocal errs
            t = time.perf_counter()
            try:
                r = await c.post(a.url, json=bodies[i % len(bodies)])
                if r.status_code != 200: errs += 1
            except Exception:
                errs += 1
            lat.append((time.perf_counter() - t) * 1000)

        tasks, n = [], int(a.rate * a.seconds)
        start = time.perf_counter()
        for i in range(n):  # open-loop: send on schedule regardless of response time
            await asyncio.sleep(max(0, start + i / a.rate - time.perf_counter()))
            tasks.append(asyncio.create_task(one(i)))
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - start

    lat = np.array(lat)
    res = {"requests": n, "target_rps": a.rate, "achieved_rps": round(n / wall, 1), "errors": errs,
           "p50_ms": round(float(np.percentile(lat, 50)), 1), "p95_ms": round(float(np.percentile(lat, 95)), 1),
           "p99_ms": round(float(np.percentile(lat, 99)), 1), "max_ms": round(float(lat.max()), 1)}
    print(json.dumps(res, indent=2))
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    asyncio.run(main())

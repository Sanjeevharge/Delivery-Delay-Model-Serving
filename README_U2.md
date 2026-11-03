# U2 – Serve and monitor the delay model

Run from the repo root (same folder as `delivery_delay.py`). PowerShell; set once:
```powershell
$py = "C:\Users\Sanjeev\anaconda3\python.exe"
& $py -m pip install -r requirements-dev.txt
```

| Step | Command | Pass if |
|---|---|---|
| 1 Tests | `& $py -m pytest -q tests` | `10 passed` |
| 2 Train (tracked) | `& $py -m u2.train --data data --out artifacts` | prints `FINAL: {...}`; `artifacts\model_bundle.joblib` exists |
| 3 MLflow UI | `& $py -m mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://localhost:5000 | experiment `udaan-delivery-delay`, 1 parent + 4 child runs, metrics and artifacts logged |
| 4 Reproducibility | `& $py -m u2.train --data data --out artifacts_r1` then `--out artifacts_r2`; `Compare-Object (gc artifacts_r1\results_models.csv) (gc artifacts_r2\results_models.csv)` (also `results_final.csv`) | no output (files identical) |
| 5 Serve (local) | `$env:MODEL_PATH="artifacts\model_bundle.joblib"; & $py -m uvicorn u2.app:app --port 8000` | `curl.exe localhost:8000/health` → `status ok` |
| 6 Docker | `& $py -m u2.pin_reqs` then `docker compose up --build` | `curl.exe localhost:8000/health` → ok |
| 7 Load test | `& $py -m u2.loadtest --stream artifacts\test_stream.csv.gz --rate 50 --seconds 30` | `errors: 0`, `achieved_rps` ≈ 50, note `p95_ms` |
| 8 Monitoring | `& $py -m u2.monitor --scenario clean --out monitor_out` and `--scenario drift` | clean: ~0 alerts; drift: alert fires, prints batches-to-alert |
| 9 CI | push to GitHub | Actions run green |

## Limitations
- Seller-history features are supplied by the caller (a feature store in production); the API does not compute them.
- Drift scenario is **synthetic** (see `inject_drift` in `u2/monitor.py`).
- `seller_hist_*` are cumulative, so they drift by construction; they are excluded from drift/range alerts.
- Latency measured on one machine, single worker, no network hop.

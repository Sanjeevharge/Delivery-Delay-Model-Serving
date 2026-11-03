"""FastAPI service: /predict, /batch_predict, /health. Logs every prediction to JSONL."""
import json, os, time, uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from u2.common import FEATS, predict_prob

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model_bundle.joblib")
LOG_PATH = Path(os.getenv("PRED_LOG", "logs/predictions.jsonl"))
MAX_BATCH = 1000


class Order(BaseModel):
    """One order at purchase/approval time. Seller-history fields come from the caller (feature store)."""
    purchase_hour: int = Field(ge=0, le=23)
    purchase_dow: int = Field(ge=0, le=6)
    purchase_month: int = Field(ge=1, le=12)
    promised_days: float = Field(gt=0, le=200)
    approval_hours: Optional[float] = Field(default=None, ge=0, le=2000)
    shipping_limit_days: Optional[float] = Field(default=None, ge=-5, le=200)
    n_items: int = Field(ge=1, le=100)
    n_sellers: int = Field(ge=1, le=20)
    price_total: float = Field(ge=0, le=1e6)
    freight_total: float = Field(ge=0, le=1e5)
    freight_ratio: float = Field(ge=0, le=1e6)
    weight_total_g: Optional[float] = Field(default=None, ge=0, le=1e6)
    volume_total_cm3: Optional[float] = Field(default=None, ge=0, le=1e8)
    distance_km: Optional[float] = Field(default=None, ge=0, le=20000)
    same_state: int = Field(ge=0, le=1)
    installments: Optional[float] = Field(default=None, ge=0, le=60)
    n_payments: Optional[int] = Field(default=None, ge=0, le=50)
    seller_hist_n: float = Field(ge=0)
    seller_hist_late_rate: float = Field(ge=0, le=1)
    customer_state: str = Field(min_length=2, max_length=2)
    seller_state: str = Field(min_length=2, max_length=2)
    category: str = Field(min_length=1, max_length=80)
    payment_type: str = Field(min_length=1, max_length=40)


class Prediction(BaseModel):
    late_probability: float
    flag: bool
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    orders: List[Order] = Field(min_length=1, max_length=MAX_BATCH)


STATE = {"bundle": None}


@asynccontextmanager
async def lifespan(_):
    try:
        get_bundle()
    except Exception as e:  # /health will report it
        print("model load failed:", e)
    yield


app = FastAPI(title="Delivery-delay model", version="1.0", lifespan=lifespan)


def get_bundle():
    if STATE["bundle"] is None:
        STATE["bundle"] = joblib.load(MODEL_PATH)
    return STATE["bundle"]


def _version(b):
    return f"{b['model_name']}:{b['run_id'][:8]}"


def _log(rows, probs, flags, b, latency_ms):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = time.time()
    with LOG_PATH.open("a", encoding="utf-8") as f:
        for r, p, fl in zip(rows, probs, flags):
            f.write(json.dumps({"id": uuid.uuid4().hex, "ts": ts, "model_version": _version(b),
                                "features": r, "prob": float(p), "flag": bool(fl),
                                "latency_ms": round(latency_ms, 3)}) + "\n")


def _score(orders: List[Order]):
    b = get_bundle()
    t0 = time.perf_counter()
    rows = [o.model_dump() for o in orders]
    X = pd.DataFrame(rows)[FEATS]
    probs = predict_prob(b, X)
    flags = probs >= b["thr"]
    _log(rows, probs, flags, b, (time.perf_counter() - t0) * 1000)
    return [Prediction(late_probability=float(p), flag=bool(f), threshold=b["thr"],
                       model_version=_version(b)) for p, f in zip(probs, flags)]


@app.get("/health")
def health():
    try:
        b = get_bundle()
        return {"status": "ok", "model": _version(b), "threshold": b["thr"]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"model not loaded: {e}")


@app.post("/predict", response_model=Prediction)
def predict(order: Order):
    return _score([order])[0]


@app.post("/batch_predict", response_model=List[Prediction])
def batch_predict(req: BatchRequest):
    return _score(req.orders)

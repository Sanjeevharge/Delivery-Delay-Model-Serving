"""Shared helpers for training, serving and monitoring."""
import numpy as np

NUM_FEATURES = ["purchase_hour", "purchase_dow", "purchase_month", "promised_days", "approval_hours",
                "shipping_limit_days", "n_items", "n_sellers", "price_total", "freight_total",
                "freight_ratio", "weight_total_g", "volume_total_cm3", "distance_km", "same_state",
                "installments", "n_payments", "seller_hist_n", "seller_hist_late_rate"]
CAT_FEATURES = ["customer_state", "seller_state", "category", "payment_type"]
FEATS = NUM_FEATURES + CAT_FEATURES


def raw_score(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def predict_prob(bundle: dict, X) -> np.ndarray:
    """Calibrated late-probability using the saved model + isotonic calibrator."""
    s = raw_score(bundle["model"], X[bundle["feats"]])
    return bundle["iso"].predict(s)

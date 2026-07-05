"""
model.py — Ensemble ML model for ICU safe capacity prediction.

What we predict:
  safe_capacity = available_beds discounted by current strain.

A hospital with 80 beds but 60 patients queuing and no specialists
has far less TRUE capacity than one with 40 beds and no queue.
The model learns this discount from cross-hospital patterns.

Prediction is then blended back with raw available_beds so that
the displayed number stays grounded in reality.
"""

import os, pickle, logging, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble        import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics         import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from config import FEATURE_COLS, TARGET_COL, MODEL_PATH

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    total = df["total_beds"].replace(0, 1).astype(float)
    avail = df["available_beds"].clip(lower=0).astype(float)
    avg_p = df["avg_daily_patients"].replace(0, 1).astype(float)
    crit  = df["critical_patients"].clip(lower=0).astype(float)
    queue = df["waiting_queue"].clip(lower=0).astype(float)
    oxy   = df.get("oxygen_supply_pct",   pd.Series(80.0, index=df.index)).clip(0, 100)
    staff = df.get("icu_specialist_count", pd.Series(5,   index=df.index)).replace(0, 1).astype(float)
    vents = df["ventilators"].replace(0, 1).astype(float)

    df["occupancy_rate"]  = ((total - avail) / total).clip(0, 1)
    df["critical_ratio"]  = (crit / avg_p).clip(0, 1)
    df["bed_pressure"]    = (queue / total).clip(0, 1)
    df["resource_strain"] = (crit / staff).clip(0, 5)

    # Readiness score (0-1): how well-equipped is this hospital right now
    oxy_n   = (oxy / 100).clip(0, 1)
    vent_n  = (df["ventilators"] / total).clip(0, 1)
    staff_n = (staff / 15).clip(0, 1)
    readiness = 0.40 * oxy_n + 0.35 * vent_n + 0.25 * staff_n

    # Pressure discount: queue + occupancy eat into real availability
    pressure = (df["bed_pressure"] * 0.6 + df["occupancy_rate"] * 0.4).clip(0, 1)

    # safe_capacity: what's actually usable given current strain
    df[TARGET_COL] = (
        avail * (1.0 - pressure * 0.45) * (0.65 + 0.35 * readiness)
    ).clip(lower=0).round(2)

    return df


class ICUModelBundle:
    def __init__(self):
        self.gb = Pipeline([("sc", StandardScaler()),
                            ("m",  GradientBoostingRegressor(
                                n_estimators=300, learning_rate=0.06,
                                max_depth=4, subsample=0.80,
                                min_samples_leaf=3, random_state=42))])
        self.rf = Pipeline([("sc", StandardScaler()),
                            ("m",  RandomForestRegressor(
                                n_estimators=300, max_features="sqrt",
                                min_samples_leaf=3, random_state=42))])
        self.metrics            = {}
        self.feature_importance = {}
        self._trained           = False
        self._feats             = []

    def _prep(self, df):
        df_fe = engineer_features(df)
        feats = [c for c in FEATURE_COLS if c in df_fe.columns]
        X     = df_fe[feats].fillna(0)
        X     = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=1)
        return X, df_fe, feats

    def fit(self, df):
        X, df_fe, feats = self._prep(df)
        y = df_fe[TARGET_COL]
        self._feats = feats

        if len(X) < 4:
            self.gb.fit(X, y); self.rf.fit(X, y)
            self.metrics = {"note": "Too few samples"}
            self._trained = True
            return self

        n_splits  = max(2, min(5, len(X) // 10)) if len(X) >= 50 else 2
        cv_scores = cross_val_score(self.gb, X, y,
                                    cv=KFold(n_splits, shuffle=True, random_state=42),
                                    scoring="neg_mean_absolute_error")
        self.gb.fit(X, y); self.rf.fit(X, y)
        ens = np.maximum(0, (self.gb.predict(X) + self.rf.predict(X)) / 2)

        self.metrics = {
            "MAE":       round(float(mean_absolute_error(y, ens)), 2),
            "R2":        round(float(r2_score(y, ens)), 4),
            "CV_MAE":    round(float(-cv_scores.mean()), 2),
            "CV_STD":    round(float(cv_scores.std()), 2),
            "n_samples": len(X),
        }
        try:
            self.feature_importance = dict(zip(feats, self.rf.named_steps["m"].feature_importances_))
        except Exception:
            pass
        self._trained = True
        logger.info("Trained: %s", self.metrics)
        return self

    def _X(self, df):
        df_fe = engineer_features(df)
        feats = [c for c in FEATURE_COLS if c in df_fe.columns]
        return df_fe[feats].fillna(0)

    def predict_with_confidence(self, df):
        X  = self._X(df)
        gb = self.gb.predict(X)
        rf = self.rf.predict(X)
        ml = np.maximum(0, (gb + rf) / 2)

        # Confidence: relative agreement between the two models
        scale = np.maximum(ml, 1)
        conf  = np.exp(-2 * np.abs(gb - rf) / scale)
        return ml, np.clip(conf, 0.05, 0.99)


def train_model(df):
    b = ICUModelBundle(); b.fit(df)
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f: pickle.dump(b, f)
    except Exception as e:
        logger.warning("Cache failed: %s", e)
    return b


def load_or_train_model(df):
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f: b = pickle.load(f)
            if b._trained:
                if set(getattr(b, "_feats", [])) != set(FEATURE_COLS):
                    logger.info("Feature mismatch — retraining")
                elif b.metrics.get("n_samples", 0) == len(df):
                    logger.info("Loaded cached model"); return b
        except Exception as e:
            logger.warning("Load error: %s", e)
    return train_model(df)


def predict_beds(bundle, df):
    df = df.copy()
    # Drop hospitals with missing coordinates before prediction
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"] != "") & (df["longitude"] != "")]
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    ml_pred, conf = bundle.predict_with_confidence(df)

    actual = df["available_beds"].values.astype(float)
    # Blend: 60% actual (ground truth) + 40% ML forward signal
    blended = np.clip(0.60 * actual + 0.40 * ml_pred, 0, df["total_beds"].values)

    df["predicted_beds"] = np.round(blended).astype(int)
    df["confidence"]     = np.round(conf, 2)
    df["contact_number"] = df.get("contact_number", pd.Series("N/A", index=df.index)).fillna("N/A").astype(str)
    return df
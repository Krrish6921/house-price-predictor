"""FastAPI house price band predictor trained from data/Housing.csv."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "Housing.csv"
NUMERIC = ["area", "bedrooms", "bathrooms", "stories", "parking"]
CATEGORICAL = ["mainroad", "guestroom", "basement", "hotwaterheating",
               "airconditioning", "prefarea", "furnishingstatus"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "price"
PRICE_LABELS = ["Budget", "Standard", "Premium", "Luxury"]

app = FastAPI(title="House Price Predictor", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def load_and_clean() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    df = df[FEATURES + [TARGET]].copy()
    for column in NUMERIC + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in CATEGORICAL:
        df[column] = df[column].astype("string").str.strip().str.lower().replace("", pd.NA)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET])
    # Remove impossible targets/measurements; remaining feature gaps are imputed.
    df = df[(df.price > 0) & (df.area > 0) & (df.bedrooms >= 0) &
            (df.bathrooms >= 0) & (df.stories >= 0) & (df.parking >= 0)]
    df = df.drop_duplicates()
    if len(df) < 40:
        raise ValueError("At least 40 valid sales rows are required to train and evaluate the model.")
    return df


def train_model() -> dict[str, Any]:
    df = load_and_clean()
    # Quantile-based target bins avoid hard-coded currency thresholds and balance classes.
    df["price_band"] = pd.qcut(df.price, q=4, labels=PRICE_LABELS, duplicates="drop")
    bands = list(df["price_band"].cat.categories)
    if len(bands) < 2:
        raise ValueError("The price column needs more variation to form price bands.")
    X, y = df[FEATURES], df["price_band"].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    model = Pipeline([("preprocess", prep),
                      ("classifier", GradientBoostingClassifier(
                          n_estimators=200, learning_rate=0.1, max_depth=1,
                          min_samples_leaf=15, subsample=1.0, random_state=42,
                      ))])
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    order = [str(b) for b in bands]
    return {
        "model": model, "medians": df.groupby("price_band", observed=False).price.median().to_dict(),
        "bands": order, "metrics": {
            "accuracy": round(float(accuracy_score(y_test, predicted)), 3),
            "macro_f1": round(float(f1_score(y_test, predicted, average="macro")), 3),
            "test_rows": len(y_test), "training_rows": len(x_train),
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=order).tolist(),
        }, "rows": len(df), "price_min": float(df.price.min()),
        "price_max": float(df.price.max()),
        "category_options": {column: sorted(df[column].dropna().astype(str).unique().tolist())
                             for column in CATEGORICAL},
    }


try:
    STATE = train_model()
except Exception as exc:
    STATE = {"error": str(exc)}


class House(BaseModel):
    area: float = Field(gt=0, le=100000)
    bedrooms: float = Field(ge=0, le=30)
    bathrooms: float = Field(ge=0, le=30)
    stories: float = Field(ge=0, le=30)
    parking: float = Field(ge=0, le=30)
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    prefarea: str
    furnishingstatus: str


@app.get("/")
def home():
    return {"message": "House Price Predictor", "app": "/static/index.html"}


@app.get("/api/summary")
def summary():
    if "error" in STATE:
        return {"ready": False, "error": STATE["error"]}
    return {"ready": True, "rows": STATE["rows"],
            "metrics": STATE["metrics"], "bands": STATE["bands"],
            "price_range": [round(STATE["price_min"]), round(STATE["price_max"])],
            "category_options": STATE["category_options"]}


@app.post("/api/predict")
def predict(house: House):
    if "error" in STATE:
        raise HTTPException(503, STATE["error"])
    values = house.model_dump()
    frame = pd.DataFrame([values], columns=FEATURES)
    model = STATE["model"]
    probs = model.predict_proba(frame)[0]
    classes = list(model.named_steps["classifier"].classes_)
    band = str(classes[int(np.argmax(probs))])
    median = STATE["medians"].get(band)
    # A probability-weighted estimate uses the median representative price of each band.
    estimate = sum(float(STATE["medians"][str(label)]) * float(prob)
                   for label, prob in zip(classes, probs))
    return {"price_estimate": round(estimate), "price_band": band,
            "confidence": round(float(max(probs)), 3),
            "band_median": round(float(median)) if median is not None else None,
            "band_probabilities": {str(label): round(float(prob), 3) for label, prob in zip(classes, probs)},
            "demo_data": False}

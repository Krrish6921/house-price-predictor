# Hearth — House Price Predictor

A FastAPI app with an HTML/CSS/JavaScript frontend, trained from `data/Housing.csv`.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --reload-exclude '.venv/**'
```

Open <http://127.0.0.1:8000/static/index.html>. API docs: <http://127.0.0.1:8000/docs>.

## Dataset and model

The CSV must be at `data/Housing.csv` and include `price`, `area`, `bedrooms`, `bathrooms`, `stories`, `parking`, `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, and `furnishingstatus`. The app removes invalid/non-finite targets, impossible measurements, and duplicate rows. Numeric gaps are median-imputed; categorical gaps are most-frequent-imputed and categories one-hot encoded.

A Gradient Boosting classifier predicts one of four price bands formed from price quartiles. The displayed indicative estimate is the probability-weighted average of the training-set median price in each band. Accuracy, macro F1, and the confusion matrix are evaluated on a stratified 20% holdout set. Dataset prices are assumed to be in Bangladeshi taka (BDT); the frontend converts them to USD using a fixed 123.8377 BDT per USD rate. Currency conversion does not change the dataset's market or location.

Endpoints: `GET /api/summary`, `POST /api/predict`.
# house-price-predictor

# House Price Prediction

This repository contains a house price regression pipeline built with `scikit-learn`.

## Project structure

- `data/` — dataset files
- `src/` — training and inference code
- `models/` — serialized model artifacts
- `requirements.txt` — Python dependencies

## Setup

1. Open PowerShell and go to the project folder:

```powershell
cd "d:\house price prediction"
```

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train the model

Train a model using the sample data and save it to `models/house_price_model.joblib`:

```powershell
python -m src.run_training --input data/sample_house_prices.csv --output models/house_price_model.joblib
```

This prints evaluation metrics including RMSE, MAE, and R².

## Predict a sample

Use the saved model to predict a single house sample from a JSON file:

```powershell
python -m src.predict --model models/house_price_model.joblib --sample-file data/sample_prediction.json
```

Or pass a JSON string directly:

```powershell
python -m src.predict --model models/house_price_model.joblib --sample "{\"sqft_living\": 2000, \"bedrooms\": 3, \"bathrooms\": 2, \"floors\": 1, \"zipcode\": \"98178\"}"
```

## Use the model in Python

```python
from src.house_price_prediction import load_model, predict_samples

pipeline = load_model("models/house_price_model.joblib")
sample = {
    "sqft_living": 2000,
    "bedrooms": 3,
    "bathrooms": 2,
    "floors": 1,
    "zipcode": "98178"
}
print(predict_samples(pipeline, [sample]))
```

## Notes

- If you want to push this project to GitHub, initialize Git in the project folder and add a remote.
- The model can be retrained with any CSV that contains a `price` column and other feature columns.

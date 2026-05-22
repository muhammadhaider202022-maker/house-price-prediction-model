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
cd "#file location of the project"
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

If you need me to push it for you, please provide the GitHub repository URL or the repository name and I can add the remote and push it.

## Vercel deployment

This project now includes a Python entrypoint for Vercel:

- `app.py` defines a WSGI `app` entrypoint
- `pyproject.toml` sets `tool.vercel.entrypoint = "app:app"`
- `vercel.json` excludes local virtual environments and test artifacts

To deploy on Vercel, push the repository to GitHub and connect the repo in the Vercel dashboard. Vercel will use the `app.py` entrypoint and install dependencies from `pyproject.toml` or `requirements.txt`.

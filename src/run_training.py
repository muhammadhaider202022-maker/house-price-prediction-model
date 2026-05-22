import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.house_price_prediction import TARGET_COLUMN, load_data, save_model, train_model, evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a house price prediction model")
    parser.add_argument("--input", type=str, default="data/sample_house_prices.csv", help="Path to the training CSV file")
    parser.add_argument("--output", type=str, default="models/house_price_model.joblib", help="Path where the trained model will be saved")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to reserve for testing")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for train/test split and model training")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees for the random forest")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum depth of the random forest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(args.input, target=TARGET_COLUMN)

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    model = train_model(
        X_train, y_train, random_state=args.random_state, n_estimators=args.n_estimators, max_depth=args.max_depth
    )
    metrics = evaluate_model(model, X_test, y_test)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, args.output)

    print("Training completed successfully")
    print(f"Saved model to: {args.output}")
    print("Evaluation metrics:")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAE:  {metrics['mae']:.2f}")
    print(f"  R²:   {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()

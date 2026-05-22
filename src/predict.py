import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.house_price_prediction import load_model, load_json_sample, predict_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict house prices using a trained model")
    parser.add_argument("--model", type=str, default="models/house_price_model.joblib", help="Path to the trained model artifact")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sample",
        type=str,
        help="JSON string representing a single sample, e.g. '{\"sqft_living\": 2000, \"bedrooms\": 3}'.",
    )
    group.add_argument(
        "--sample-file",
        type=str,
        help="Path to a JSON file containing a single sample object.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if args.sample_file:
        sample_path = Path(args.sample_file)
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_path}")
        sample_text = sample_path.read_text(encoding="utf-8")
        sample = load_json_sample(sample_text)
    else:
        sample = load_json_sample(args.sample)

    model = load_model(str(model_path))
    prediction = predict_samples(model, [sample])
    print(json.dumps({"prediction": prediction[0]}, indent=2))


if __name__ == "__main__":
    main()

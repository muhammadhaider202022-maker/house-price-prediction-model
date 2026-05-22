import json
import os
from urllib.parse import parse_qs

from src.house_price_prediction import load_json_sample, load_model, predict_samples

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "house_price_model.joblib")

_cached_model = None


def get_model():
    global _cached_model
    if _cached_model is None:
        _cached_model = load_model(MODEL_PATH)
    return _cached_model


def json_response(status_code, body):
    reason = {
        200: "OK",
        400: "Bad Request",
        404: "Not Found",
        405: "Method Not Allowed",
        500: "Internal Server Error",
    }.get(status_code, "OK")
    status = f"{status_code} {reason}"
    headers = [("Content-Type", "application/json")]
    return status, headers, [json.dumps(body).encode("utf-8")]


def parse_request_body(environ):
    try:
        length = int(environ.get("CONTENT_LENGTH", 0))
    except (ValueError, TypeError):
        length = 0
    if length > 0:
        return environ["wsgi.input"].read(length).decode("utf-8")
    return environ["wsgi.input"].read().decode("utf-8")


def app(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET")

    if path == "/":
        status, headers, body = json_response(200, {
            "message": "House Price Prediction API",
            "routes": ["/predict"],
        })
        start_response(status, headers)
        return body

    if path == "/predict":
        if method not in ("GET", "POST"):
            status, headers, body = json_response(405, {"error": f"Method {method} not allowed"})
            start_response(status, headers)
            return body

        try:
            if method == "GET":
                query = parse_qs(environ.get("QUERY_STRING", ""))
                sample_json = query.get("sample", [None])[0]
                if sample_json is None:
                    raise ValueError("Missing 'sample' query parameter with JSON payload")
            else:
                sample_json = parse_request_body(environ)

            sample = load_json_sample(sample_json)
            model = get_model()
            prediction = predict_samples(model, [sample])

            status, headers, body = json_response(200, {"prediction": prediction[0]})
            start_response(status, headers)
            return body
        except Exception as exc:
            status, headers, body = json_response(400, {"error": str(exc)})
            start_response(status, headers)
            return body

    status, headers, body = json_response(404, {"error": "Not found"})
    start_response(status, headers)
    return body

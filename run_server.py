import argparse
from wsgiref.simple_server import make_server

from app import app


def parse_args():
    parser = argparse.ArgumentParser(description="Run the house price prediction WSGI app locally.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Starting House Price Prediction API at http://{args.host}:{args.port}")
    with make_server(args.host, args.port, app) as httpd:
        httpd.serve_forever()

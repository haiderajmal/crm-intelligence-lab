"""Dependency-free read-only JSON API for generated CRM results."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .database import CRMDatabase


def create_handler(database_path: str | Path) -> type[BaseHTTPRequestHandler]:
    database = CRMDatabase(database_path)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, payload: dict) -> None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - HTTP method name
            path = urlparse(self.path).path
            if path == "/health":
                self._send(HTTPStatus.OK, {"status": "ok", "customers": database.customer_count()})
                return
            if path == "/experiments/latest":
                result = database.latest_experiment()
                self._send(HTTPStatus.OK if result else HTTPStatus.NOT_FOUND, result or {"error": "not found"})
                return
            if path.startswith("/customers/"):
                try:
                    customer_id = int(path.rsplit("/", 1)[-1])
                except ValueError:
                    self._send(HTTPStatus.BAD_REQUEST, {"error": "customer ID must be an integer"})
                    return
                customer = database.get_customer(customer_id)
                self._send(HTTPStatus.OK if customer else HTTPStatus.NOT_FOUND, customer or {"error": "not found"})
                return
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def serve(database_path: str | Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), create_handler(database_path))
    print(f"CRM Lab API listening on http://{host}:{port}")
    server.serve_forever()

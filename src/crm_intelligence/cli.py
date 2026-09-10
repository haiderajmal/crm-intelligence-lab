"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import serve
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CRM Customer Intelligence Lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the reproducible analytics pipeline")
    run.add_argument("--customers", type=int, default=1_200)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--output", default="artifacts")
    api = subparsers.add_parser("serve", help="serve previously generated results")
    api.add_argument("--database", default="artifacts/crm_lab.sqlite3")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        result = run_pipeline(Path(args.output), args.customers, args.seed)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    serve(args.database, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

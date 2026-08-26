"""Command-line interface for Tontaube inference."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tontaube",
        description="Run the Tontaube inference server and browser interface.",
    )
    commands = parser.add_subparsers(dest="command")

    serve = commands.add_parser("serve", help="start the inference API")
    serve.add_argument(
        "--host",
        default=None,
        help="bind address (default: TTS_HOST or 127.0.0.1)",
    )
    serve.add_argument(
        "--port",
        default=None,
        type=int,
        help="HTTP port (default: TTS_PORT, PORT, or 8080)",
    )

    ui = commands.add_parser("ui", help="start the standalone browser interface")
    ui.add_argument(
        "--host",
        default=os.getenv("TTS_UI_HOST", "127.0.0.1"),
        help="bind address (default: TTS_UI_HOST or 127.0.0.1)",
    )
    ui.add_argument(
        "--port",
        default=int(os.getenv("TTS_UI_PORT", "3000")),
        type=int,
        help="HTTP port (default: TTS_UI_PORT or 3000)",
    )

    commands.add_parser(
        "preflight",
        help="download, prepare, and validate enabled models without starting the API",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return
    if args.command == "serve":
        from app.main import main as serve_main

        serve_main(host=args.host, port=args.port)
        return
    if args.command == "ui":
        from app.ui_server import serve

        serve(args.host, args.port)
        return
    if args.command == "preflight":
        from app.preflight import main as preflight_main

        preflight_main()
        return
    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()

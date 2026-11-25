"""Command-line interface for Tontaube inference."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
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
    serve.add_argument(
        "--model",
        choices=["v1.1", "flash"],
        default=None,
        help="TontaubeV1.1 or TontaubeV1.1 Flash (default: TTS_MODEL or v1.1)",
    )
    serve.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default=None,
        help="where the model runs (default: LITE_DEVICE, else the GPU when one is available)",
    )
    serve.add_argument(
        "--rate",
        type=int,
        choices=[48000, 24000],
        default=None,
        help="TontaubeV1.1 Flash: output sample rate; 24000 is faster (default: FLASH_SAMPLE_RATE or 48000)",
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

    verbalize = commands.add_parser(
        "verbalize",
        help="turn written English into spoken text with the verbalizer alone (a file, standard input, or an API)",
    )
    verbalize.add_argument("file", nargs="?", help="text file to read (default: standard input)")
    verbalize.add_argument("--serve", action="store_true", help="start the verbalizer API (POST /v1/verbalize)")
    verbalize.add_argument("--host", default=None, help="bind address (default: VERBALIZER_HOST or 127.0.0.1)")
    verbalize.add_argument("--port", default=None, type=int, help="HTTP port (default: VERBALIZER_PORT or 8081)")

    commands.add_parser(
        "preflight",
        help="download, prepare, and validate enabled models without starting the API",
    )

    quantize = commands.add_parser(
        "quantize",
        help="build a pre-quantized bundle for the lite backend (TTS_BACKEND=lite)",
        add_help=False,
    )
    quantize.add_argument("args", nargs=argparse.REMAINDER, help="see `tontaube quantize --help`")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Known noise from dependencies: SyntaxWarnings on their first import under Python 3.12, torch's weight_norm
    # deprecation, and the bundled VibeVoice code's notice that the optional APEX package is absent.
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", message=".*weight_norm", category=FutureWarning)
    logging.getLogger("VibeVoice.vibevoice.modular.modular_vibevoice_tokenizer").setLevel(logging.ERROR)
    if argv[:1] == ["quantize"]:
        # The quantizer owns its options (argparse REMAINDER cannot pass
        # leading options such as --help through a subparser).
        from app.lite.quantize.cli import main as quantize_main

        quantize_main(argv[1:])
        return
    parser = _parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return
    if args.command == "serve":
        # app.config reads these when it is first imported.
        if args.model:
            os.environ["TTS_MODEL"] = args.model
        if args.device:
            os.environ["LITE_DEVICE"] = "cuda" if args.device == "gpu" else "cpu"
        if args.rate:
            os.environ["FLASH_SAMPLE_RATE"] = str(args.rate)
        # Before anything imports torch models or downloads weights: a GPU that the
        # installed PyTorch build does not support is handled here, not after minutes
        # of downloads (see app/torch_variant.py). `preflight` does not touch the GPU.
        from app.torch_variant import check_torch_variant

        check_torch_variant(argv)

        from app.main import main as serve_main

        serve_main(host=args.host, port=args.port)
        return
    if args.command == "verbalize":
        from app.torch_variant import check_torch_variant

        check_torch_variant(argv)

        from app import verbalizer_server

        if args.serve:
            verbalizer_server.serve(args.host, args.port)
        else:
            verbalizer_server.verbalize_file(args.file)
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

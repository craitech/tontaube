#!/usr/bin/env python3
"""Extract the acoustic tokenizer from a VibeVoice-1.5B model bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.vibevoice_assets import extract_acoustic_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite {output_dir}")
    try:
        extract_acoustic_tokenizer(args.source_dir.resolve(), output_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Extracted VibeVoice acoustic tokenizer to {output_dir}")


if __name__ == "__main__":
    main()

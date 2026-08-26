#!/bin/bash
set -euo pipefail

if [[ "${REQUIRE_API_KEY:-0}" == "1" && -z "${TTS_API_KEY:-}" ]]; then
    echo "ERROR: TTS_API_KEY is required when REQUIRE_API_KEY=1" >&2
    exit 64
fi

if [[ $# -gt 0 ]]; then
    exec tontaube "$@"
fi

exec tontaube serve

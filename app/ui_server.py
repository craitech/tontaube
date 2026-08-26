"""Serve the bundled browser UI without loading the inference runtime."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from app.voice_catalog import VOICE_FILES_DIRECTORY, available_voice_files


UI_DIR = Path(__file__).with_name("ui")
DEFAULT_VOICE_DIR = Path(__file__).resolve().parent.parent / "voices"
UI_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/ui": ("index.html", "text/html; charset=utf-8"),
    "/ui/": ("index.html", "text/html; charset=utf-8"),
    "/ui/assets/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/ui/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/ui/assets/tontaube-logo.png": ("tontaube-logo.png", "image/png"),
}

_CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "connect-src 'self' http: https:; media-src 'self' blob:; "
    "img-src 'self' data:; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)


def voice_directory() -> Path:
    """Return the voice folder owned by the standalone UI process."""
    return Path(os.getenv("TTS_UI_VOICE_PATH", str(DEFAULT_VOICE_DIR))).expanduser()


def available_ui_voices(language: str) -> list[dict[str, str]]:
    """List local UI voices without exposing filesystem paths to the browser."""
    return [
        {
            "name": path.stem,
            "url": f"/ui/voice-files/{quote(path.name)}",
            **({"style": metadata["style"]} if "style" in metadata else {}),
        }
        for path, metadata in available_voice_files(voice_directory(), language)
    ]


def resolve_ui_voice_file(filename: str) -> Path | None:
    """Resolve a catalogued voice while preventing traversal and symlink escapes."""
    filename = unquote(filename)
    if Path(filename).name != filename:
        return None
    directory = voice_directory().resolve()
    files_directory = (directory / VOICE_FILES_DIRECTORY).resolve()
    candidate = (files_directory / filename).resolve()
    if (
        candidate.parent != files_directory
        or not any(candidate == path for path, _ in available_voice_files(directory))
        or not candidate.is_file()
    ):
        return None
    return candidate


class UIRequestHandler(BaseHTTPRequestHandler):
    """Serve the packaged UI and its local sample voices."""

    def _send_bytes(self, content: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlsplit(self.path)
        if parsed.path == "/ui/voices":
            language = parse_qs(parsed.query).get("language", ["english"])[0]
            try:
                content = json.dumps(
                    {"voices": available_ui_voices(language)},
                    separators=(",", ":"),
                ).encode("utf-8")
            except ValueError as exc:
                content = json.dumps({"error": str(exc)}).encode("utf-8")
                self._send_bytes(content, "application/json; charset=utf-8", 422)
                return
            self._send_bytes(content, "application/json; charset=utf-8")
            return

        voice_prefix = "/ui/voice-files/"
        if parsed.path.startswith(voice_prefix):
            voice = resolve_ui_voice_file(parsed.path[len(voice_prefix):])
            if voice is None:
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(voice.name)[0] or "application/octet-stream"
            self._send_bytes(voice.read_bytes(), content_type)
            return

        route = UI_ROUTES.get(parsed.path)
        if route is None:
            self.send_error(404)
            return

        filename, content_type = route
        content = (UI_DIR / filename).read_bytes()
        if filename == "index.html":
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", _CONTENT_SECURITY_POLICY)
            self.end_headers()
            self.wfile.write(content)
            return
        self._send_bytes(content, content_type)

    def log_message(self, format: str, *args: object) -> None:
        if os.getenv("TTS_DEBUG", "0") == "1" or os.getenv(
            "TTS_LOG_LEVEL", "INFO"
        ).strip().upper() == "DEBUG":
            print(f"{self.address_string()} - {format % args}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the Tontaube browser UI without loading any models."
    )
    parser.add_argument(
        "--host",
        default=os.getenv("TTS_UI_HOST", "127.0.0.1"),
        help="bind address (default: 127.0.0.1 or TTS_UI_HOST)",
    )
    parser.add_argument(
        "--port",
        default=int(os.getenv("TTS_UI_PORT", "3000")),
        type=int,
        help="HTTP port (default: 3000 or TTS_UI_PORT)",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    serve(args.host, args.port)


def serve(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), UIRequestHandler)
    host, port = server.server_address[:2]
    print(f"Tontaube UI available at http://{host}:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

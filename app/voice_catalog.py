"""Discover sample voices and their optional catalogue metadata."""

from __future__ import annotations

import json
from pathlib import Path


LANGUAGE_ALIASES: dict[str, str] = {
    "en": "english", "english": "english",
    "de": "german", "german": "german",
    "es": "spanish", "spanish": "spanish",
    "fr": "french", "french": "french",
    "it": "italian", "italian": "italian",
    "nl": "dutch", "dutch": "dutch",
    "pt": "portuguese", "portuguese": "portuguese",
}
SUPPORTED_VOICE_STYLES: tuple[str, ...] = (
    "audiobook",
    "conversational",
    "agentic",
)
VOICE_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}
)
VOICE_MANIFEST_FILENAME = "manifest.json"
VOICE_FILES_DIRECTORY = "samples"


def _voice_manifest(directory: Path) -> dict[str, dict[str, str]]:
    path = directory / VOICE_MANIFEST_FILENAME
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read voice manifest {path}: {exc}") from exc

    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    raw_voices = document.get("voices")
    if document.get("version") != 1 or not isinstance(raw_voices, dict):
        raise ValueError(f"{path} must contain version 1 and a voices object")

    voices: dict[str, dict[str, str]] = {}
    for filename, raw_metadata in raw_voices.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"invalid filename in {path}: {filename!r}")
        if not isinstance(raw_metadata, dict):
            raise ValueError(f"metadata for {filename!r} in {path} must be an object")

        metadata: dict[str, str] = {}
        language = raw_metadata.get("language")
        if language is not None:
            if not isinstance(language, str):
                raise ValueError(f"language for {filename!r} in {path} must be a string")
            canonical = LANGUAGE_ALIASES.get(language.strip().lower())
            if canonical is None:
                raise ValueError(f"unsupported language for {filename!r} in {path}: {language!r}")
            metadata["language"] = canonical

        style = raw_metadata.get("style")
        if style is not None:
            if not isinstance(style, str):
                raise ValueError(f"style for {filename!r} in {path} must be a string")
            style = style.strip().lower()
            if style not in SUPPORTED_VOICE_STYLES:
                raise ValueError(f"unsupported style for {filename!r} in {path}: {style!r}")
            metadata["style"] = style
        voices[filename] = metadata
    return voices


def available_voice_files(
    directory: Path,
    language: str | None = None,
) -> list[tuple[Path, dict[str, str]]]:
    """Return flat voice files, optionally filtered by catalogued language."""
    if not directory.is_dir():
        return []
    canonical_language = None
    if language is not None:
        canonical_language = LANGUAGE_ALIASES.get(language.strip().lower())
        if canonical_language is None:
            raise ValueError(f"unsupported language: {language}")

    manifest = _voice_manifest(directory)
    files_directory = directory / VOICE_FILES_DIRECTORY
    if not files_directory.is_dir():
        return []
    voices = []
    for path in sorted(files_directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.lower() not in VOICE_AUDIO_EXTENSIONS:
            continue
        metadata = manifest.get(path.name, {})
        declared_language = metadata.get("language")
        if (
            canonical_language is not None
            and declared_language is not None
            and declared_language != canonical_language
        ):
            continue
        voices.append((path, metadata))
    return voices

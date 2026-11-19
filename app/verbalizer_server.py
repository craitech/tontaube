"""The verbalizer on its own: written English in, spoken text out, without loading the speech models.

`tontaube verbalize FILE` prints the spoken form of a text file (or of standard input);
`tontaube verbalize --serve` starts an API with one endpoint:

    POST /v1/verbalize   {"text": "It costs $5."}        -> {"text": "It costs five dollars."}
                         {"text": ["first", "second"]}   -> {"text": ["...", "..."]}

Texts of one request, and requests that arrive together, share model passes.
"""
from __future__ import annotations

import asyncio
import os
import secrets
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from app.config import MAX_REQUEST_TEXT_CHARS, REQUIRE_API_KEY, TTS_API_KEY, ServerConfig


def load_verbalizer():
    """The verbalizer as the speech server would load it (same model and settings)."""
    from app.model_sources import resolve_verbalizer_model_path
    from app.verbalizer import Verbalizer

    config = ServerConfig()
    return Verbalizer(
        resolve_verbalizer_model_path(config.verbalization_model_path),
        gpu_util=config.verbalization_gpu_util,
        max_num_seqs=config.verbalization_max_num_seqs,
        growth_guard=config.verbalizer_growth_guard,
        draft=config.verbalizer_draft,
        guard=config.verbalizer_guard,
        backend=config.backend,
        lite_quant=config.lite_verbalizer_quant,
    )


class VerbalizeRequest(BaseModel):
    text: str | list[str]

    @field_validator("text")
    @classmethod
    def _check_text(cls, value):
        texts = value if isinstance(value, list) else [value]
        if not texts or not any(text.strip() for text in texts):
            raise ValueError("text must not be empty")
        if sum(len(text) for text in texts) > MAX_REQUEST_TEXT_CHARS:
            raise ValueError(f"text must contain at most {MAX_REQUEST_TEXT_CHARS} characters")
        return value


def create_app() -> FastAPI:
    state = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if REQUIRE_API_KEY and not TTS_API_KEY:
            raise RuntimeError("TTS_API_KEY is required when REQUIRE_API_KEY=1")
        state["verbalizer"] = load_verbalizer()
        yield

    app = FastAPI(title="Tontaube verbalizer", lifespan=lifespan)

    @app.middleware("http")
    async def api_key_guard(request: Request, call_next):
        if REQUIRE_API_KEY and request.method != "OPTIONS" and request.url.path != "/healthz":
            provided_key = request.headers.get("x-api-key")
            if not provided_key or not secrets.compare_digest(provided_key, TTS_API_KEY):
                return JSONResponse(content={"error": "unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED)
        return await call_next(request)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/v1/verbalize")
    async def verbalize(req: VerbalizeRequest):
        verbalizer = state["verbalizer"]
        if isinstance(req.text, list):
            return {"text": list(await asyncio.gather(*[verbalizer.verbalize(text) for text in req.text]))}
        return {"text": await verbalizer.verbalize(req.text)}

    return app


def serve(host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=host or os.getenv("VERBALIZER_HOST", "127.0.0.1"),
        port=port or int(os.getenv("VERBALIZER_PORT", "8081")),
    )


def verbalize_file(path: str | None) -> None:
    """Print the spoken form of a text file, or of standard input when no file is given."""
    text = open(path, encoding="utf-8").read() if path else sys.stdin.read()
    verbalizer = load_verbalizer()
    sys.stdout.write(asyncio.run(verbalizer.verbalize(text)))
    sys.stdout.write("\n")

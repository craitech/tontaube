"""Core TTS pipeline: text segments, prompts, codebook engines (app/engines.py), codec, retries."""
import asyncio
import gc
import hashlib
import math
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch

os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

from app.config import (
    ServerConfig,
    END_OF_SPEECH,
    TEXT_MARKER,
    AUDIO_MARKER,
    UTT_END,
    UTT_CONT,
    SEMANTIC_VOCAB_SIZE,
    ACOUSTIC_VOCAB_SIZE,
    DUALCODEC_SAMPLE_RATE,
    DC_SILENCE_TOKEN,
    SILENCE_TOKEN_ID,
    COLLAPSE_SILENCE_TOKEN_IDS,
    ORDINARY_SILENCE_TOKEN_IDS,
    SILENCE_TOKEN_IDS,
    DEFAULT_LANGUAGE,
    DEFAULT_TAG,
    DEFAULT_STREAMING_INITIAL_SECONDS,
    DUALCODEC_HZ,
    DUALCODEC_MODEL_TYPE,
    MAX_VOICE_PROMPT_TOKENS,
    STREAMING_SECONDS_QUANTUM,
    build_system_prompt,
)
from app.model.positions import get_codebook_offset
from app.controls import ControlCatalog, render_system_tag
from app.cb0_ctc import normalize_text
from app.text import strip_special_token_strings
from app.cb0_guard import (
    PrevAudioChain,
    RateWindow,
    SegmentGuard,
    load_recognizer,
    longest_run,
    parse_cb0_header,
    retry_seed,
)
from app.quality_gate import build_quality_gate
from app.utils import (
    check_boundary_tokens,
    check_cb0_window_ids,
    load_tokenizer,
    tokenize_text,
    encode_voice_prompt,
    resolve_voice_prompt,
    truncate_voice_prompt,
    codes_to_audio_tensor,
    get_dc_inference,
    free_memory,
)
from app.dualcodec_utils import (
    decode_chunked_with_context, split_at_markers, VV_HZ, VV_HOP_SAMPLES, code_samples, finish_request_end,
    apply_output_headroom, decode_with_prefix_context, start_fade,
)
from app.stream_lead_trim import trim_stream_lead
from app.stream_early_check import EARLY_LEAD_SILENCE_TOKEN, active_fraction
from app.vv_adapter import (
    CONTEXT_CODES as VV_ADAPTER_CONTEXT_CODES, DC_HZ as VV_ADAPTER_DC_HZ, N_ACOUSTIC as VV_ADAPTER_N_ACOUSTIC,
    activate_vv_adapter, check_grid,
    VV_ADAPTER_FILE, check_stream_lookahead, correct_stream_frames, load_vv_adapter, stream_tail_audio,
)
from app.vibevoice_utils import (
    load_vibevoice_tokenizer,
    vv_encode,
    vv_decode_streaming,
    VVDecodeEngine,
)
from app.model_sources import (
    read_codebook_metadata,
    resolve_codebook_model_paths,
    resolve_runtime_model_paths,
    resolve_verbalizer_model_path,
)
from app.engines import SamplingParams, TokensPrompt, build_codebook_engines, shutdown_engine
from app.logging_utils import get_logger

logger = get_logger("inference")

@dataclass
class _CodebookCtx:
    """Resolved token IDs and engine for a single codebook."""
    tokenizer: object
    engine: object
    n_codebooks: int
    pad_id: int
    dc_base_id: int
    eos_id: int
    audio_marker_id: int
    hz: float


@dataclass
class _StreamingState:
    """Mutable accumulator state for streaming generation."""
    seg_offset: int
    n_pfx: int
    cb0_segments: list[list[int]]         # [prefix_slot?] + [seg0, seg1, ...]
    acoustic_prefix_per_cb: list[list[int]]  # per-codebook prefix codes
    ac_segs: list[list[list[int]]]        # ac_segs[cb_idx][seg_idx] = codes
    accumulated_sem: list[int]
    accumulated_ac: list[list[int]]       # accumulated_ac[cb_idx] = flat codes
    n_codebooks: int


def _extract_acoustic_codes(
    output_ids: list[int], n_tokens: int, start_id: int, pad_id: int,
) -> list[int]:
    """Extract local acoustic codes from raw output, padding to n_tokens."""
    codes = []
    for oid in output_ids[:n_tokens]:
        if oid == pad_id:
            continue
        if start_id <= oid < start_id + ACOUSTIC_VOCAB_SIZE:
            codes.append(oid - start_id)
        else:
            codes.append(0)
    while len(codes) < n_tokens:
        codes.append(0)
    return codes


class TTSEngine:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.engines: list = []
        self.tokenizers: list = []
        self.dc_inference = None
        self.vv_tokenizer = None
        self.vv_engine: VVDecodeEngine | None = None
        self.vv_adapter = None  # VV_ADAPTER_CKPT (app/vv_adapter.py); None = off
        # End tokens: the CB0 bundle's (<|utt_end|>, <|utt_cont|>) ids when it declares them; None = a legacy bundle.
        self.boundary_token_ids: tuple[int, int] | None = None
        self.prompt_semantic = None
        self.prompt_acoustic = None
        self._voice_prompt_cache: OrderedDict[str, tuple[list, list]] = OrderedDict()
        self._voice_prompt_cache_max = 128
        # Generation encodes on the event loop, /encode_voice in a worker thread: one DualCodec encode at a time.
        self._voice_prompt_lock = threading.Lock()
        self.verbalizer = None
        # Slider catalog of a control-enabled CB0 bundle (controls.json), else None.
        self.controls: ControlCatalog | None = None
        # app/cb0_guard.py: rate factors for the chars/s window and the chunk length,
        # and the CB0-CTC guard's recogniser (loaded in load()); None = off.
        rate_factors = (
            RateWindow.from_config(config)
            if config.cb0_rate_aware_cps or config.cb0_rate_aware_chunk_chars else None
        )
        self.rate_window = rate_factors if config.cb0_rate_aware_cps else None
        self.rate_chunk_window = rate_factors if config.cb0_rate_aware_chunk_chars else None
        self.cb0_ctc = None
        # app/quality_gate.py: the /predict quality gate (built in load()); None = off.
        self.quality_gate = None

    def _n_codebooks(self) -> int:
        return len(self.engines)

    def _cb_ctx(self, codebook: int) -> _CodebookCtx:
        """Resolve token IDs, engine, and config for a codebook."""
        tokenizer = self.tokenizers[codebook]
        return _CodebookCtx(
            tokenizer=tokenizer,
            engine=self.engines[codebook],
            n_codebooks=self._n_codebooks(),
            pad_id=tokenizer.pad_token_id,
            dc_base_id=tokenizer.convert_tokens_to_ids('<|dc_0_0|>'),
            eos_id=tokenizer.convert_tokens_to_ids(END_OF_SPEECH),
            audio_marker_id=tokenizer.convert_tokens_to_ids(AUDIO_MARKER),
            hz=DUALCODEC_HZ,
        )

    # ------------------------------------------------------------------
    # Shared helpers (DRY: each piece of logic exists in exactly one place)
    # ------------------------------------------------------------------

    def _text_window(
        self,
        seg_idx: int,
        segments: list[str],
        use_context: bool,
        prefix: dict | None = None,
        system_tag: str = '',
        lookahead_chars: int | None = 50,
        boundary_total_segments: int | None = None,
        restart: bool = False,
        boundary_tokens: bool = False,
        drop_space: bool = False,
    ) -> tuple[str, int, int]:
        """Build text window for a segment.  Returns (window_text, win_start, win_end).

        ``restart`` (with ``use_context``; CB0_DROP_BAD_PREV_AUDIO): a window shaped
        like segment 0's but starting at ``seg_idx``: the segment plus the next one's lookahead,
        no previous text. Boundary cues stay those of the segment's real position.

        ``boundary_tokens`` (CB0 only; ServerConfig.cb0_boundary_tokens): a token window. The
        segment ``seg_idx`` ends in UTT_END if it is the request's last, else in UTT_CONT; no
        segment gets the trailing '\\n', and the previous text and the lookahead carry no token.
        ``drop_space`` (token windows only): no leading ' ' on any segment, and none between a
        prefix and the window.
        """
        if drop_space and not boundary_tokens:
            raise ValueError("drop_space applies to token windows only; legacy windows never change")
        n = len(segments)
        boundary_total = n if boundary_total_segments is None else boundary_total_segments
        if not use_context:
            win_start, win_end = seg_idx, seg_idx + 1
        elif seg_idx == 0 or restart:
            win_start, win_end = seg_idx, min(seg_idx + 2, n)
        else:
            win_start = max(0, seg_idx - 1)
            win_end = min(seg_idx + 2, n)

        # Boundary cues distinguish the actual beginning and end of a request
        # from internal chunks. Do this here (after API-side
        # sanitizing/verbalizing), because those preprocessing steps call
        # ``strip()`` and would otherwise erase the boundary signal.
        win_segs = [s.strip() for s in segments[win_start:win_end]]
        next_offset = seg_idx - win_start + 1
        if lookahead_chars is not None:
            for i in range(next_offset, len(win_segs)):
                s = win_segs[i]
                if len(s) > lookahead_chars:
                    cut = s.rfind(' ', 0, lookahead_chars)
                    win_segs[i] = s[:cut] if cut > 0 else s[:lookahead_chars]

        for local_idx, segment in enumerate(win_segs):
            global_idx = win_start + local_idx
            if global_idx > 0 and not drop_space:
                segment = ' ' + segment
            if not boundary_tokens:
                if global_idx == boundary_total - 1:
                    segment += '\n'
            elif global_idx == seg_idx:
                segment += UTT_END if global_idx == boundary_total - 1 else UTT_CONT
            win_segs[local_idx] = segment

        # Every non-initial chunk already carries its one leading space (none under drop_space).
        window_text = TEXT_MARKER.join(win_segs)
        if prefix:
            separator = '' if drop_space or window_text.startswith((' ', '\n')) else ' '
            window_text = prefix['text'].strip() + TEXT_MARKER + separator + window_text
        window_text = system_tag + window_text
        return window_text, win_start, win_end

    def _resolve_params(
        self,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        voice_path: str | None = None,
        voice_audio_b64: str | None = None,
        voice_tokens: list[list[int]] | None = None,
        prompt_max_tokens: int | None = None,
        acoustic_temperature: float | None = None,
        acoustic_top_k: int | None = None,
        language: str = DEFAULT_LANGUAGE,
        tag: str = DEFAULT_TAG,
        use_system_tag: bool = True,
    ) -> tuple[dict, tuple, str, float, int]:
        """Resolve and default all generation parameters.

        Returns (sampling_dict, voice_prompt, system_tag, acoustic_temperature, acoustic_top_k).
        """
        acoustic_temperature = (
            acoustic_temperature
            if acoustic_temperature is not None
            else self.config.acoustic_temperature
        )
        acoustic_top_k = acoustic_top_k if acoustic_top_k is not None else -1
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        top_k = top_k if top_k is not None else self.config.top_k
        frequency_penalty = frequency_penalty if frequency_penalty is not None else self.config.frequency_penalty
        if use_system_tag:
            system_prompt = build_system_prompt(language, tag)
            system_tag = f'<|im_start|>{system_prompt}<|im_end|>\n'
        else:
            system_tag = ''

        prompt_cap = (
            prompt_max_tokens
            if prompt_max_tokens is not None
            else self.config.prompt_max_tokens
        )
        if prompt_cap < 1:
            raise ValueError("prompt_max_tokens must be at least 1")
        voice_prompt = truncate_voice_prompt(
            self.prompt_semantic,
            self.prompt_acoustic,
            prompt_cap,
        )
        if voice_tokens and voice_tokens[0]:
            voice_prompt = truncate_voice_prompt(
                voice_tokens[0],
                voice_tokens[1:],
                prompt_cap,
            )
        elif voice_audio_b64 or voice_path:
            voice_prompt = self._resolve_voice_prompt(
                voice_audio_b64,
                voice_path,
                prompt_max_tokens=prompt_cap,
            )

        sampling = dict(
            temperature=temperature, top_p=top_p, top_k=top_k,
            frequency_penalty=frequency_penalty, seed=seed,
        )
        return sampling, voice_prompt, system_tag, acoustic_temperature, acoustic_top_k

    def _cb0_system_tags(
        self,
        system_tag: str,
        n_segments: int,
        language: str,
        tag: str,
        use_system_tag: bool,
        controls=None,
        segment_controls: list | None = None,
    ) -> tuple[str, list[str]]:
        """CB0 system headers with slider tokens: (request-level tag, one tag per segment).

        CB1-3 keep ``system_tag`` unchanged. Without controls both equal ``system_tag``.
        """
        if controls is None and segment_controls is None:
            return system_tag, [system_tag] * n_segments
        if self.controls is None:
            raise ValueError(
                "slider controls need a CB0 bundle with controls.json (set CB0_MODEL_PATH)"
            )
        if not use_system_tag:
            raise ValueError("slider controls need use_system_tag=True (they live in the header)")
        if segment_controls is not None and len(segment_controls) != n_segments:
            raise ValueError(
                f"segment_controls has {len(segment_controls)} entries for {n_segments} segments"
            )
        system_prompt = build_system_prompt(language, tag)
        request_tag = render_system_tag(system_prompt, self.controls.render(controls))
        segment_tags = [
            render_system_tag(
                system_prompt,
                self.controls.render(self.controls.merge(
                    controls, segment_controls[i] if segment_controls is not None else None,
                )),
            )
            for i in range(n_segments)
        ]
        return request_tag, segment_tags

    def chunk_char_scales(
        self,
        n_segments: int,
        language: str,
        tag: str,
        use_system_tag: bool,
        controls=None,
        segment_controls: list | None = None,
    ) -> list[float] | None:
        """Per-segment scale (<= 1) of the text splitter's chunk lengths, so slow-rate
        pieces stay within the CB0 token cap: the rate factor of each segment's CB0 header (as for
        the chars/s window), capped at 1. None = unscaled (flag off or no sliders).
        """
        if self.rate_chunk_window is None or (controls is None and segment_controls is None):
            return None
        try:
            _, tags = self._cb0_system_tags(
                "", n_segments, language, tag, use_system_tag,
                controls=controls, segment_controls=segment_controls,
            )
        except ValueError:
            return None  # reported after splitting (app.main._require_controls)
        return [min(1.0, self.rate_chunk_window.factor(*parse_cb0_header(t)[1:])) for t in tags]

    async def _resolve_prefix(
        self,
        prefix_text: str | None,
        prefix_tokens: list[list[int]] | None,
        sampling: dict,
        voice_prompt: tuple,
        max_new_tokens: int | None,
        acoustic_temperature: float,
        acoustic_top_k: int,
        system_tag: str = '',
        silence_logit_bias: float | None = None,
        semantic_system_tag: str | None = None,
    ) -> tuple[dict | None, dict[int, list[int]]]:
        """Generate or load prefix context. Returns (prefix_dict, prefix_acoustic_codes)."""
        # End tokens: the client's prefix text is not sanitized; strip literal <|...|> strings only.
        prefix_text = strip_special_token_strings(prefix_text) if prefix_text else prefix_text
        if not prefix_text:
            return None, {}
        if semantic_system_tag is None:
            semantic_system_tag = system_tag

        ctx0 = self._cb_ctx(0)
        n_codebooks = self._n_codebooks()
        prefix_acoustic_codes: dict[int, list[int]] = {}

        if prefix_tokens:
            prefix_sem_codes = prefix_tokens[0]
            prev_pfx_acoustic: list[list[int]] = []
            for cb in range(1, n_codebooks):
                if cb < len(prefix_tokens) and prefix_tokens[cb]:
                    codes = prefix_tokens[cb]
                else:
                    codes = await self._generate_acoustic(
                        [prefix_text], semantic_codes=prefix_sem_codes,
                        marker_indices=[], prev_acoustic=prev_pfx_acoustic,
                        codebook=cb, voice_prompt=voice_prompt,
                        acoustic_temperature=acoustic_temperature,
                        acoustic_top_k=acoustic_top_k, system_tag=system_tag,
                    )
                prefix_acoustic_codes[cb] = codes
                prev_pfx_acoustic.append(codes)
            logger.debug("Using %d provided semantic prefix tokens", len(prefix_sem_codes))
        else:
            logger.debug("Generating context prefix for %r", prefix_text)
            prefix_sem_codes, _ = await self._generate_semantic(
                [prefix_text], sampling=sampling, voice_prompt=voice_prompt,
                max_new_tokens=max_new_tokens, system_tag=semantic_system_tag,
                silence_logit_bias=silence_logit_bias,
            )
            prev_pfx_acoustic: list[list[int]] = []
            for cb in range(1, n_codebooks):
                pfx_cb_codes = await self._generate_acoustic(
                    [prefix_text],
                    semantic_codes=prefix_sem_codes,
                    marker_indices=[],
                    prev_acoustic=prev_pfx_acoustic,
                    codebook=cb,
                    voice_prompt=voice_prompt,
                    acoustic_temperature=acoustic_temperature,
                    acoustic_top_k=acoustic_top_k,
                    system_tag=system_tag,
                )
                prefix_acoustic_codes[cb] = pfx_cb_codes
                prev_pfx_acoustic.append(pfx_cb_codes)

        prefix_sem_global = [ctx0.dc_base_id + c for c in prefix_sem_codes]
        prefix = {
            'text': prefix_text.strip(),
            'semantic_codes': prefix_sem_codes,
            'semantic_global': prefix_sem_global,
        }
        logger.debug("Context prefix contains %d semantic tokens", len(prefix_sem_codes))
        return prefix, prefix_acoustic_codes

    def _build_streaming_state(
        self,
        prefix: dict | None,
        prefix_acoustic_codes: dict[int, list[int]],
    ) -> _StreamingState:
        """Initialize streaming accumulators from prefix."""
        n_codebooks = self._n_codebooks()
        seg_offset = 1 if prefix else 0
        n_pfx = len(prefix['semantic_codes']) if prefix else 0

        cb0_segments: list[list[int]] = []
        if prefix:
            cb0_segments.append(list(prefix['semantic_codes']))

        acoustic_prefix_per_cb: list[list[int]] = []
        for cb in range(1, n_codebooks):
            if prefix and prefix_acoustic_codes.get(cb) is not None:
                pfx = list(prefix_acoustic_codes[cb][:n_pfx])
                pfx += [0] * (n_pfx - len(pfx))
                acoustic_prefix_per_cb.append(pfx)
            else:
                acoustic_prefix_per_cb.append([])

        ac_segs: list[list[list[int]]] = [[] for _ in range(n_codebooks - 1)]
        if prefix:
            for cb_idx in range(n_codebooks - 1):
                cb = cb_idx + 1
                pfx = list((prefix_acoustic_codes.get(cb) or [])[:n_pfx])
                pfx += [0] * (n_pfx - len(pfx))
                ac_segs[cb_idx].append(pfx)

        return _StreamingState(
            seg_offset=seg_offset,
            n_pfx=n_pfx,
            cb0_segments=cb0_segments,
            acoustic_prefix_per_cb=acoustic_prefix_per_cb,
            ac_segs=ac_segs,
            accumulated_sem=[],
            accumulated_ac=[[] for _ in range(n_codebooks - 1)],
            n_codebooks=n_codebooks,
        )

    async def _generate_all_acoustic_for_segment(
        self,
        state: _StreamingState,
        seg_idx: int,
        n_real_segs: int,
        segments: list[str],
        voice_prompt: tuple,
        prefix: dict | None,
        acoustic_temperature: float,
        acoustic_top_k: int,
        system_tag: str = '',
        early_forced_prefixes: list[list[int]] | None = None,
    ) -> None:
        """Generate all acoustic codebooks for one segment, updating state in-place."""
        for cb in range(1, state.n_codebooks):
            cb_idx = cb - 1
            prev_cb_code_segments = state.ac_segs[:cb_idx]
            # Acoustic codebooks (cb1+) do NOT use cross-segment context
            prev_same_cb_segment = None
            forced = (
                early_forced_prefixes[cb_idx]
                if early_forced_prefixes and cb_idx < len(early_forced_prefixes)
                else None
            )
            seg_codes = await self._generate_acoustic_segment(
                real_idx=seg_idx,
                n_real_segs=n_real_segs,
                segments=segments,
                codebook=cb,
                cb0_segments=state.cb0_segments,
                prev_cb_code_segments=prev_cb_code_segments,
                prev_same_cb_segment=prev_same_cb_segment,
                voice_prompt=voice_prompt,
                prefix=prefix,
                accumulated_prefix=state.acoustic_prefix_per_cb[cb_idx],
                seg_offset=state.seg_offset,
                forced_prefix_codes=forced,
                acoustic_temperature=acoustic_temperature,
                acoustic_top_k=acoustic_top_k,
                system_tag=system_tag,
                boundary_total_segments=len(segments),
            )
            state.ac_segs[cb_idx].append(seg_codes)
            state.accumulated_ac[cb_idx].extend(seg_codes)

    async def _emit_vv_incremental(
        self,
        state: _StreamingState,
        total_vv_emitted: int,
        vv_slot: int,
        dc_hz: float,
        chunk_s: float,
        flush: bool = False,
        vv_adapter=None,
    ) -> tuple['torch.Tensor | None', int]:
        """DC decode -> VV encode -> VV decode. Returns (audio_chunk_or_None, new_total_vv_emitted).

        vv_adapter (STREAM_VV_ADAPTER, _stream_vv_adapter): corrects the new VibeVoice frames before the decode
        (app/vv_adapter.correct_stream_frames); the flush gets /predict's tail fix and keeps its length with the adapter
        off. None: unchanged."""
        pad = self.config.streaming_dc_pad_tokens
        effective_dc = len(state.accumulated_sem) - pad

        if not flush:
            if effective_dc <= 0:
                return None, total_vv_emitted
            target_seconds = int(effective_dc / dc_hz / chunk_s) * chunk_s
            target_vv_frames = int(target_seconds * VV_HZ)
            if target_vv_frames <= total_vv_emitted:
                return None, total_vv_emitted
        else:
            target_vv_frames = None  # emit all remaining

        # Re-encode from `first_token` on. Without a configured context this is
        # the whole stream; with one, decoding starts that many seconds before
        # the first unemitted frame (and no later than the VibeVoice adapter's
        # left context). Emission boundaries are whole 0.4 s units (5 DualCodec
        # tokens = 3 VibeVoice frames), so frame indices map exactly.
        first_token = 0
        context_s = self.config.streaming_decode_context_seconds
        if context_s is not None and total_vv_emitted % 3 == 0:
            first_token = max(0, total_vv_emitted * 5 // 3 - int(round(context_s * dc_hz)))
            if vv_adapter is not None:
                first_token = min(first_token, max(0, total_vv_emitted // 3 * 5 - VV_ADAPTER_CONTEXT_CODES))
            first_token -= first_token % 5
        frame_offset = first_token * 3 // 5
        with torch.no_grad():
            audio = codes_to_audio_tensor(
                self.dc_inference,
                state.accumulated_sem[first_token:],
                [codes[first_token:] for codes in state.accumulated_ac],
            )
        n_codes = len(state.accumulated_sem)
        if not n_codes:
            vv_adapter = None  # no codes, nothing to correct (as /predict)
        tail_frames_off = None
        if flush and vv_adapter is not None:
            # the adapter's tail fix: encode the codes' audio padded to the adapter's frame grid (3 * ceil(K / 5)
            # frames); the encoder alone pads the last, partial frame (ceil), which is the flush's length without it
            tail_frames_off = frame_offset - (-audio.shape[-1] // VV_HOP_SAMPLES)
            audio = stream_tail_audio(audio, n_codes - first_token)
        frames = vv_encode(self.vv_tokenizer, audio)

        def new_frames(end: int) -> torch.Tensor:
            """The stream's frames [total_vv_emitted, end) to decode: the encoder's, or corrected by the adapter."""
            if vv_adapter is None or end <= total_vv_emitted:
                return frames[:, total_vv_emitted - frame_offset:end - frame_offset]
            return correct_stream_frames(
                vv_adapter, self.dc_inference, state.accumulated_sem, state.accumulated_ac, audio, frames,
                total_vv_emitted, end, final=flush, first_code=first_token,
            )

        if flush:
            audio_chunk = await self.vv_engine.decode(new_frames(frame_offset + frames.shape[1]), vv_slot)
            if tail_frames_off is not None and not self.config.decode_end_fix:
                # the flush's length with the adapter off (with DECODE_END_FIX the caller ends it at the codes' length)
                audio_chunk = audio_chunk[..., :max(0, tail_frames_off - total_vv_emitted) * VV_HOP_SAMPLES]
            new_emitted = frame_offset + frames.shape[1]
        else:
            logger.debug(
                "Streaming decode: %d DualCodec tokens, VibeVoice frames %d:%d",
                len(state.accumulated_sem),
                total_vv_emitted,
                target_vv_frames,
            )
            audio_chunk = await self.vv_engine.decode(new_frames(target_vv_frames), vv_slot)
            new_emitted = target_vv_frames

        if audio_chunk.shape[-1] > 0:
            return audio_chunk, new_emitted
        return None, new_emitted

    @staticmethod
    def _append_trailing_silence(
        semantic_codes: list[int],
        acoustic_codes: list[list[int]],
        n_silence: int,
    ) -> None:
        """Append trailing silence tokens to avoid cut-off."""
        if n_silence > 0:
            semantic_codes.extend([DC_SILENCE_TOKEN] * n_silence)
            for ac in acoustic_codes:
                ac.extend([0] * n_silence)

    def _checked_vv_adapter(self, path: str, sha256: str):
        vv_adapter = load_vv_adapter(path, sha256, self.config.vv_adapter_allow_unlisted)
        check_grid("the decode context (chunk_context_seconds)",
                   int(self.config.chunk_context_seconds * VV_ADAPTER_DC_HZ))
        if self.config.stream_vv_adapter:
            check_stream_lookahead(vv_adapter, self.config.streaming_dc_pad_tokens)
        return vv_adapter

    async def load(self):
        # VV_ADAPTER_CKPT: check the VibeVoice adapter before anything else loads (the file,
        # VV_ADAPTER_SHA256, the approved list, the checkpoint format; app/vv_adapter.py), so a bad setting stops the
        # server in seconds, not after the vLLM startup. The checked model waits on the CPU until VibeVoice is loaded.
        vv_adapter = None
        if self.config.vv_adapter_ckpt:
            vv_adapter = self._checked_vv_adapter(self.config.vv_adapter_ckpt, self.config.vv_adapter_sha256)

        # Load verbalization model first (fast, surfaces config errors early)
        if self.config.enable_verbalization:
            from app.verbalizer import Verbalizer
            verbalizer_path = resolve_verbalizer_model_path(
                self.config.verbalization_model_path
            )
            logger.info("Loading verbalization model from %s", verbalizer_path)
            self.verbalizer = Verbalizer(
                verbalizer_path,
                gpu_util=self.config.verbalization_gpu_util,
                max_num_seqs=self.config.verbalization_max_num_seqs,
                growth_guard=self.config.verbalizer_growth_guard,
                draft=self.config.verbalizer_draft,
                guard=self.config.verbalizer_guard,
                backend=self.config.backend,
                lite_quant=self.config.lite_verbalizer_quant,
            )
            logger.info("Verbalization model loaded")

        model_paths = list(
            resolve_codebook_model_paths(self.config.codebook_model_dirs)
        )
        bundled_adapter = Path(model_paths[0]).parent / VV_ADAPTER_FILE
        if vv_adapter is None and not self.config.vv_adapter_ckpt and bundled_adapter.is_file():
            # the model root's decoder adapter: accepted only if its hash is on the approved list
            sha = hashlib.sha256(bundled_adapter.read_bytes()).hexdigest()
            vv_adapter = self._checked_vv_adapter(str(bundled_adapter), sha)
        runtime_paths = resolve_runtime_model_paths(
            include_vibevoice=self.config.enable_vibevoice
        )
        logger.info("Loading %d Tontaube codebook models", len(self.config.codebook_model_dirs))
        expected_n_codebooks = len(model_paths)
        for cb, (model_dir_name, model_path) in enumerate(
            zip(self.config.codebook_model_dirs, model_paths)
        ):
            target_codebook, n_codebooks = read_codebook_metadata(model_path)
            if target_codebook != cb:
                raise ValueError(
                    f"{model_path} targets cb{target_codebook}, expected cb{cb}"
                )
            if n_codebooks != expected_n_codebooks:
                raise ValueError(
                    f"{model_path} declares {n_codebooks} codebooks, expected "
                    f"{expected_n_codebooks} for this stack"
                )
            tokenizer = load_tokenizer(model_path)
            if cb == 0:
                self.controls = ControlCatalog.load(model_path)
                if self.controls is not None:
                    self.controls.check_tokenizer(tokenizer)
                    logger.info("CB0 slider controls: %s", self.controls.describe())
            boundary_ids = check_boundary_tokens(
                model_path, tokenizer, cb,
                slider_tokens=sorted(self.controls.all_tokens) if cb == 0 and self.controls is not None else (),
            )
            if cb == 0:
                self.boundary_token_ids = boundary_ids
                if self.config.cb0_boundary_tokens and boundary_ids is None:
                    raise ValueError(
                        f"CB0_BOUNDARY_TOKENS=1 needs a CB0 bundle that declares the boundary rows; "
                        f"{model_path} has no boundary_tokens.json"
                    )
                if boundary_ids is not None:
                    logger.info(
                        "CB0 boundary tokens %s (token windows %s)", boundary_ids,
                        "on" if self.config.cb0_boundary_tokens else "off",
                    )

            self.tokenizers.append(tokenizer)
            logger.debug(
                "CB%d: %s (target=%d, codebooks=%d)",
                cb,
                model_dir_name,
                target_codebook,
                n_codebooks,
            )

        # Load DualCodec (shared across languages).
        logger.info("Loading DualCodec (%s)", DUALCODEC_MODEL_TYPE)
        self.dc_inference = get_dc_inference(
            runtime_paths.dualcodec,
            runtime_paths.w2vbert,
            DUALCODEC_MODEL_TYPE,
        )
        # fp16 weights only help on CUDA; CPU layer_norm rejects fp32 input
        # against fp16 weights, so leave the CPU model in its native fp32.
        if torch.cuda.is_available():
            self.dc_inference.model = self.dc_inference.model.to(torch.float16)
            sm = self.dc_inference.semantic_cfg.get("semantic_model")
            if sm is not None:
                self.dc_inference.semantic_cfg["semantic_model"] = sm.to(torch.float16)
            logger.debug("DualCodec weights converted to float16")
        else:
            logger.debug("DualCodec using float32 on CPU")
        if self.config.backend == "lite":
            self._slim_semantic_model()

        # CB0-CTC recogniser for the CB0 retry guard (CB0_CTC_GUARD, on by default).
        self.cb0_ctc = load_recognizer(self.config, model_paths[0])
        # /predict quality gate (PREDICT_QUALITY_GATE=1, opt-in; needs the recogniser).
        if self.config.predict_quality_gate:
            self.quality_gate = build_quality_gate(self.config, self.cb0_ctc, self.rate_window)

        # Encode the default voice prompt once for the shared stack.
        self.prompt_semantic, self.prompt_acoustic = self._encode_default_prompt(
            self.dc_inference, len(self.config.codebook_model_dirs),
        )

        self.engines = await build_codebook_engines(self.config, model_paths, self.tokenizers)

        # Load VibeVoice acoustic tokenizer for postprocessing (shared).
        if self.config.enable_vibevoice:
            logger.info("Loading VibeVoice acoustic tokenizer")
            if runtime_paths.vibevoice is None:
                raise RuntimeError("VibeVoice was enabled but its model did not resolve")
            self.vv_tokenizer = load_vibevoice_tokenizer(
                runtime_paths.vibevoice, cpu_quant=self.config.lite_vibevoice_quant)
            if self.config.backend == "lite":
                self._quantize_vibevoice()
            self.vv_engine = VVDecodeEngine(
                self.vv_tokenizer,
                max_queue_size=self.config.max_vv_decode_queue_size,
            )
            self.vv_engine.start()
            logger.info("VibeVoice acoustic tokenizer loaded")
            if vv_adapter is not None:
                self.vv_adapter = activate_vv_adapter(
                    vv_adapter, self.vv_tokenizer.device, stream=self.config.stream_vv_adapter,
                )
        elif vv_adapter is not None:
            logger.warning("VV_ADAPTER_CKPT is set but VibeVoice is disabled: the adapter is not loaded")

        # Optional classifier-free guidance on the voice prompt for CB1-3 (CFG_WEIGHT_PER_CB, default off)
        if any(w > 0 for w in self.config.cfg_weight_per_cb[1:]):
            from app.cfg_acoustic import maybe_attach
            maybe_attach(self, model_paths, self.config.cfg_weight_per_cb)

        if self.config.backend == "lite":
            free_memory()  # return loading/quantization temporaries to the driver
        # The loaded stack lives for the whole process: keep its ~0.5M objects out of
        # the cyclic GC so per-request gc.collect() calls only scan new objects.
        gc.freeze()
        logger.info("Tontaube server ready with %d codebooks", len(self.engines))

    def _slim_semantic_model(self) -> None:
        """Drop unused W2V-BERT layers; optionally stream the rest from host memory."""
        from app.codec.w2v import offload_semantic_model, truncate_semantic_model

        dropped = truncate_semantic_model(self.dc_inference)
        offload = self.config.lite_offload_w2v and torch.cuda.is_available()
        if offload:
            offload_semantic_model(self.dc_inference)
        free_memory()
        logger.info(
            "W2V-BERT: dropped %d unused layers%s",
            dropped,
            "; encoder layers stream from host memory during voice encodes" if offload else "",
        )

    def _quantize_vibevoice(self) -> None:
        from app.codec.vibevoice_quant import quantize_vibevoice
        from app.lite.plans import parse_spec
        from app.vibevoice_utils import _warmup_vibevoice

        spec = parse_spec(self.config.lite_vibevoice_quant)
        if spec.bits == 16 or self.vv_tokenizer.device.type != "cuda":
            return  # the quantized FFN kernels are GPU kernels
        n, saved = quantize_vibevoice(self.vv_tokenizer, spec)
        _warmup_vibevoice(self.vv_tokenizer, self.vv_tokenizer.device)
        logger.info("VibeVoice: %d FFN layers as %s (-%.0f MiB)", n, spec, saved / 2**20)

    def _encode_default_prompt(self, dc_inference, gen_n_codebooks: int):
        if not os.path.exists(self.config.default_voice):
            raise FileNotFoundError(
                f"Default voice not found at {self.config.default_voice}; "
                "set DEFAULT_VOICE to a valid reference file"
            )
        return encode_voice_prompt(
            dc_inference,
            self.config.default_voice,
            self.config.prompt_seconds,
            self.config.prompt_max_tokens,
            gen_n_codebooks,
        )


    async def _generate_with_engine(
        self,
        engine,
        prompt_ids: list[int],
        sampling_params: SamplingParams,
        priority: int = 0,
    ) -> list[int]:
        from app.priority_gate import current_priority, get_gate, Priority
        if current_priority.get() == Priority.LOW:
            await get_gate().wait_if_low()
        request_id = str(uuid.uuid4())

        results_generator = engine.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_ids),
            sampling_params=sampling_params,
            request_id=request_id,
            priority=priority,
        )

        final_output = None
        async for output in results_generator:
            final_output = output

        # await engine.do_log_stats()

        if final_output and final_output.outputs:
            out = final_output.outputs[0]
            return list(out.token_ids)
        return []

    async def generate(
        self,
        text: str | list[str],
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        silence_logit_bias: float | None = None,
        seed: int | None = None,
        voice_path: str | None = None,
        voice_audio_b64: str | None = None,
        voice_tokens: list[list[int]] | None = None,
        prefix_text: str | None = None,
        prefix_tokens: list[list[int]] | None = None,
        max_new_tokens: int | None = None,
        vibevoice_postprocess: bool | None = None,
        acoustic_temperature: float | None = None,
        acoustic_top_k: int | None = None,
        prompt_max_tokens: int | None = None,
        language: str = DEFAULT_LANGUAGE,
        tag: str = DEFAULT_TAG,
        use_system_tag: bool = True,
        vllm_priority: int | None = None,
        controls: dict | str | None = None,
        segment_controls: list | None = None,
        return_piece_frames: bool = False,
    ):
        """``controls``: request-level sliders ({name: bin} or a control-token string);
        ``segment_controls``: one entry per text segment, merged over ``controls``
        (see app/controls.py). Both only affect the CB0 header.

        Returns ``(audio, codebook_tokens)``; with ``return_piece_frames`` (/predict's include_timestamps)
        ``(audio, codebook_tokens, piece_frames)``: ``piece_frames[i] = (start, end)``, the CB0 frames of text
        segment i in ``codebook_tokens[0]``. The segments' codes are contiguous and decoded in one pass (no inserted
        silence, no crossfade; a prefix is cut off at a frame boundary), so frame k starts at audio sample
        k * 1920 (24 kHz). Segments after a CB0 early stop have no entry; the trailing silence tokens are in none."""
        start_time = time.time()

        sampling, voice_prompt, system_tag, acoustic_temperature, acoustic_top_k = self._resolve_params(
            temperature=temperature, top_p=top_p, top_k=top_k,
            frequency_penalty=frequency_penalty, seed=seed,
            voice_path=voice_path, voice_audio_b64=voice_audio_b64,
            voice_tokens=voice_tokens, prompt_max_tokens=prompt_max_tokens,
            acoustic_temperature=acoustic_temperature, acoustic_top_k=acoustic_top_k,
            language=language, tag=tag, use_system_tag=use_system_tag,
        )
        effective_vv_tokenizer = None if vibevoice_postprocess is False else self.vv_tokenizer
        segments = [text] if isinstance(text, str) else list(text)
        cb0_system_tag, cb0_segment_tags = self._cb0_system_tags(
            system_tag, len(segments), language, tag, use_system_tag,
            controls=controls, segment_controls=segment_controls,
        )

        prefix, prefix_acoustic_codes = await self._resolve_prefix(
            prefix_text, prefix_tokens, sampling, voice_prompt,
            max_new_tokens, acoustic_temperature, acoustic_top_k,
            system_tag=system_tag, silence_logit_bias=silence_logit_bias,
            semantic_system_tag=cb0_system_tag,
        )
        n_codebooks = self._n_codebooks()

        # PREDICT_QUALITY_GATE: the request's quality gate (None = off).
        gate_request = (
            self.quality_gate.new_request(
                sampling.get('seed'), voice_prompt, len(segments), self.config.cb0_max_retries,
            )
            if self.quality_gate is not None else None
        )
        gate_passes: list = []
        gate_kept_pass = None

        base_temperature = sampling.get('temperature', 0.8)
        for _rms_attempt in range(1, self.config.cb0_max_retries + 1):
            # Resolve the priority passed to vLLM's min-heap scheduler.
            # Server-side default is 100 (normal); callers can pass smaller
            # values to jump the queue or larger values to run as background
            # work (e.g. very long requests yielding to short ones).
            resolved_priority = 100 if vllm_priority is None else int(vllm_priority)

            attempt_sampling = dict(sampling)
            attempt_sampling['temperature'] = (
                base_temperature + (_rms_attempt - 1) * self.config.vv_retry_temperature_step
            )
            if gate_request is not None:
                # quality gate: a fresh base seed per request-level pass (pass 1 keeps the request seed)
                attempt_sampling['seed'] = gate_request.begin_pass(_rms_attempt)

            semantic_codes, marker_indices = await self._generate_semantic(
                segments, sampling=attempt_sampling, voice_prompt=voice_prompt,
                prefix=prefix,
                max_new_tokens=max_new_tokens,
                system_tag=cb0_system_tag,
                segment_system_tags=cb0_segment_tags,
                vllm_priority=resolved_priority,
                silence_logit_bias=silence_logit_bias,
                use_context=True,
                **({'quality_gate': gate_request} if gate_request is not None else {}),
            )

            acoustic_codes = []
            prev_acoustic: list[list[int]] = []
            for cb in range(1, n_codebooks):
                cb_codes = await self._generate_acoustic(
                    segments,
                    semantic_codes=semantic_codes,
                    marker_indices=marker_indices,
                    prev_acoustic=prev_acoustic,
                    codebook=cb,
                    voice_prompt=voice_prompt,
                    use_context=False,
                    prefix=prefix,
                    prefix_acoustic=prefix_acoustic_codes.get(cb),
                    acoustic_temperature=acoustic_temperature,
                    acoustic_top_k=acoustic_top_k,
                    system_tag=system_tag,
                    vllm_priority=resolved_priority,
                )
                acoustic_codes.append(cb_codes)
                prev_acoustic.append(cb_codes)

            # Strip prefix from output
            prefix_codes = None
            if prefix:
                prefix_len = len(prefix['semantic_codes'])
                if self.config.prefix_joint_decode:
                    # PREFIX_JOINT_DECODE: keep the last chunk_context_seconds of the prefix codes as the
                    # decoder's left context. Only for a client splice: every codebook row passed in prefix_tokens and
                    # none shorter than the prefix (no generated or zero-padded prefix audio decoded before the join).
                    if prefix_tokens and len(prefix_tokens) >= n_codebooks and all(
                            len(prefix_tokens[cb]) >= prefix_len for cb in range(1, n_codebooks)):
                        ctx = min(prefix_len, round(self.config.chunk_context_seconds * DUALCODEC_HZ))
                        prefix_codes = (semantic_codes[prefix_len - ctx:prefix_len],
                                        [ac[prefix_len - ctx:prefix_len] for ac in acoustic_codes])
                    else:
                        logger.info("Prefix joint decode skipped: prefix_tokens missing or incomplete")
                semantic_codes = semantic_codes[prefix_len:]
                acoustic_codes = [ac[prefix_len:] for ac in acoustic_codes]
                marker_indices = [m - prefix_len for m in marker_indices[1:]]
            # each generated segment's CB0 frames (before the trailing silence): starts [0] + markers
            piece_bounds = [0] + list(marker_indices) + [len(semantic_codes)]
            piece_frames = list(zip(piece_bounds[:-1], piece_bounds[1:]))

            self._append_trailing_silence(
                semantic_codes, acoustic_codes, self.config.trailing_silence_tokens,
            )

            # Decode + VV postprocess
            dc_hz = DUALCODEC_HZ
            short_audio = len(semantic_codes) / dc_hz < 60
            free_memory()
            if prefix_codes is not None:
                audio_tensor = decode_with_prefix_context(
                    self.dc_inference, effective_vv_tokenizer,
                    prefix_codes[0], prefix_codes[1],
                    semantic_codes, acoustic_codes,
                    hz=dc_hz,
                    context_seconds=self.config.chunk_context_seconds,
                    end_fix=self.config.decode_end_fix,
                    end_fade_ms=self.config.decode_end_fade_ms,
                    vv_adapter=self.vv_adapter,
                    window_seconds=self.config.decode_window_seconds,
                )
            else:
                audio_tensor = decode_chunked_with_context(
                    self.dc_inference, effective_vv_tokenizer,
                    semantic_codes, acoustic_codes,
                    hz=dc_hz,
                    context_seconds=self.config.chunk_context_seconds,
                    end_fix=self.config.decode_end_fix,
                    end_fade_ms=self.config.decode_end_fade_ms,
                    vv_adapter=self.vv_adapter,
                    window_seconds=self.config.decode_window_seconds,
                )

            if short_audio:
                retry_reason = self._check_audio_quality(audio_tensor)
                if retry_reason is None:
                    break
                if gate_request is not None:
                    gate_passes.append(self._gate_pass_record(
                        _rms_attempt, retry_reason, audio_tensor, semantic_codes, acoustic_codes, gate_request,
                    ))
                    gate_passes[-1]['piece_frames'] = piece_frames
                logger.warning(
                    "Generation quality retry %d/%d: %s",
                    _rms_attempt,
                    self.config.cb0_max_retries,
                    retry_reason,
                )
            else:
                break
        else:
            if gate_request is not None and gate_passes:
                # quality gate: no pass passed the request-level check: keep the best pass, not the last one
                best_pass = min(gate_passes, key=lambda r: r['rank'])
                audio_tensor, semantic_codes, acoustic_codes, piece_frames = (
                    best_pass['audio'], best_pass['semantic'], best_pass['acoustic'], best_pass['piece_frames'],
                )
                gate_kept_pass = best_pass['pass']
                logger.warning(
                    "Quality gate: no pass passed the request check; keeping pass %d of %d (%s)",
                    best_pass['pass'], len(gate_passes), best_pass['reason'],
                )
        if gate_request is not None:
            n_passes = len(gate_passes) + (0 if gate_kept_pass is not None else 1)
            gate_request.finish(
                gate_kept_pass if gate_kept_pass is not None else n_passes, n_passes,
                [r['reason'] for r in gate_passes],
            )

        # OUTPUT_HEADROOM_DB: fixed output gain on the final decode (VibeVoice,
        # or DualCodec with VibeVoice off), after DECODE_END_FIX and before main.py's trim / pause cap /
        # MossFormer2 / int16 export. Applied after the silence retry check so retry decisions are unchanged.
        # In place on the fresh decode tensor (no copy); 0 dB leaves it untouched.
        audio_tensor = apply_output_headroom(audio_tensor, self.config.output_headroom_db, inplace=True)

        end_time = time.time()
        processing_time = end_time - start_time

        audio_samples = audio_tensor.shape[-1]
        audio_duration = audio_samples / DUALCODEC_SAMPLE_RATE
        rtf = processing_time / audio_duration if audio_duration > 0 else 0.0

        mode = f"{len(segments)} segments" if len(segments) > 1 else "single"
        if prefix:
            mode += " + prefix"
        logger.info(
            "Generated %.2fs of audio in %.2fs (RTF %.3f, %s)",
            audio_duration,
            processing_time,
            rtf,
            mode,
        )

        codebook_tokens = [semantic_codes] + acoustic_codes
        if return_piece_frames:
            return audio_tensor, codebook_tokens, piece_frames
        return audio_tensor, codebook_tokens

    def _early_retry_sampling(self, sampling: dict, attempt: int) -> dict:
        """STREAM_EARLY_CHECK: a fresh sample for early attempt >= 2 (/predict's retry temperature step and another seed)."""
        retry = dict(sampling)
        retry["temperature"] = sampling["temperature"] + (attempt - 1) * self.config.vv_retry_temperature_step
        if sampling.get("seed") is not None:
            retry["seed"] = sampling["seed"] + attempt - 1
        return retry

    def _early_chunk_rejection(
        self, early_cb0: list[int], early_audio: 'torch.Tensor | None', early_target_vv: int, n_chars: int,
    ) -> str | None:
        """STREAM_EARLY_CHECK: why the early chunk must not be played, or None.

        The early CB0 check with 12794 counted as silence; on the decoded audio of the part that would be played, /predict's
        silent-audio check, the active-speech floor (STREAM_EARLY_MIN_ACTIVE) and the optional pace floor
        (STREAM_EARLY_PACE_MIN_ACTIVE, 0 = off).
        """
        reason = self._check_cb0_generation(
            len(early_cb0), n_chars, False, DUALCODEC_HZ, codes=early_cb0, early_mode=True,
        )
        if reason is not None or early_audio is None:
            return reason
        played = early_audio[..., :early_target_vv * VV_HOP_SAMPLES]
        reason = self._check_audio_quality(played)
        if reason is not None:
            return reason
        active = active_fraction(played)
        if active < self.config.stream_early_min_active:
            return f"near_silent (active {active:.2f} < {self.config.stream_early_min_active:.2f})"
        if active < self.config.stream_early_pace_min_active:
            return f"slow_start (active {active:.2f} < {self.config.stream_early_pace_min_active:.2f})"
        return None

    @staticmethod
    def _check_audio_quality(
        audio_tensor: 'torch.Tensor',
    ) -> str | None:
        """Retain silence/invalid-output retries without comparing raw and VV audio."""
        post_vv = audio_tensor.float().cpu()
        if post_vv.numel() == 0:
            return "empty_audio"
        if not torch.isfinite(post_vv).all().item():
            return "non_finite_audio"
        post_vv_rms = post_vv.pow(2).mean().sqrt().item()
        logger.debug(
            "Decode quality: post_rms=%.6f post_peak=%.6f samples=%d",
            post_vv_rms,
            post_vv.abs().max().item(),
            post_vv.shape[-1],
        )
        if post_vv_rms <= 0.002:
            return f"silent (RMS={post_vv_rms:.6f})"
        return None

    @staticmethod
    def _gate_pass_record(pass_idx, reason, audio_tensor, semantic_codes, acoustic_codes, gate_request) -> dict:
        """Quality gate: one request-level pass that failed ``_check_audio_quality``, with its ranking
        key (best first): finite non-empty audio, fewest kept pieces at the CB0 cap, fewest gate flags on the kept
        pieces, lowest max logistic p, highest RMS, earliest."""
        post = audio_tensor.float().cpu()
        bad = post.numel() == 0 or not torch.isfinite(post).all().item()
        rms = 0.0 if bad else post.pow(2).mean().sqrt().item()
        caps, flags, maxp = gate_request.pass_rank()
        return {
            'pass': pass_idx, 'reason': reason, 'audio': audio_tensor, 'semantic': list(semantic_codes),
            'acoustic': [list(a) for a in acoustic_codes], 'rank': (bad, caps, flags, maxp, -rms, pass_idx),
        }

    def _resolve_voice_prompt(
        self,
        voice_audio_b64: str | None,
        voice_path: str | None,
        prompt_max_tokens: int | None = None,
        n_codebooks: int | None = None,
    ) -> tuple[list | None, list | None]:
        with self._voice_prompt_lock:
            return resolve_voice_prompt(
                voice_audio_b64, voice_path,
                self.dc_inference,
                self.config.prompt_seconds,
                prompt_max_tokens if prompt_max_tokens is not None else self.config.prompt_max_tokens,
                n_codebooks or self.config.generate_n_codebooks,
                self._voice_prompt_cache,
                self._voice_prompt_cache_max,
                (self.prompt_semantic, self.prompt_acoustic),
            )

    def encode_voice(
        self, voice_audio_b64: str, prompt_max_tokens: int | None = None, n_codebooks: int | None = None,
    ) -> list[list[int]]:
        """Encode a reference into `n_codebooks` token rows (row 0 semantic, then acoustic).

        prompt_max_tokens None returns the full prompt (the first prompt_seconds, at most
        MAX_VOICE_PROMPT_TOKENS) instead of the server's default cap; n_codebooks None means the
        codebooks generation uses. Shares the voice-prompt cache with generation (per codebook
        count), so a later request with the same audio skips the encode. Blocking: call it off
        the event loop.
        """
        semantic, acoustic = self._resolve_voice_prompt(
            voice_audio_b64, None, prompt_max_tokens=prompt_max_tokens or MAX_VOICE_PROMPT_TOKENS,
            n_codebooks=n_codebooks,
        )
        return [semantic, *(acoustic or [])]

    def _check_cb0_generation(
        self, n_tokens: int, n_chars: int, hit_max_tokens: bool, hz: float,
        codes: list[int] | None = None, early_mode: bool = False,
        cps_scale: float = 1.0, max_cps: float | None = None,
    ) -> str | None:
        """Return failure reason string, or None if generation is acceptable.

        ``cps_scale`` scales both chars/s bounds (rate-aware window, app/cb0_guard.py); ``max_cps`` replaces
        cb0_max_cps as the upper bound (_cb0_max_cps).
        """
        if hit_max_tokens and not early_mode:
            return "max_tokens_reached"
        # When token cap is expected (early chunk), skip CPS/min_tokens — only check silence
        if not early_mode:
            if n_tokens < self.config.cb0_min_tokens:
                return f"too_few_tokens ({n_tokens})"
            if n_tokens > 0 and n_chars > 30:
                duration = n_tokens / hz
                cps = n_chars / duration
                min_cps = self.config.cb0_min_cps * cps_scale
                max_cps = (max_cps or self.config.cb0_max_cps) * cps_scale
                if cps < min_cps or cps > max_cps:
                    if cps_scale != 1.0:
                        return f"cps_out_of_range (cps={cps:.1f}, window {min_cps:.1f}-{max_cps:.1f})"
                    return f"cps_out_of_range (cps={cps:.1f})"
        if codes and self.config.cb0_max_silence_ratio > 0:
            check = codes[:50]
            silence_set = set(SILENCE_TOKEN_IDS)
            if self.config.cb0_silence_count_digital:
                silence_set.add(DC_SILENCE_TOKEN)  # digital silence
            if early_mode and self.config.stream_early_check:
                silence_set.add(EARLY_LEAD_SILENCE_TOKEN)  # STREAM_EARLY_CHECK: the model's own lead-silence code
            n_silence = sum(1 for c in check if c in silence_set)
            ratio = n_silence / len(check)
            if ratio > self.config.cb0_max_silence_ratio:
                return f"too_much_silence ({n_silence}/{len(check)} = {ratio:.0%})"
        if codes and not early_mode and self.config.cb0_gap_check and self.config.cb0_max_gap_seconds > 0:
            # Whole-piece gap check. The check above sees only the first 50 codes, so it misses a long silence
            # later in the piece. Same silence measure (the early-silence check's code set), over the whole piece;
            # an early streaming chunk is not a whole piece and keeps the checks above only.
            gap_set = set(SILENCE_TOKEN_IDS)
            if self.config.cb0_silence_count_digital:
                gap_set.add(DC_SILENCE_TOKEN)
            gap_s = longest_run(codes, gap_set) / hz
            if gap_s > self.config.cb0_max_gap_seconds:
                return f"long_gap ({gap_s:.2f}s > {self.config.cb0_max_gap_seconds:g}s)"
        return None

    def _cps_window_distance(
        self, n_tokens: int, n_chars: int, hz: float, cps_scale: float = 1.0, max_cps: float | None = None,
    ) -> float:
        """STREAM_RETRY_FIX: how far an attempt's chars/s lies outside the CB0 window (log ratio; 0 inside)."""
        if n_tokens <= 0 or n_chars <= 0:
            return math.inf
        cps = n_chars / (n_tokens / hz)
        min_cps = self.config.cb0_min_cps * cps_scale
        max_cps = (max_cps or self.config.cb0_max_cps) * cps_scale
        if cps < min_cps:
            return math.log(min_cps / cps)
        if cps > max_cps:
            return math.log(cps / max_cps)
        return 0.0

    def _cb0_max_cps(self, system_tag: str) -> float:
        """Chars/s upper bound for the style in a CB0 header (cb0_max_cps_by_style, else cb0_max_cps)."""
        return self.config.cb0_max_cps_by_style.get(parse_cb0_header(system_tag)[1] or "", self.config.cb0_max_cps)

    def _cb0_segment_checks(
        self, system_tag: str, text: str, early_mode: bool,
    ) -> tuple[float, SegmentGuard | None]:
        """(chars/s window scale, CB0-CTC guard or None) for one CB0 segment.

        Both follow the segment's CB0 header: the rate slider bin and style pick the scale; the
        guard runs for the languages the recogniser knows, never on an early streaming chunk
        (it covers only part of the text) or a text without letters.
        """
        language, style, rate_bin = parse_cb0_header(system_tag)
        cps_scale = self.rate_window.factor(style, rate_bin) if self.rate_window is not None else 1.0
        guard = None
        if (
            self.config.cb0_ctc_guard
            and self.cb0_ctc is not None
            and not early_mode
            and language in self.config.cb0_ctc_guard_languages
            and normalize_text(text)
        ):
            digits = any(ch.isdigit() for ch in text)
            max_cer = self.config.cb0_ctc_digit_max_cer if digits else self.config.cb0_ctc_max_cer
            guard = SegmentGuard(self.cb0_ctc, text, max_cer, self.config.cb0_ctc_max_retries)
        return cps_scale, guard

    async def _generate_semantic_segment(
        self,
        seg_idx: int,
        segments: list[str],
        prev_chunk_global: list[int] | None,
        sampling: dict,
        voice_prompt: tuple,
        prefix: dict | None,
        max_new_tokens: int | None,
        forced_prefix: list[int] | None = None,
        priority: int = 0,
        system_tag: str = '',
        allow_marker_iteration: bool = False,
        stop_at_eos: bool = True,
        early_mode: bool = False,
        boundary_total_segments: int | None = None,
        silence_logit_bias: float | None = None,
        use_context: bool | None = None,
        prev_audio: PrevAudioChain | None = None,
        restart_context: bool = False,
        stream_retry_fix: bool = False,
        quality_gate=None,
    ) -> tuple[list[int], bool]:
        """Core semantic generation for one segment.

        Returns (global_semantic_ids, hit_eos).

        allow_marker_iteration: if True, continue generating past audio_markers
            (batch single-segment mode — model may emit markers for long text).
        stop_at_eos: if False, exclude EOS from stop tokens and replace any EOS
            in output with audio_marker (streaming non-final segments).
        silence_logit_bias: optional per-request base bias for the CB0
            collapse-silence group. The configured retry-step adjustment is added.
        prev_audio: the request's previous-audio chain (CB0_DROP_BAD_PREV_AUDIO): a
            retry of a non-speech attempt drops ``prev_chunk_global`` and the previous text
            (the window is rebuilt without context), and the kept attempt's CB0-CTC CER is
            recorded so a failed segment is not passed on as previous audio.
        restart_context: generate like a chunk start at ``seg_idx`` (the chain dropped the
            previous segment): no previous audio, a segment-0-shaped window starting here
            (``_text_window(restart=True)``). The non-speech retry rebuilds the same window.
        quality_gate: the /predict request's ``RequestGate`` (PREDICT_QUALITY_GATE;
            app/quality_gate.py) or None. When it gates this piece it makes the
            retry decision after serving's own check (instead of the CB0-CTC guard), sets each
            attempt's seed, and picks the kept attempt.
        """
        ctx = self._cb_ctx(0)
        n_segs = len(segments)

        if use_context is None:
            use_context = prev_chunk_global is not None or seg_idx == 0
        window_text, win_start, win_end = self._text_window(
            seg_idx, segments, use_context=use_context or restart_context,
            prefix=prefix, system_tag=system_tag,
            boundary_total_segments=boundary_total_segments, restart=restart_context,
            boundary_tokens=self.config.cb0_boundary_tokens,
            drop_space=self.config.cb0_boundary_drop_space,
        )
        text_ids = tokenize_text(window_text, ctx.tokenizer)
        check_cb0_window_ids(window_text, text_ids, self.boundary_token_ids, self.config.cb0_boundary_tokens)

        vp_cap = self.config.prompt_max_tokens_per_cb[0] if self.config.prompt_max_tokens_per_cb else None
        prompt_semantic = voice_prompt[0][:vp_cap] if voice_prompt[0] and vp_cap else voice_prompt[0]

        force_silence = (self.config.force_first_silence and seg_idx == 0 and not forced_prefix
                         and not (prefix and not self.config.prefix_force_first_silence))
        silence_token = ctx.dc_base_id + SILENCE_TOKEN_ID

        last_is_marker = False
        last_is_eos = False
        semantic_only: list[int] = []
        cps_scale, guard = self._cb0_segment_checks(system_tag, segments[seg_idx], early_mode)
        max_cps = self._cb0_max_cps(system_tag)
        digits = sum(ch.isdigit() for ch in segments[seg_idx])
        cps_chars = len(segments[seg_idx]) + (self.config.cb0_cps_digit_weight - 1) * digits
        passed = False
        failed_attempts: list[tuple[list[int], bool, bool, str]] = []
        failure = None  # the last attempt's failure (None: it passed)
        gate = (
            quality_gate.piece(seg_idx, segments[seg_idx], system_tag, early_mode)
            if quality_gate is not None else None
        )
        n_attempts = self.config.cb0_max_retries if gate is None else gate.attempt_cap

        for attempt in range(1, n_attempts + 1):
            # Serving's retry schedule (temperature / silence bias) steps once per attempt; with the
            # quality gate only after serving's own failures (a gate retry keeps the base values).
            retry_steps = attempt - 1 if gate is None else gate.retry_steps
            # Keep the first retry at base temperature; increase only after it.
            # Reset from the requested base for every segment; never mutate the
            # shared sampling dictionary or compound increases across segments.
            attempt_sampling = {
                **sampling,
                "temperature": sampling["temperature"]
                + max(0, retry_steps - 1) * self.config.cb0_retry_temperature_step,
            }
            if gate is not None:
                # the gate's seed schedule; pass-through pieces (gate None) take the branches below
                attempt_sampling["seed"] = gate.seed(attempt)   # fresh seed per retry
            elif stream_retry_fix and attempt > 1 and sampling.get("seed") is not None:
                # STREAM_RETRY_FIX: a fresh sample per retry. With the request's seed and attempt 1's temperature,
                # attempt 2 repeats attempt 1, and later attempts stay correlated. Streaming keeps this schedule
                # (one reseed per attempt, never both).
                attempt_sampling["seed"] = sampling["seed"] + attempt - 1
            elif "seed" in sampling:
                # A fresh seed per retry. With the request seed on every attempt (and the first retry at base
                # temperature) attempt 2 would regenerate attempt 1 exactly. Attempt 1 keeps the request seed.
                attempt_sampling["seed"] = retry_seed(sampling["seed"], attempt)
            raw_accumulated = []
            if force_silence:
                raw_accumulated.append(silence_token)
            if forced_prefix:
                raw_accumulated.extend(ctx.dc_base_id + t for t in forced_prefix)
            token_limit = (
                max_new_tokens
                if max_new_tokens is not None
                else self.config.max_new_tokens
            )
            remaining = max(0, token_limit - len(raw_accumulated))
            hit_eos = False
            iteration = 0

            while remaining > 0:
                iteration += 1

                # Front-prompt scheme: [cb0 prompt][text][PAD][audio context...]
                input_ids = self._front_prompt_block(0, prompt_semantic, None, ctx.dc_base_id)
                input_ids.extend(text_ids)
                input_ids.append(ctx.pad_id)

                if prefix:
                    input_ids.extend(prefix['semantic_global'])
                    input_ids.append(ctx.audio_marker_id)

                if allow_marker_iteration:
                    # Single-segment batch: feed all accumulated tokens back
                    input_ids.extend(raw_accumulated)
                elif prev_chunk_global:
                    input_ids.extend(prev_chunk_global)
                    input_ids.append(ctx.audio_marker_id)

                # For multi-segment first chunk (no prev context yet): add forced prefix / silence to input
                if not allow_marker_iteration and not prev_chunk_global:
                    if force_silence:
                        input_ids.append(silence_token)
                    if forced_prefix:
                        for t in forced_prefix:
                            input_ids.append(ctx.dc_base_id + t)

                n_prev = len(prev_chunk_global) if prev_chunk_global else 0
                logger.debug(
                    "CB0 segment %d/%d iteration %d: window=%d:%d accumulated=%d "
                    "previous_audio=%d input_tokens=%d",
                    seg_idx,
                    n_segs,
                    iteration,
                    win_start,
                    win_end,
                    len(raw_accumulated),
                    n_prev,
                    len(input_ids),
                )

                stop_ids = [ctx.eos_id, ctx.audio_marker_id] if stop_at_eos else [ctx.audio_marker_id]
                base_collapse_silence_bias = (
                    self.config.cb0_silence_logit_bias
                    if silence_logit_bias is None else silence_logit_bias
                )
                collapse_silence_bias = (
                    base_collapse_silence_bias
                    + retry_steps * self.config.cb0_silence_logit_bias_step
                )
                logit_bias = {ctx.eos_id: self.config.cb0_eos_logit_bias}
                for sid in COLLAPSE_SILENCE_TOKEN_IDS:
                    logit_bias[ctx.dc_base_id + sid] = collapse_silence_bias
                for sid in ORDINARY_SILENCE_TOKEN_IDS:
                    logit_bias[ctx.dc_base_id + sid] = self.config.cb0_ordinary_silence_logit_bias
                sampling_params = SamplingParams(
                    **attempt_sampling,
                    max_tokens=remaining,
                    min_tokens=min(
                        remaining,
                        max(0, self.config.cb0_min_tokens - len(raw_accumulated)),
                    ),
                    stop_token_ids=stop_ids,
                    logit_bias=logit_bias,
                )

                output_ids = await self._generate_with_engine(
                    ctx.engine, input_ids, sampling_params, priority=priority,
                )

                if not output_ids:
                    last_is_marker = False
                    last_is_eos = False
                    break

                # For non-final streaming segments, replace any EOS with marker
                if not stop_at_eos:
                    output_ids = [ctx.audio_marker_id if t == ctx.eos_id else t for t in output_ids]

                last_is_marker = output_ids[-1] == ctx.audio_marker_id
                last_is_eos = output_ids[-1] == ctx.eos_id

                n_semantic = sum(1 for t in output_ids
                                if ctx.dc_base_id <= t < ctx.dc_base_id + SEMANTIC_VOCAB_SIZE)
                logger.debug(
                    "CB0 segment %d/%d iteration %d: output=%d semantic=%d "
                    "marker=%s eos=%s",
                    seg_idx,
                    n_segs,
                    iteration,
                    len(output_ids),
                    n_semantic,
                    last_is_marker,
                    last_is_eos,
                )

                # Marker iteration: continue generating past audio_markers
                if allow_marker_iteration and last_is_marker:
                    raw_accumulated.extend(output_ids)
                    remaining -= len(output_ids)
                    continue

                if last_is_marker or last_is_eos:
                    raw_accumulated.extend(output_ids[:-1])
                    if last_is_eos:
                        hit_eos = True
                else:
                    raw_accumulated.extend(output_ids)
                break

            semantic_only = [t for t in raw_accumulated
                             if ctx.dc_base_id <= t < ctx.dc_base_id + SEMANTIC_VOCAB_SIZE]
            semantic_codes_local = [t - ctx.dc_base_id for t in semantic_only]

            hit_max = not hit_eos and not last_is_marker
            failure = self._check_cb0_generation(
                len(semantic_only), cps_chars, hit_max, ctx.hz,
                codes=semantic_codes_local, early_mode=early_mode, cps_scale=cps_scale, max_cps=max_cps,
            )
            if gate is not None:
                # the quality gate decides (it replaces the guard's retry decision)
                failure = await gate.check(
                    failure, semantic_codes_local, (semantic_only, last_is_eos, last_is_marker),
                    seed=attempt_sampling.get("seed"),
                )
            elif failure is None and guard is not None:
                failure = await guard.check(
                    semantic_codes_local, (semantic_only, last_is_eos, last_is_marker),
                )
            if failure is None:
                passed = True
                break
            failed_attempts.append((semantic_only, last_is_eos, last_is_marker, failure))
            logger.warning(
                "CB0 segment %d retry %d/%d: %s (temperature=%.2f)",
                seg_idx,
                attempt,
                n_attempts,
                failure,
                attempt_sampling["temperature"],
            )
            if (
                prev_audio is not None and guard is not None and prev_chunk_global
                and (await guard.score(semantic_codes_local))[1]
            ):
                # after non-speech output the same previous audio mostly runs away again:
                # retry like a chunk start, without the previous segment's text as well.
                prev_chunk_global = None
                window_text, win_start, win_end = self._text_window(
                    seg_idx, segments, use_context=True,
                    prefix=prefix, system_tag=system_tag,
                    boundary_total_segments=boundary_total_segments, restart=True,
                    boundary_tokens=self.config.cb0_boundary_tokens,
                    drop_space=self.config.cb0_boundary_drop_space,
                )
                text_ids = tokenize_text(window_text, ctx.tokenizer)
                check_cb0_window_ids(
                    window_text, text_ids, self.boundary_token_ids, self.config.cb0_boundary_tokens)
                logger.warning(
                    "CB0 segment %d: non-speech attempt (empty CB0-CTC transcript); "
                    "retrying without previous audio and text",
                    seg_idx,
                )

        best = guard.best() if guard is not None else None
        if best is not None and best[0] is not semantic_only:
            # Guard retries spent or attempts exhausted: keep the lowest-CER serving-ok attempt.
            semantic_only, last_is_eos, last_is_marker = best
            logger.warning(
                "CB0 segment %d: keeping the serving-ok attempt with the lowest CB0-CTC CER",
                seg_idx,
            )
        # Never ship a rejected piece silently. The fallback is unchanged (the lowest-CER serving-ok attempt,
        # else the last attempt), but a kept attempt that failed a check is labelled.
        shipped_failure = failure
        if gate is None and best is None and stream_retry_fix and not passed and not early_mode and failed_attempts:
            # STREAM_RETRY_FIX: every attempt failed and there is no CB0-CTC best: keep the attempt whose chars/s is
            # closest to the window, not the last (hottest) one.
            closest = min(failed_attempts, key=lambda a: self._cps_window_distance(
                len(a[0]), cps_chars, ctx.hz, cps_scale, max_cps))
            if closest[0] is not semantic_only:
                semantic_only, last_is_eos, last_is_marker, shipped_failure = closest
                semantic_codes_local = [t - ctx.dc_base_id for t in semantic_only]
                logger.warning(
                    "CB0 segment %d: all attempts failed; keeping the one closest to the chars/s window", seg_idx,
                )
        if gate is not None:
            # the accepted attempt, else the best one by the gate's ranking (logged by the gate)
            semantic_only, last_is_eos, last_is_marker = await gate.finish(
                (semantic_only, last_is_eos, last_is_marker),
            )
            # the gate's check() returns None on every stop, so `failure` cannot label its pieces; a piece
            # kept without the gate's acceptance (best-ranked, best of the serving failures, last) ships labelled.
            summary = gate.request.pieces[-1]
            shipped_failure = None if summary["kept_how"] == "accepted" else (
                f"quality_gate {summary['kept_how']} (stop: {summary['stop']}; "
                f"fired: {', '.join(summary['kept_fired']) or 'none'})"
            )
        elif best is not None:
            best_cer = guard.best_cer()
            shipped_failure = (
                f"ctc_cer ({best_cer:.2f} > {guard.max_cer:g})" if best_cer > guard.max_cer else None
            )
        if shipped_failure is not None:
            logger.warning(
                "CB0 segment %d: FAIL-SHIPPED after %d attempts: %s",
                seg_idx,
                attempt,
                shipped_failure,
            )
        if prev_audio is not None:
            kept_cer = None
            if guard is not None and gate is not None:
                # the guard's CER of the attempt the gate kept (PrevAudioChain semantics unchanged)
                kept_cer, _ = await guard.score([t - ctx.dc_base_id for t in semantic_only])
            elif guard is not None:
                kept_cer = guard.best_cer()
                if kept_cer is None:  # no attempt passed serving's check: score the kept (last) one
                    kept_cer, _ = await guard.score(semantic_codes_local)
            prev_audio.add(semantic_only, kept_cer)

        n_sem = len(semantic_only)
        logger.debug(
            "CB0 segment %d/%d: %d tokens (%.2fs), marker=%s, eos=%s",
            seg_idx,
            n_segs,
            n_sem,
            n_sem / ctx.hz,
            last_is_marker,
            last_is_eos,
        )
        return semantic_only, last_is_eos

    async def _generate_semantic(
        self,
        segments: list[str],
        sampling: dict,
        voice_prompt: tuple[list | None, list | None],
        prefix: dict | None = None,
        max_new_tokens: int | None = None,
        first_seg_forced_prefix: list[int] | None = None,
        system_tag: str = '',
        vllm_priority: int = 100,
        silence_logit_bias: float | None = None,
        use_context: bool = True,
        segment_system_tags: list[str] | None = None,
        quality_gate=None,
    ) -> tuple[list[int], list[int]]:
        """Generate semantic tokens for one or more segments.

        ``segment_system_tags`` (optional, one per segment) overrides ``system_tag``
        for that segment's CB0 calls (per-segment slider headers).
        ``quality_gate``: the /predict request's ``app.quality_gate.RequestGate``, or None (off).

        Returns (local_semantic_codes, marker_indices).
        """
        ctx = self._cb_ctx(0)
        n_segs = len(segments)
        single = n_segs == 1

        audio_chunks: list[list[int]] = []
        prev_global: list[int] | None = None
        finished = False
        # CB0_DROP_BAD_PREV_AUDIO: after a segment that failed the guard, the next
        # one starts without previous audio and text.
        chain = (
            PrevAudioChain(self.config.cb0_ctc_chain_max_cer)
            if self.config.cb0_drop_bad_prev_audio else None
        )

        for seg_idx in range(n_segs):
            if finished:
                break
            forced = first_seg_forced_prefix if seg_idx == 0 else None
            seg_global, hit_eos = await self._generate_semantic_segment(
                seg_idx, segments,
                prev_chunk_global=prev_global if use_context else None,
                sampling=sampling, voice_prompt=voice_prompt,
                prefix=prefix, max_new_tokens=max_new_tokens,
                forced_prefix=forced,
                system_tag=(
                    segment_system_tags[seg_idx] if segment_system_tags else system_tag
                ),
                allow_marker_iteration=single,
                priority=vllm_priority,
                silence_logit_bias=silence_logit_bias,
                use_context=use_context,
                prev_audio=chain,
                restart_context=(
                    use_context and chain is not None and chain.drops_context(seg_idx)
                ),
                **({'quality_gate': quality_gate} if quality_gate is not None else {}),
            )
            audio_chunks.append(seg_global)
            prev_global = chain.previous if chain is not None else seg_global

            if not single:
                if hit_eos and seg_idx < n_segs - 1:
                    logger.warning(
                        "Ignoring premature EOS in CB0 segment %d/%d",
                        seg_idx,
                        n_segs,
                    )
                elif hit_eos or (not seg_global):
                    finished = True

        # Flatten and convert to local codes, recording marker positions
        codes: list[int] = []
        marker_indices: list[int] = []
        cumlen = 0

        if prefix:
            codes.extend(prefix['semantic_codes'])
            cumlen = len(prefix['semantic_codes'])
            marker_indices.append(cumlen)

        for chunk_idx, chunk in enumerate(audio_chunks):
            if chunk_idx > 0:
                marker_indices.append(cumlen)
            for tid in chunk:
                codes.append(tid - ctx.dc_base_id)
            cumlen += len(chunk)

        prefix_info = f", prefix={len(prefix['semantic_codes'])} tokens" if prefix else ""
        logger.debug(
            "CB0 result: %d segments, %d chunks, %d tokens (%.2fs)%s",
            n_segs,
            len(audio_chunks),
            len(codes),
            len(codes) / ctx.hz,
            prefix_info,
        )
        return codes, marker_indices

    async def _generate_acoustic(
        self,
        segments: list[str],
        semantic_codes: list[int],
        marker_indices: list[int],
        prev_acoustic: list[list[int]],
        codebook: int,
        voice_prompt: tuple[list | None, list | None],
        use_context: bool = True,
        prefix: dict | None = None,
        prefix_acoustic: list[int] | None = None,
        acoustic_temperature: float | None = None,
        acoustic_top_k: int = -1,
        system_tag: str = '',
        vllm_priority: int = 100,
    ) -> list[int]:
        """Generate acoustic tokens for a codebook across all segments.

        Thin wrapper: splits codes at markers and loops over _generate_acoustic_segment.
        """
        cb0_segments = split_at_markers(semantic_codes, marker_indices)
        prev_cb_segments = [split_at_markers(p, marker_indices) for p in prev_acoustic]

        seg_offset = 1 if prefix else 0
        n_real_segs = len(cb0_segments) - seg_offset

        # Prepare prefix codes for target codebook (pad/truncate to match semantic prefix length)
        accumulated_prefix: list[int] = []
        if prefix and prefix_acoustic is not None:
            n_prefix_tokens = len(cb0_segments[0])
            accumulated_prefix = list(prefix_acoustic[:n_prefix_tokens])
            while len(accumulated_prefix) < n_prefix_tokens:
                accumulated_prefix.append(0)
            logger.debug("CB%d acoustic prefix: %d tokens", codebook, n_prefix_tokens)

        if use_context:
            accumulated = []
            for real_idx in range(n_real_segs):
                prev_same = accumulated[-1] if accumulated else None
                codes = await self._generate_acoustic_segment(
                    real_idx, n_real_segs, segments, codebook,
                    cb0_segments, prev_cb_segments, prev_same,
                    voice_prompt, prefix=prefix,
                    accumulated_prefix=accumulated_prefix, seg_offset=seg_offset,
                    acoustic_temperature=acoustic_temperature,
                    acoustic_top_k=acoustic_top_k, system_tag=system_tag,
                    priority=vllm_priority,
                )
                accumulated.append(codes)
        else:
            accumulated = list(await asyncio.gather(*[
                self._generate_acoustic_segment(
                    real_idx, n_real_segs, segments, codebook,
                    cb0_segments, prev_cb_segments, None,
                    voice_prompt, prefix=prefix,
                    accumulated_prefix=accumulated_prefix, seg_offset=seg_offset,
                    acoustic_temperature=acoustic_temperature,
                    acoustic_top_k=acoustic_top_k, system_tag=system_tag,
                    priority=vllm_priority,
                )
                for real_idx in range(n_real_segs)
            ]))

        result = list(accumulated_prefix)
        for seg in accumulated:
            result.extend(seg)
        logger.debug(
            "CB%d result: %d segments, %d tokens%s",
            codebook,
            n_real_segs,
            len(result),
            f", prefix={len(accumulated_prefix)}" if prefix else "",
        )
        return result

    @staticmethod
    def _front_prompt_block(
        target_cb: int,
        prompt_semantic: list | None,
        prompt_acoustic: list[list[int]] | None,
        dc_base_id: int,
    ) -> list[int]:
        """Front voice-prompt block: [cb0 prompt][cb1 prompt]...[cbK prompt].

        Every codebook's block reuses positions 1..P (the streams share one
        timeline), so all blocks must have the same length. Sits BEFORE the text.
        """
        ids = [dc_base_id + c for c in (prompt_semantic or [])]
        for j in range(1, target_cb + 1):
            codes = prompt_acoustic[j - 1] if prompt_acoustic and j - 1 < len(prompt_acoustic) else None
            offset = get_codebook_offset(j)
            ids.extend(dc_base_id + offset + c for c in (codes or []))
        return ids

    @staticmethod
    def _audio_row(
        cb_idx: int,
        code_segments: list[list[int]],
        dc_base_id: int,
        pad_id: int,
        prefix_segment: list[int] | None,
        win_start: int,
        win_end: int,
        seg_offset: int,
        marker_id: int | None = None,
        eos_id: int | None = None,
    ) -> list[int]:
        """One codebook audio row: PAD + [prefix (+marker)] + windowed segments [+ EOS].

        Prompts live in the front block, NOT in the rows. Only the cb0 row uses
        markers between segments and a trailing EOS; cb1+ rows are plain audio.
        """
        offset = get_codebook_offset(cb_idx) if cb_idx > 0 else 0
        ids = [pad_id]

        if prefix_segment is not None:
            for code in prefix_segment:
                ids.append(dc_base_id + offset + code)
            if marker_id is not None:
                ids.append(marker_id)

        for wi in range(win_start, win_end):
            ci = wi + seg_offset
            if ci >= len(code_segments):
                continue
            if wi > win_start and marker_id is not None:
                ids.append(marker_id)
            for code in code_segments[ci]:
                ids.append(dc_base_id + offset + code)

        if eos_id is not None:
            ids.append(eos_id)

        return ids

    async def _generate_acoustic_segment(
        self,
        real_idx: int,
        n_real_segs: int,
        segments: list[str],
        codebook: int,
        cb0_segments: list[list[int]],
        prev_cb_code_segments: list[list[list[int]]],
        prev_same_cb_segment: list[int] | None,
        voice_prompt: tuple,
        prefix: dict | None = None,
        accumulated_prefix: list[int] | None = None,
        seg_offset: int = 0,
        forced_prefix_codes: list[int] | None = None,
        priority: int = 0,
        acoustic_temperature: float | None = None,
        acoustic_top_k: int = -1,
        system_tag: str = '',
        boundary_total_segments: int | None = None,
    ) -> list[int]:
        """Core acoustic generation for a single segment of one codebook."""
        accumulated_prefix = accumulated_prefix or []
        ctx = self._cb_ctx(codebook)
        _, prompt_acoustic = voice_prompt
        cb_offset = get_codebook_offset(codebook)
        start_id = ctx.dc_base_id + cb_offset

        # Apply per-engine VP cap
        vp_cap = (
            self.config.prompt_max_tokens_per_cb[codebook]
            if codebook < len(self.config.prompt_max_tokens_per_cb)
            else None
        )
        prompt_semantic = voice_prompt[0][:vp_cap] if voice_prompt[0] and vp_cap else voice_prompt[0]
        if vp_cap and prompt_acoustic:
            prompt_acoustic = [ac[:vp_cap] for ac in prompt_acoustic]

        code_idx = real_idx + seg_offset
        n_tokens = len(cb0_segments[code_idx])
        if n_tokens == 0:
            return []

        window_text, win_start, win_end = self._text_window(
            real_idx, segments[:n_real_segs], use_context=False,
            prefix=prefix, system_tag=system_tag,
            boundary_total_segments=boundary_total_segments,
        )

        # Front-prompt scheme: [cb0..cbK prompts][text][cb0 row][cb1..K-1 rows][target row]
        input_ids = self._front_prompt_block(
            codebook, prompt_semantic, prompt_acoustic, ctx.dc_base_id)
        input_ids.extend(tokenize_text(window_text, ctx.tokenizer))

        input_ids.extend(self._audio_row(
            0, cb0_segments, ctx.dc_base_id, ctx.pad_id,
            prefix_segment=cb0_segments[0] if prefix else None,
            win_start=win_start, win_end=win_end, seg_offset=seg_offset,
            marker_id=ctx.audio_marker_id, eos_id=ctx.eos_id,
        ))

        for prev_cb_idx, segs in enumerate(prev_cb_code_segments):
            input_ids.extend(self._audio_row(
                prev_cb_idx + 1, segs, ctx.dc_base_id, ctx.pad_id,
                prefix_segment=segs[0] if prefix and len(segs) > 0 else None,
                win_start=win_start, win_end=win_end, seg_offset=seg_offset,
            ))

        # Target codebook row: PAD, then any already-known own codes.
        input_ids.append(ctx.pad_id)

        if prefix and accumulated_prefix:
            for code in accumulated_prefix:
                input_ids.append(ctx.dc_base_id + cb_offset + code)

        if prev_same_cb_segment is not None:
            for code in prev_same_cb_segment:
                input_ids.append(ctx.dc_base_id + cb_offset + code)

        n_prefix = len(forced_prefix_codes) if forced_prefix_codes else 0
        if forced_prefix_codes:
            for code in forced_prefix_codes:
                input_ids.append(ctx.dc_base_id + cb_offset + code)

        logger.debug(
            "CB%d segment %d/%d: window=%d:%d input_tokens=%d target=%d%s",
            codebook,
            real_idx,
            n_real_segs,
            win_start,
            win_end,
            len(input_ids),
            n_tokens,
            f", forced_prefix={n_prefix}" if n_prefix else "",
        )

        remaining_tokens = n_tokens - n_prefix
        if remaining_tokens <= 0:
            return list(forced_prefix_codes)

        acoustic_temperature = (
            acoustic_temperature
            if acoustic_temperature is not None
            else self.config.acoustic_temperature
        )
        sampling_params = SamplingParams(
            temperature=acoustic_temperature,
            top_p=self.config.acoustic_top_p,
            top_k=acoustic_top_k,
            max_tokens=remaining_tokens,
        )
        output_ids = await self._generate_with_engine(
            ctx.engine, input_ids, sampling_params, priority=priority,
        )
        new_codes = _extract_acoustic_codes(output_ids, remaining_tokens, start_id, ctx.pad_id)
        return list(forced_prefix_codes) + new_codes if forced_prefix_codes else new_codes

    def _stream_vv_adapter(self, n_codebooks: int):
        """STREAM_VV_ADAPTER: the VibeVoice adapter for this stream's VibeVoice frames (the one /predict uses, loaded
        from VV_ADAPTER_CKPT), or None (flag off, no adapter, or codes the adapter can't read)."""
        if self.vv_adapter is None or not self.config.stream_vv_adapter:
            return None
        if n_codebooks - 1 != VV_ADAPTER_N_ACOUSTIC:
            logger.warning("VibeVoice adapter skipped for this stream: needs %d-level codes (got %d)",
                           VV_ADAPTER_N_ACOUSTIC + 1, n_codebooks)
            return None
        return self.vv_adapter

    async def generate_streaming(
        self,
        text: str | list[str],
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        silence_logit_bias: float | None = None,
        seed: int | None = None,
        voice_path: str | None = None,
        voice_audio_b64: str | None = None,
        voice_tokens: list[list[int]] | None = None,
        prefix_text: str | None = None,
        prefix_tokens: list[list[int]] | None = None,
        max_new_tokens: int | None = None,
        streaming_initial_seconds: float = DEFAULT_STREAMING_INITIAL_SECONDS,
        acoustic_temperature: float | None = None,
        acoustic_top_k: int | None = None,
        prompt_max_tokens: int | None = None,
        text_preprocessor: 'Callable[[str], Awaitable[str]] | None' = None,
        language: str = DEFAULT_LANGUAGE,
        tag: str = DEFAULT_TAG,
        use_system_tag: bool = True,
        controls: dict | str | None = None,
        segment_controls: list | None = None,
        timings: list | None = None,
    ):
        """Wrapper that guarantees VV slot cleanup even on cancellation/error.

        ``timings``: a list that receives one ``(event, seconds since start, audio seconds generated so far)`` tuple per
        step: ``"early"`` when the early chunk is ready, ``"segment"`` after each text segment, ``"end"`` after the
        flush (app/stream_calibration.py). None = not recorded."""
        if self.vv_tokenizer is None:
            raise RuntimeError(
                "generate_streaming requires the VibeVoice acoustic tokenizer"
            )
        vv_slot = self.vv_engine.alloc_slot() if self.vv_engine else None
        trim_lead = self.config.stream_lead_trim
        # PREFIX_START_FADE: a response to a request with a prefix (non-empty prefix_text, as in
        # _resolve_prefix) fades in over its first prefix_start_fade_ms, carried across chunks by the sample offset.
        start_fade_ms = self.config.prefix_start_fade_ms if self.config.prefix_start_fade and prefix_text else None
        emitted = 0
        try:
            async for chunk in self._generate_streaming_inner(
                text=text, temperature=temperature, top_p=top_p, top_k=top_k,
                frequency_penalty=frequency_penalty,
                silence_logit_bias=silence_logit_bias, seed=seed,
                voice_path=voice_path, voice_audio_b64=voice_audio_b64,
                voice_tokens=voice_tokens,
                prefix_text=prefix_text, prefix_tokens=prefix_tokens,
                max_new_tokens=max_new_tokens, vv_slot=vv_slot,
                streaming_initial_seconds=streaming_initial_seconds,
                acoustic_temperature=acoustic_temperature, acoustic_top_k=acoustic_top_k,
                prompt_max_tokens=prompt_max_tokens,
                text_preprocessor=text_preprocessor,
                language=language, tag=tag, use_system_tag=use_system_tag,
                controls=controls, segment_controls=segment_controls, timings=timings,
            ):
                if trim_lead and chunk.shape[-1] > 0:
                    # STREAM_LEAD_TRIM: /predict's leading edge trim on the first non-empty chunk only, before
                    # the headroom gain (the threshold is at the decoder's level, as /predict's matched one).
                    trim_lead = False
                    lead_ms = self.config.stream_lead_trim_padding_ms
                    if prefix_text and self.config.prefix_lead_trim_padding_ms is not None:
                        lead_ms = self.config.prefix_lead_trim_padding_ms   # PREFIX_LEAD_TRIM_PADDING_MS (prefixed requests only)
                    trimmed = trim_stream_lead(chunk, padding_ms=lead_ms)
                    logger.debug("Stream lead trim: cut %d samples", chunk.shape[-1] - trimmed.shape[-1])
                    chunk = trimmed
                # OUTPUT_HEADROOM_DB: the same fixed gain per streamed chunk
                # (early chunk, segment chunks and the DECODE_END_FIX flush), before the stream writers'
                # int16 encode. A per-sample gain, so streaming-safe; 0 dB yields the chunk itself.
                chunk = apply_output_headroom(chunk, self.config.output_headroom_db)
                if start_fade_ms is not None:
                    chunk = start_fade(chunk, start_fade_ms, DUALCODEC_SAMPLE_RATE, offset=emitted)
                    emitted += chunk.shape[-1]
                yield chunk
        finally:
            if vv_slot is not None:
                self.vv_engine.free_slot(vv_slot)
                logger.debug("Released VibeVoice streaming slot %s", vv_slot)
            if not torch.cuda.is_available():
                free_memory()  # return the stream's freed host buffers to the OS

    async def _generate_streaming_inner(
        self,
        text: str | list[str],
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        silence_logit_bias: float | None = None,
        seed: int | None = None,
        voice_path: str | None = None,
        voice_audio_b64: str | None = None,
        voice_tokens: list[list[int]] | None = None,
        prefix_text: str | None = None,
        prefix_tokens: list[list[int]] | None = None,
        max_new_tokens: int | None = None,
        streaming_initial_seconds: float = DEFAULT_STREAMING_INITIAL_SECONDS,
        vv_slot: int | None = None,
        acoustic_temperature: float | None = None,
        acoustic_top_k: int | None = None,
        prompt_max_tokens: int | None = None,
        text_preprocessor: 'Callable[[str], Awaitable[str]] | None' = None,
        language: str = DEFAULT_LANGUAGE,
        tag: str = DEFAULT_TAG,
        use_system_tag: bool = True,
        controls: dict | str | None = None,
        segment_controls: list | None = None,
        timings: list | None = None,
    ):
        """Async generator yielding audio Tensors as they become available.

        Interleaves cb0->cb1->cb2->VV per segment.
        Yields torch.Tensor of shape [1, 1, T] (float32, CPU).
        """
        start_time = time.time()

        sampling, voice_prompt, system_tag, acoustic_temperature, acoustic_top_k = self._resolve_params(
            temperature=temperature, top_p=top_p, top_k=top_k,
            frequency_penalty=frequency_penalty, seed=seed,
            voice_path=voice_path, voice_audio_b64=voice_audio_b64,
            voice_tokens=voice_tokens, prompt_max_tokens=prompt_max_tokens,
            acoustic_temperature=acoustic_temperature, acoustic_top_k=acoustic_top_k,
            language=language, tag=tag, use_system_tag=use_system_tag,
        )
        segments = [text] if isinstance(text, str) else list(text)
        if text_preprocessor and segments:
            segments[0] = await text_preprocessor(segments[0])
        cb0_system_tag, cb0_segment_tags = self._cb0_system_tags(
            system_tag, len(segments), language, tag, use_system_tag,
            controls=controls, segment_controls=segment_controls,
        )

        prefix, prefix_acoustic_codes = await self._resolve_prefix(
            prefix_text, prefix_tokens, sampling, voice_prompt,
            max_new_tokens, acoustic_temperature, acoustic_top_k,
            system_tag=system_tag, silence_logit_bias=silence_logit_bias,
            semantic_system_tag=cb0_system_tag,
        )
        # --- Early cb0 via _generate_semantic_segment ---
        effective_max_new_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else self.config.max_new_tokens
        )
        dc_hz = DUALCODEC_HZ
        pad = self.config.streaming_dc_pad_tokens
        requested_early_dc = int(round(streaming_initial_seconds * dc_hz))
        early_dc_tokens = requested_early_dc + pad
        if early_dc_tokens > effective_max_new_tokens:
            raise ValueError(
                f"max_new_tokens must be at least {early_dc_tokens} when "
                f"streaming_initial_seconds={streaming_initial_seconds:g}"
            )
        # --- VV state and config ---
        chunk_s = self.config.streaming_chunk_seconds
        n_codebooks = self._n_codebooks()
        total_vv_emitted = 0
        stream_adapter = self._stream_vv_adapter(n_codebooks)  # STREAM_VV_ADAPTER; None = the production frames

        async def early_cb0_attempt(attempt_sampling):
            """The early chunk's CB0 codes (early mode)."""
            early_global, _ = await self._generate_semantic_segment(
                seg_idx=0, segments=[segments[0]],
                prev_chunk_global=None,
                sampling=attempt_sampling, voice_prompt=voice_prompt,
                prefix=prefix, max_new_tokens=early_dc_tokens,
                priority=0, system_tag=cb0_segment_tags[0],
                early_mode=True,
                boundary_total_segments=len(segments),
                silence_logit_bias=silence_logit_bias,
                stream_retry_fix=self.config.stream_retry_fix,
            )
            ctx0 = self._cb_ctx(0)
            cb0 = [t - ctx0.dc_base_id for t in early_global
                   if ctx0.dc_base_id <= t < ctx0.dc_base_id + SEMANTIC_VOCAB_SIZE]
            if self.config.force_first_silence and not cb0:
                cb0 = [SILENCE_TOKEN_ID]
            logger.debug("Generated %d early CB0 streaming tokens", len(cb0))
            return cb0

        async def early_acoustic(cb0):
            """CB1-3 and the DualCodec decode of the early chunk when it fills a quantum: (early_ac_per_cb, early_audio or None,
            early_target_vv)."""
            ac_per_cb: list[list[int]] = []
            audio, target_vv = None, 0
            effective_early_dc = len(cb0) - pad
            if cb0 and effective_early_dc > 0:
                dc_tokens_per_quantum = int(round(STREAMING_SECONDS_QUANTUM * dc_hz))
                vv_frames_per_quantum = int(round(STREAMING_SECONDS_QUANTUM * VV_HZ))
                available_quanta = effective_early_dc // dc_tokens_per_quantum
                requested_quanta = round(
                    streaming_initial_seconds / STREAMING_SECONDS_QUANTUM
                )
                target_vv = (
                    min(available_quanta, requested_quanta)
                    * vv_frames_per_quantum
                )
                if target_vv > 0:
                    # Build a temporary streaming state for the early chunk
                    early_state = self._build_streaming_state(prefix, prefix_acoustic_codes)
                    early_state.cb0_segments.append(cb0)
                    early_state.accumulated_sem.extend(cb0)

                    await self._generate_all_acoustic_for_segment(
                        early_state, seg_idx=0, n_real_segs=1,
                        segments=segments, voice_prompt=voice_prompt,
                        prefix=prefix,
                        acoustic_temperature=acoustic_temperature,
                        acoustic_top_k=acoustic_top_k,
                        system_tag=system_tag,
                    )
                    # Extract early acoustic codes for later use as forced prefixes
                    for cb_idx in range(n_codebooks - 1):
                        n_prefix_slots = 1 if prefix else 0
                        ac_per_cb.append(early_state.ac_segs[cb_idx][n_prefix_slots])
                    logger.debug(
                        "Early stream: %d DualCodec tokens -> %d VibeVoice frames",
                        len(cb0),
                        target_vv,
                    )
                    with torch.no_grad():
                        audio = codes_to_audio_tensor(self.dc_inference, cb0, ac_per_cb)
            return ac_per_cb, audio, target_vv

        early_cb0: list[int] = []
        early_ac_per_cb: list[list[int]] = []
        early_audio, early_target_vv = None, 0
        if segments:
            # STREAM_EARLY_CHECK: check the early chunk before it is played; re-sample a failed one from scratch, and if every
            # attempt fails, start at segment 0 without an early chunk (never force a failed early prefix into segment 0).
            n_early = max(1, self.config.stream_early_check_attempts) if self.config.stream_early_check else 1
            for early_try in range(1, n_early + 1):
                attempt_sampling = sampling if early_try == 1 else self._early_retry_sampling(sampling, early_try)
                early_cb0 = await early_cb0_attempt(attempt_sampling)
                reason = None
                if self.config.stream_early_check:     # the codes first: a failed start skips CB1-3 and the decode
                    reason = self._early_chunk_rejection(early_cb0, None, 0, len(segments[0]))
                if reason is None:
                    early_ac_per_cb, early_audio, early_target_vv = await early_acoustic(early_cb0)
                    if not self.config.stream_early_check:
                        break
                    reason = self._early_chunk_rejection(early_cb0, early_audio, early_target_vv, len(segments[0]))
                    if reason is None:
                        break
                logger.warning("Stream early chunk retry %d/%d: %s", early_try, n_early, reason)
            else:
                logger.warning(
                    "Stream early chunk: all %d attempts failed; streaming starts at segment 0 without an early chunk",
                    n_early,
                )
                early_cb0, early_ac_per_cb, early_audio, early_target_vv = [], [], None, 0

        # --- Early VibeVoice decode + yield first audio ---
        if early_audio is not None:
            early_frames = vv_encode(self.vv_tokenizer, early_audio)
            if stream_adapter is not None:
                early_frames = correct_stream_frames(
                    stream_adapter, self.dc_inference, early_cb0, early_ac_per_cb, early_audio, early_frames,
                    0, early_target_vv,
                )
            audio_chunk = await self.vv_engine.decode(
                early_frames[:, :early_target_vv], vv_slot,
            )
            total_vv_emitted = early_target_vv
            if timings is not None:
                timings.append(("early", time.time() - start_time, early_target_vv / VV_HZ))
            yield audio_chunk

        # --- Main streaming loop: per-segment cb0->acoustic->VV ---
        state = self._build_streaming_state(prefix, prefix_acoustic_codes)
        n_real_segs = len(segments)

        logger.debug(
            "Starting stream: %d segments, DualCodec %.1f Hz, VibeVoice %.1f Hz, padding=%d",
            n_real_segs,
            dc_hz,
            VV_HZ,
            self.config.streaming_dc_pad_tokens,
        )

        ctx0 = self._cb_ctx(0)
        prev_chunk_global: list[int] | None = None
        chain = (
            PrevAudioChain(self.config.cb0_ctc_chain_max_cer)
            if self.config.cb0_drop_bad_prev_audio else None
        )
        for seg_idx in range(n_real_segs):
            if text_preprocessor and seg_idx > 0:
                segments[seg_idx] = await text_preprocessor(segments[seg_idx])

            is_final = seg_idx >= n_real_segs - 1
            seg_cb0_global, hit_eos = await self._generate_semantic_segment(
                seg_idx=seg_idx, segments=segments,
                prev_chunk_global=prev_chunk_global,
                sampling=sampling, voice_prompt=voice_prompt,
                prefix=prefix, max_new_tokens=max_new_tokens,
                forced_prefix=early_cb0 if seg_idx == 0 and early_cb0 else None,
                priority=10, system_tag=cb0_segment_tags[seg_idx],
                stop_at_eos=is_final,
                silence_logit_bias=silence_logit_bias,
                prev_audio=chain,
                restart_context=chain is not None and chain.drops_context(seg_idx),
                stream_retry_fix=self.config.stream_retry_fix,
            )
            local_cb0 = [t - ctx0.dc_base_id for t in seg_cb0_global
                         if ctx0.dc_base_id <= t < ctx0.dc_base_id + SEMANTIC_VOCAB_SIZE]
            state.cb0_segments.append(local_cb0)
            state.accumulated_sem.extend(local_cb0)
            prev_chunk_global = chain.previous if chain is not None else seg_cb0_global

            early_forced = early_ac_per_cb if seg_idx == 0 and early_ac_per_cb else None
            await self._generate_all_acoustic_for_segment(
                state, seg_idx=seg_idx, n_real_segs=n_real_segs,
                segments=segments, voice_prompt=voice_prompt,
                prefix=prefix,
                acoustic_temperature=acoustic_temperature,
                acoustic_top_k=acoustic_top_k,
                system_tag=system_tag,
                early_forced_prefixes=early_forced,
            )

            # VV incremental emit
            chunk, total_vv_emitted = await self._emit_vv_incremental(
                state, total_vv_emitted, vv_slot, dc_hz, chunk_s, vv_adapter=stream_adapter,
            )
            if timings is not None:
                timings.append(("segment", time.time() - start_time, len(state.accumulated_sem) / dc_hz))
            if chunk is not None:
                yield chunk

            if hit_eos and seg_idx < n_real_segs - 1:
                logger.warning("Stopping stream after premature EOS in segment %d", seg_idx)
                break

        # Trailing silence + flush
        self._append_trailing_silence(
            state.accumulated_sem, state.accumulated_ac, self.config.trailing_silence_tokens,
        )
        logger.debug(
            "Flushing stream after %d DualCodec tokens and %d VibeVoice frames",
            len(state.accumulated_sem),
            total_vv_emitted,
        )
        emitted_before_flush = total_vv_emitted
        chunk, total_vv_emitted = await self._emit_vv_incremental(
            state, total_vv_emitted, vv_slot, dc_hz, chunk_s, flush=True, vv_adapter=stream_adapter,
        )
        if chunk is not None and self.config.decode_end_fix:
            # DECODE_END_FIX: the flush decodes every encoder frame, so it runs up to one
            # VibeVoice frame past the codes; end the stream at the codes' length and fade it out.
            chunk = finish_request_end(
                chunk,
                code_samples(len(state.accumulated_sem), dc_hz) - emitted_before_flush * VV_HOP_SAMPLES,
                self.config.decode_end_fade_ms,
            )
        if timings is not None:
            timings.append(("end", time.time() - start_time, len(state.accumulated_sem) / dc_hz))
        if chunk is not None:
            yield chunk

        elapsed = time.time() - start_time
        audio_duration = len(state.accumulated_sem) / dc_hz
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
        logger.info(
            "Streamed %.2fs of audio in %.2fs (RTF %.3f)",
            audio_duration,
            elapsed,
            rtf,
        )

    def unload(self):
        for engine in self.engines:
            shutdown_engine(engine)
        self.engines.clear()
        if self.dc_inference:
            del self.dc_inference
        if self.vv_engine:
            self.vv_engine.stop()
            self.vv_engine = None
        if self.vv_tokenizer:
            del self.vv_tokenizer
        self.vv_adapter = None
        free_memory()


_engine: TTSEngine | None = None


async def get_engine() -> TTSEngine:
    global _engine
    if _engine is None:
        config = ServerConfig()
        _engine = TTSEngine(config)
        await _engine.load()
    return _engine


def unload_engine() -> None:
    global _engine
    if _engine is not None:
        _engine.unload()
        _engine = None

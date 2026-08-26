"""Core TTS inference using vLLM."""
import asyncio
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass

import torch

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

from vllm import SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.config import CompilationConfig
from vllm.config.compilation import CUDAGraphMode

from app.config import (
    ServerConfig,
    END_OF_SPEECH,
    TEXT_MARKER,
    AUDIO_MARKER,
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
    STREAMING_SECONDS_QUANTUM,
    build_system_prompt,
)
from app.model.vllm_model import get_codebook_offset
from app.utils import (
    load_tokenizer,
    tokenize_text,
    encode_voice_prompt,
    resolve_voice_prompt,
    truncate_voice_prompt,
    codes_to_audio_tensor,
    get_dc_inference,
    free_memory,
)
from app.dualcodec_utils import decode_chunked_with_context, split_at_markers, VV_HZ
from app.vibevoice_utils import (
    load_vibevoice_tokenizer,
    vv_encode,
    vv_decode_streaming,
    VVDecodeEngine,
)
from app.model.vllm_plugin import register as register_vllm_model
from app.model_sources import (
    read_codebook_metadata,
    resolve_codebook_model_paths,
    resolve_runtime_model_paths,
    resolve_verbalizer_model_path,
)
from app.logging_utils import get_logger

logger = get_logger("inference")

register_vllm_model()


@dataclass
class _CodebookCtx:
    """Resolved token IDs and engine for a single codebook."""
    tokenizer: object
    engine: AsyncLLMEngine
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
        self.engines: list[AsyncLLMEngine] = []
        self.tokenizers: list = []
        self.dc_inference = None
        self.vv_tokenizer = None
        self.vv_engine: VVDecodeEngine | None = None
        self.prompt_semantic = None
        self.prompt_acoustic = None
        self._voice_prompt_cache: OrderedDict[str, tuple[list, list]] = OrderedDict()
        self._voice_prompt_cache_max = 128
        self.verbalizer = None

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
    ) -> tuple[str, int, int]:
        """Build text window for a segment.  Returns (window_text, win_start, win_end)."""
        n = len(segments)
        boundary_total = n if boundary_total_segments is None else boundary_total_segments
        if not use_context:
            win_start, win_end = seg_idx, seg_idx + 1
        elif seg_idx == 0:
            win_start, win_end = 0, min(2, n)
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
            if global_idx > 0:
                segment = ' ' + segment
            if global_idx == boundary_total - 1:
                segment += '\n'
            win_segs[local_idx] = segment

        # Every non-initial chunk already carries its one leading space.
        window_text = TEXT_MARKER.join(win_segs)
        if prefix:
            separator = '' if window_text.startswith((' ', '\n')) else ' '
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
        frequency_penalty = frequency_penalty if frequency_penalty is not None else 0.0
        system_prompt = build_system_prompt(language, tag)
        system_tag = f'<|im_start|>{system_prompt}<|im_end|>\n'

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
    ) -> tuple[dict | None, dict[int, list[int]]]:
        """Generate or load prefix context. Returns (prefix_dict, prefix_acoustic_codes)."""
        if not prefix_text:
            return None, {}

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
                max_new_tokens=max_new_tokens, system_tag=system_tag,
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
    ) -> tuple['torch.Tensor | None', int]:
        """DC decode -> VV encode -> VV decode. Returns (audio_chunk_or_None, new_total_vv_emitted)."""
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

        with torch.no_grad():
            audio = codes_to_audio_tensor(self.dc_inference, state.accumulated_sem, state.accumulated_ac)
        frames = vv_encode(self.vv_tokenizer, audio)

        if flush:
            audio_chunk = await self.vv_engine.decode(frames[:, total_vv_emitted:], vv_slot)
            new_emitted = frames.shape[1]
        else:
            logger.debug(
                "Streaming decode: %d DualCodec tokens, VibeVoice frames %d:%d",
                len(state.accumulated_sem),
                total_vv_emitted,
                target_vv_frames,
            )
            audio_chunk = await self.vv_engine.decode(
                frames[:, total_vv_emitted:target_vv_frames], vv_slot,
            )
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

    async def load(self):
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
            )
            logger.info("Verbalization model loaded")

        model_paths = list(
            resolve_codebook_model_paths(self.config.codebook_model_dirs)
        )
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

        # Encode the default voice prompt once for the shared stack.
        self.prompt_semantic, self.prompt_acoustic = self._encode_default_prompt(
            self.dc_inference, len(self.config.codebook_model_dirs),
        )

        gpu_mem_fracs = self.config.gpu_memory_per_cb
        max_num_seqs = self.config.max_num_seqs_per_cb
        max_num_batched_tokens = self.config.max_num_batched_tokens_per_cb
        logger.info(
            "Initializing %d vLLM engines with the %s capacity profile",
            len(model_paths),
            self.config.capacity_profile_name,
        )

        engine_tasks = []
        for cb, model_dir_name in enumerate(self.config.codebook_model_dirs):
            max_len = self.config.max_model_len_per_cb[cb]
            gpu_mem = gpu_mem_fracs[cb]
            logger.info(
                "CB%d: context=%d, active_sequences=%d, batched_tokens=%d, GPU fraction=%.3f",
                cb,
                max_len,
                max_num_seqs[cb],
                max_num_batched_tokens[cb],
                gpu_mem,
            )
            engine_tasks.append(self._init_async_engine(
                model_paths[cb],
                target_codebook=cb,
                max_model_len=max_len,
                gpu_memory_utilization=gpu_mem,
                scheduling_policy="priority",
                max_num_seqs=max_num_seqs[cb],
                max_num_batched_tokens=max_num_batched_tokens[cb],
            ))

        self.engines = list(await asyncio.gather(*engine_tasks))

        # Load VibeVoice acoustic tokenizer for postprocessing (shared).
        if self.config.enable_vibevoice:
            logger.info("Loading VibeVoice acoustic tokenizer")
            if runtime_paths.vibevoice is None:
                raise RuntimeError("VibeVoice was enabled but its model did not resolve")
            self.vv_tokenizer = load_vibevoice_tokenizer(runtime_paths.vibevoice)
            self.vv_engine = VVDecodeEngine(
                self.vv_tokenizer,
                max_queue_size=self.config.max_vv_decode_queue_size,
            )
            self.vv_engine.start()
            logger.info("VibeVoice acoustic tokenizer loaded")

        logger.info("Tontaube server ready with %d codebooks", len(self.engines))

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

    async def _init_async_engine(
        self,
        model_path: str,
        target_codebook: int,
        max_model_len: int,
        gpu_memory_utilization: float = 0.2,
        max_num_seqs: int | None = 8,
        max_num_batched_tokens: int | None = None,
        scheduling_policy: str = "fcfs",
    ) -> AsyncLLMEngine:
        logger.debug("Loading CB%d with vLLM from %s", target_codebook, model_path)

        # CPU-only path: no GPU memory fraction, no CUDA graphs / inductor.
        # vllm-cpu rejects compilation_config and gpu_memory_utilization.
        if not torch.cuda.is_available():
            engine_args = AsyncEngineArgs(
                model=model_path,
                dtype="bfloat16",
                hf_overrides={
                    "architectures": ["Qwen3DualCodecForCausalLM"],
                    "target_codebook": target_codebook,
                },
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs or 4,
                max_num_batched_tokens=max_num_batched_tokens,
                enforce_eager=True,
                enable_prefix_caching=False,
                disable_log_stats=True,
                scheduling_policy=scheduling_policy,
            )
            return AsyncLLMEngine.from_engine_args(engine_args)

        # Dense capture at small batch sizes (always used) + sparser capture up
        # to max_num_seqs so high-concurrency decode steps don't fall back to
        # piecewise execution. Each captured size costs ~7 MiB VRAM.
        effective_max_seqs = max_num_seqs or 8
        capture_sizes = [s for s in (1, 2, 3, 4, 5, 6, 7, 8) if s <= effective_max_seqs]
        for s in (12, 16, 20, 24, 28, 32, 40, 48, 56, 64):
            if s <= effective_max_seqs and s not in capture_sizes:
                capture_sizes.append(s)
        if effective_max_seqs not in capture_sizes:
            capture_sizes.append(effective_max_seqs)

        compilation_config = CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL,
            cudagraph_capture_sizes=capture_sizes,
        )

        attention_backend = os.environ.get("VLLM_ATTENTION_BACKEND")

        engine_args = AsyncEngineArgs(
            model=model_path,
            dtype="bfloat16",
            hf_overrides={
                "architectures": ["Qwen3DualCodecForCausalLM"],
                "target_codebook": target_codebook,
            },
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=False,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            disable_log_stats=True,
            max_num_batched_tokens=max_num_batched_tokens,
            compilation_config=compilation_config,
            scheduling_policy=scheduling_policy,
            **({'max_num_seqs': max_num_seqs} if max_num_seqs is not None else {}),
            **({'attention_backend': attention_backend} if attention_backend else {}),
        )

        return AsyncLLMEngine.from_engine_args(engine_args)

    async def _generate_with_engine(
        self,
        engine: AsyncLLMEngine,
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
        vllm_priority: int | None = None,
    ):
        start_time = time.time()

        sampling, voice_prompt, system_tag, acoustic_temperature, acoustic_top_k = self._resolve_params(
            temperature=temperature, top_p=top_p, top_k=top_k,
            frequency_penalty=frequency_penalty, seed=seed,
            voice_path=voice_path, voice_audio_b64=voice_audio_b64,
            voice_tokens=voice_tokens, prompt_max_tokens=prompt_max_tokens,
            acoustic_temperature=acoustic_temperature, acoustic_top_k=acoustic_top_k,
            language=language, tag=tag,
        )
        effective_vv_tokenizer = None if vibevoice_postprocess is False else self.vv_tokenizer
        segments = [text] if isinstance(text, str) else list(text)

        prefix, prefix_acoustic_codes = await self._resolve_prefix(
            prefix_text, prefix_tokens, sampling, voice_prompt,
            max_new_tokens, acoustic_temperature, acoustic_top_k,
            system_tag=system_tag, silence_logit_bias=silence_logit_bias,
        )
        n_codebooks = self._n_codebooks()

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

            semantic_codes, marker_indices = await self._generate_semantic(
                segments, sampling=attempt_sampling, voice_prompt=voice_prompt,
                prefix=prefix,
                max_new_tokens=max_new_tokens,
                system_tag=system_tag,
                vllm_priority=resolved_priority,
                silence_logit_bias=silence_logit_bias,
                use_context=True,
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
            if prefix:
                prefix_len = len(prefix['semantic_codes'])
                semantic_codes = semantic_codes[prefix_len:]
                acoustic_codes = [ac[prefix_len:] for ac in acoustic_codes]
                marker_indices = [m - prefix_len for m in marker_indices[1:]]

            self._append_trailing_silence(
                semantic_codes, acoustic_codes, self.config.trailing_silence_tokens,
            )

            # Decode + VV postprocess
            dc_hz = DUALCODEC_HZ
            short_audio = len(semantic_codes) / dc_hz < 60
            free_memory()
            result = decode_chunked_with_context(
                self.dc_inference, effective_vv_tokenizer,
                semantic_codes, acoustic_codes,
                hz=dc_hz,
                context_seconds=self.config.chunk_context_seconds,
                return_pre_vv=short_audio,
            )
            if short_audio:
                audio_tensor, raw_audio = result
            else:
                audio_tensor = result
                raw_audio = None

            if short_audio and raw_audio is not None:
                retry_reason = self._check_audio_quality(raw_audio, audio_tensor)
                del raw_audio
                if retry_reason is None:
                    break
                logger.warning(
                    "Generation quality retry %d/%d: %s",
                    _rms_attempt,
                    self.config.cb0_max_retries,
                    retry_reason,
                )
            else:
                break

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
        return audio_tensor, codebook_tokens

    @staticmethod
    def _check_audio_quality(
        raw_audio: 'torch.Tensor',
        audio_tensor: 'torch.Tensor',
    ) -> str | None:
        """Compare pre-VV and post-VV audio. Returns retry reason or None if OK."""
        pre_vv = raw_audio.float().cpu()
        post_vv = audio_tensor.float().cpu()
        pre_vv_rms = pre_vv.pow(2).mean().sqrt().item()
        post_vv_rms = post_vv.pow(2).mean().sqrt().item()
        n = min(pre_vv.shape[-1], post_vv.shape[-1])
        diff_rms = (pre_vv[..., :n] - post_vv[..., :n]).pow(2).mean().sqrt().item()
        logger.debug(
            "Decode quality: pre_rms=%.6f pre_peak=%.6f post_rms=%.6f "
            "post_peak=%.6f difference_rms=%.6f samples=%d->%d",
            pre_vv_rms,
            pre_vv.abs().max().item(),
            post_vv_rms,
            post_vv.abs().max().item(),
            diff_rms,
            pre_vv.shape[-1],
            post_vv.shape[-1],
        )
        if post_vv_rms <= 0.002:
            return f"silent (RMS={post_vv_rms:.6f})"
        rms_diff = abs(pre_vv_rms - post_vv_rms)
        if diff_rms > 0.10:
            return (
                f"VV divergence (pre={pre_vv_rms:.6f} post={post_vv_rms:.6f} "
                f"diff between rms={rms_diff:.6f} rms of diff={diff_rms:.6f})"
            )
        if rms_diff > 0.03:
            return (
                f"VV rms diff (pre={pre_vv_rms:.6f} post={post_vv_rms:.6f} "
                f"diff between rms={rms_diff:.6f} rms of diff={diff_rms:.6f})"
            )
        return None

    def _resolve_voice_prompt(
        self,
        voice_audio_b64: str | None,
        voice_path: str | None,
        prompt_max_tokens: int | None = None,
    ) -> tuple[list | None, list | None]:
        return resolve_voice_prompt(
            voice_audio_b64, voice_path,
            self.dc_inference,
            self.config.prompt_seconds,
            prompt_max_tokens if prompt_max_tokens is not None else self.config.prompt_max_tokens,
            self.config.generate_n_codebooks,
            self._voice_prompt_cache,
            self._voice_prompt_cache_max,
            (self.prompt_semantic, self.prompt_acoustic),
        )

    def _check_cb0_generation(
        self, n_tokens: int, n_chars: int, hit_max_tokens: bool, hz: float,
        codes: list[int] | None = None, early_mode: bool = False,
    ) -> str | None:
        """Return failure reason string, or None if generation is acceptable."""
        if hit_max_tokens and not early_mode:
            return "max_tokens_reached"
        # When token cap is expected (early chunk), skip CPS/min_tokens — only check silence
        if not early_mode:
            if n_tokens < self.config.cb0_min_tokens:
                return f"too_few_tokens ({n_tokens})"
            if n_tokens > 0 and n_chars > 30:
                duration = n_tokens / hz
                cps = n_chars / duration
                if cps < self.config.cb0_min_cps or cps > self.config.cb0_max_cps:
                    return f"cps_out_of_range (cps={cps:.1f})"
        if codes and self.config.cb0_max_silence_ratio > 0:
            check = codes[:50]
            silence_set = set(SILENCE_TOKEN_IDS)
            n_silence = sum(1 for c in check if c in silence_set)
            ratio = n_silence / len(check)
            if ratio > self.config.cb0_max_silence_ratio:
                return f"too_much_silence ({n_silence}/{len(check)} = {ratio:.0%})"
        return None

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
    ) -> tuple[list[int], bool]:
        """Core semantic generation for one segment.

        Returns (global_semantic_ids, hit_eos).

        allow_marker_iteration: if True, continue generating past audio_markers
            (batch single-segment mode — model may emit markers for long text).
        stop_at_eos: if False, exclude EOS from stop tokens and replace any EOS
            in output with audio_marker (streaming non-final segments).
        silence_logit_bias: optional per-request base bias for the CB0
            collapse-silence group. The configured retry-step adjustment is added.
        """
        ctx = self._cb_ctx(0)
        n_segs = len(segments)

        if use_context is None:
            use_context = prev_chunk_global is not None or seg_idx == 0
        window_text, win_start, win_end = self._text_window(
            seg_idx, segments, use_context=use_context,
            prefix=prefix, system_tag=system_tag,
            boundary_total_segments=boundary_total_segments,
        )
        text_ids = tokenize_text(window_text, ctx.tokenizer)

        vp_cap = self.config.prompt_max_tokens_per_cb[0] if self.config.prompt_max_tokens_per_cb else None
        prompt_semantic = voice_prompt[0][:vp_cap] if voice_prompt[0] and vp_cap else voice_prompt[0]

        force_silence = self.config.force_first_silence and seg_idx == 0 and not forced_prefix
        silence_token = ctx.dc_base_id + SILENCE_TOKEN_ID

        last_is_marker = False
        last_is_eos = False
        semantic_only: list[int] = []

        for attempt in range(1, self.config.cb0_max_retries + 1):
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
                    + (attempt - 1) * self.config.cb0_silence_logit_bias_step
                )
                logit_bias = {ctx.eos_id: self.config.cb0_eos_logit_bias}
                for sid in COLLAPSE_SILENCE_TOKEN_IDS:
                    logit_bias[ctx.dc_base_id + sid] = collapse_silence_bias
                for sid in ORDINARY_SILENCE_TOKEN_IDS:
                    logit_bias[ctx.dc_base_id + sid] = self.config.cb0_ordinary_silence_logit_bias
                sampling_params = SamplingParams(
                    **sampling,
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
                len(semantic_only), len(segments[seg_idx]), hit_max, ctx.hz,
                codes=semantic_codes_local, early_mode=early_mode,
            )
            if failure is None:
                break
            logger.warning(
                "CB0 segment %d retry %d/%d: %s",
                seg_idx,
                attempt,
                self.config.cb0_max_retries,
                failure,
            )

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
    ) -> tuple[list[int], list[int]]:
        """Generate semantic tokens for one or more segments.

        Returns (local_semantic_codes, marker_indices).
        """
        ctx = self._cb_ctx(0)
        n_segs = len(segments)
        single = n_segs == 1

        audio_chunks: list[list[int]] = []
        prev_global: list[int] | None = None
        finished = False

        for seg_idx in range(n_segs):
            if finished:
                break
            forced = first_seg_forced_prefix if seg_idx == 0 else None
            seg_global, hit_eos = await self._generate_semantic_segment(
                seg_idx, segments,
                prev_chunk_global=prev_global if use_context else None,
                sampling=sampling, voice_prompt=voice_prompt,
                prefix=prefix, max_new_tokens=max_new_tokens,
                forced_prefix=forced, system_tag=system_tag,
                allow_marker_iteration=single,
                priority=vllm_priority,
                silence_logit_bias=silence_logit_bias,
                use_context=use_context,
            )
            audio_chunks.append(seg_global)
            prev_global = seg_global

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
    ):
        """Wrapper that guarantees VV slot cleanup even on cancellation/error."""
        if self.vv_tokenizer is None:
            raise RuntimeError(
                "generate_streaming requires the VibeVoice acoustic tokenizer"
            )
        vv_slot = self.vv_engine.alloc_slot() if self.vv_engine else None
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
                language=language, tag=tag,
            ):
                yield chunk
        finally:
            if vv_slot is not None:
                self.vv_engine.free_slot(vv_slot)
                logger.debug("Released VibeVoice streaming slot %s", vv_slot)

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
            language=language, tag=tag,
        )
        segments = [text] if isinstance(text, str) else list(text)
        if text_preprocessor and segments:
            segments[0] = await text_preprocessor(segments[0])

        prefix, prefix_acoustic_codes = await self._resolve_prefix(
            prefix_text, prefix_tokens, sampling, voice_prompt,
            max_new_tokens, acoustic_temperature, acoustic_top_k,
            system_tag=system_tag, silence_logit_bias=silence_logit_bias,
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
        early_cb0: list[int] = []
        if segments:
            early_global, _ = await self._generate_semantic_segment(
                seg_idx=0, segments=[segments[0]],
                prev_chunk_global=None,
                sampling=sampling, voice_prompt=voice_prompt,
                prefix=prefix, max_new_tokens=early_dc_tokens,
                priority=0, system_tag=system_tag,
                early_mode=True,
                boundary_total_segments=len(segments),
                silence_logit_bias=silence_logit_bias,
            )
            ctx0 = self._cb_ctx(0)
            early_cb0 = [t - ctx0.dc_base_id for t in early_global
                         if ctx0.dc_base_id <= t < ctx0.dc_base_id + SEMANTIC_VOCAB_SIZE]
            if self.config.force_first_silence and not early_cb0:
                early_cb0 = [SILENCE_TOKEN_ID]
            logger.debug("Generated %d early CB0 streaming tokens", len(early_cb0))
        # --- VV state and config ---
        chunk_s = self.config.streaming_chunk_seconds
        n_codebooks = self._n_codebooks()
        total_vv_emitted = 0

        # --- Early acoustic + yield first audio ---
        early_ac_per_cb: list[list[int]] = []
        if early_cb0:
            effective_early_dc = len(early_cb0) - pad
            if effective_early_dc > 0:
                dc_tokens_per_quantum = int(round(STREAMING_SECONDS_QUANTUM * dc_hz))
                vv_frames_per_quantum = int(round(STREAMING_SECONDS_QUANTUM * VV_HZ))
                available_quanta = effective_early_dc // dc_tokens_per_quantum
                requested_quanta = round(
                    streaming_initial_seconds / STREAMING_SECONDS_QUANTUM
                )
                early_target_vv = (
                    min(available_quanta, requested_quanta)
                    * vv_frames_per_quantum
                )
                if early_target_vv > 0:
                    # Build a temporary streaming state for the early chunk
                    early_state = self._build_streaming_state(prefix, prefix_acoustic_codes)
                    early_state.cb0_segments.append(early_cb0)
                    early_state.accumulated_sem.extend(early_cb0)

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
                        early_ac_per_cb.append(early_state.ac_segs[cb_idx][n_prefix_slots])
                    logger.debug(
                        "Early stream: %d DualCodec tokens -> %d VibeVoice frames",
                        len(early_cb0),
                        early_target_vv,
                    )
                    with torch.no_grad():
                        early_audio = codes_to_audio_tensor(self.dc_inference, early_cb0, early_ac_per_cb)
                    early_frames = vv_encode(self.vv_tokenizer, early_audio)
                    audio_chunk = await self.vv_engine.decode(
                        early_frames[:, :early_target_vv], vv_slot,
                    )
                    total_vv_emitted = early_target_vv
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
                priority=10, system_tag=system_tag,
                stop_at_eos=is_final,
                silence_logit_bias=silence_logit_bias,
            )
            local_cb0 = [t - ctx0.dc_base_id for t in seg_cb0_global
                         if ctx0.dc_base_id <= t < ctx0.dc_base_id + SEMANTIC_VOCAB_SIZE]
            state.cb0_segments.append(local_cb0)
            state.accumulated_sem.extend(local_cb0)
            prev_chunk_global = seg_cb0_global

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
                state, total_vv_emitted, vv_slot, dc_hz, chunk_s,
            )
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
        chunk, total_vv_emitted = await self._emit_vv_incremental(
            state, total_vv_emitted, vv_slot, dc_hz, chunk_s, flush=True,
        )
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
            shutdown = getattr(engine, "shutdown", None)
            if shutdown is not None:
                shutdown()
            else:
                engine.shutdown_background_loop()
        self.engines.clear()
        if self.dc_inference:
            del self.dc_inference
        if self.vv_engine:
            self.vv_engine.stop()
            self.vv_engine = None
        if self.vv_tokenizer:
            del self.vv_tokenizer
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

"""Text verbalization: convert written text to spoken form using a fine-tuned Qwen model.

Runs a vLLM AsyncLLMEngine with ngram speculative decoding, or the lite
backend's single-process engine with weight-only quantization.
The model was trained to normalize numbers, symbols, abbreviations, etc. into
their natural spoken equivalents (e.g. "$14.75" -> "fourteen dollars seventy five cents").
"""

import re
import time
import uuid
from collections import Counter
from transformers import AutoTokenizer

from app.engines import build_verbalizer_engine
from app.logging_utils import get_logger
from app.verbalizer_context import context_tail, ends_sentence, match_first_case
from app.verbalizer_guard import Prepared, check_growth, check_output, prepare, split_continuation

logger = get_logger("verbalizer")

# Fallback counters for this process (also logged at INFO): no_output / rejected (_verbalize_plain served the
# pre-pass text), context_used / context_not_used / context_error (verbalize_continuation's joint reading).
COUNTS: Counter = Counter()


# ── Numsplit tokenization (private to this module) ───────────────────────────

_DIGIT_SPLIT_RE = re.compile(r'(\d+)')
_TAG_RE = re.compile(r'(<<[^>]+>>)')
_char_token_cache: dict[str, int] = {}


def tokenize_numsplit(text: str, tokenizer) -> list[int]:
    """Normal tokenization, but digit sequences are split into individual digits
    and each digit is tokenized separately (char-level)."""
    result = []
    for part in _DIGIT_SPLIT_RE.split(text):
        if not part:
            continue
        if part[0].isdigit():
            for c in part:
                if c not in _char_token_cache:
                    ids = tokenizer(c, add_special_tokens=False)['input_ids']
                    _char_token_cache[c] = ids[0]
                result.append(_char_token_cache[c])
        else:
            result.extend(tokenizer.encode(part, add_special_tokens=False))
    return result


# ── Verbalizer ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a text normalizer. Convert the provided text into its natural spoken form. "
    "Keep random looking strings/un-normalizable strings as they are.\n"
    "For example: Expand all numbers and special chars into words. "
    "Convert spelled out terms/initialism to full uppercase version "
    "(F.B.I \u2192 FBI, http \u2192 HTTP), and acronyms to normal capitalized words "
    "(NASA \u2192 Nasa).\n"
    "Convert symbols to words ($ \u2192 dollars, % \u2192 percent, @ \u2192 at, "
    "\u00b1 \u2192 plus or minus). Convert dates and times to spoken form.\n"
    "Never expand initialisms/acronyms into full words. Keep them as-is: "
    "VP stays VP, CEO stays CEO, US stays US, etc.\n"
    "Do not paraphrase or rephrase. Do not think. "
    "Output only the normalized spoken text, nothing else."
)

THINKING_TAG = "<think></think>"
BEGIN_TAG = "<tool_call>"
END_TAG = "</tool_call>"

VERBALIZE_MAX_SEQ_LEN = 480


def _guard(prepared: Prepared, output: str, growth_guard: bool) -> tuple[bool, str]:
    """The post-checks an output must pass: check_output (no dropped / changed word or span), then, when
    VERBALIZER_GROWTH_GUARD is on, check_growth (no added word, no runaway length)."""
    ok, reason = check_output(prepared, output)
    if ok and growth_guard:
        ok, reason = check_growth(prepared, output)
    return ok, reason


class Verbalizer:
    """Wraps a fine-tuned Qwen verbalization model served by vLLM."""

    growth_guard: bool = True   # VERBALIZER_GROWTH_GUARD (ServerConfig.verbalizer_growth_guard)

    def __init__(
        self,
        model_path: str,
        gpu_util: float = 0.18,
        max_num_seqs: int = 1,
        growth_guard: bool = True,
        backend: str = "vllm",
        lite_quant: str = "int8g128",
    ):
        self.growth_guard = growth_guard
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            fix_mistral_regex=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.engine, self.sampling_params, self._tokens_prompt = build_verbalizer_engine(
            backend, model_path, VERBALIZE_MAX_SEQ_LEN, max_num_seqs, gpu_util, lite_quant)

        # Pre-encode fixed prompt parts
        self._prefix_ids = self.tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=True)
        self._think_ids = self.tokenizer.encode(THINKING_TAG, add_special_tokens=False)

    async def _generate(self, text: str) -> str | None:
        """One model call on ``text`` (already through the pre-pass): the text between the output tags, or None
        when the model gave no output or no tagged text."""
        text_ids = tokenize_numsplit(text, self.tokenizer)
        prompt_ids = self._prefix_ids + text_ids + self._think_ids

        results = self.engine.generate(
            prompt=self._tokens_prompt(prompt_token_ids=prompt_ids),
            sampling_params=self.sampling_params,
            request_id=str(uuid.uuid4()),
        )
        final_output = None
        async for output in results:
            final_output = output

        if final_output is None or not final_output.outputs:
            return None

        generated_ids = final_output.outputs[0].token_ids

        # Strip leading think tag if the model re-emits it
        think_ids = self._think_ids
        if list(generated_ids[:len(think_ids)]) == think_ids:
            generated_ids = generated_ids[len(think_ids):]

        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._extract_tagged(decoded)

    async def _verbalize_plain(self, text: str) -> str:
        """Verbalize a plain text string (no tags).

        The guard's pre-pass first rewrites patterns the model misreads (clock
        times, file extensions, identifiers, codes) into spoken words; its
        post-check rejects an output that drops or changes them or a plain
        word, and (growth_guard) one that adds words or runs far longer than
        the input. On failure or rejection the pre-pass text is returned.
        """
        prepared = prepare(text)
        text = prepared.text
        result = await self._generate(text)
        if result is None:
            COUNTS["no_output"] += 1
            logger.info("Verbalizer gave no output: %r", text[:60])
            return text
        ok, reason = _guard(prepared, result, self.growth_guard)
        if not ok:
            COUNTS["rejected"] += 1
            logger.info("Verbalizer output rejected (%s): %r -> %r", reason, text[:60], result[:60])
            return text
        return result

    async def verbalize_continuation(self, prefix_text: str, text: str) -> str:
        """Verbalize ``text``, the rest of a sentence whose start ``prefix_text`` was already spoken (a prefix
        continuation; VERBALIZE_PREFIX_CONTEXT). <<tags>> are kept as in verbalize().

        The first plain part of ``text`` is read together with the end of the prefix (context_tail), so the model
        sees a whole sentence, and only the rest's part of the output is kept (split_continuation). That part must
        pass the guard against the rest's own pre-pass text; if the split is uncertain or the part is rejected, the
        rest is verbalized alone as before (_verbalize_plain). A prefix that ends a sentence needs no context (the
        rest starts its own sentence): the rest is verbalized alone. Either way the rest keeps its own first-letter
        case (match_first_case). Later plain parts are verbalized as in verbalize().
        """
        t0 = time.perf_counter()
        context = context_tail(prefix_text)
        result_parts = []
        first = True
        for part in _TAG_RE.split(text):
            if _TAG_RE.fullmatch(part):
                result_parts.append(part)
            elif part.strip():
                if first:
                    result_parts.append(await self._verbalize_rest(context, part))
                    first = False
                else:
                    result_parts.append(await self._verbalize_plain(part))
            else:
                result_parts.append(part)
        result = ''.join(result_parts)
        logger.debug(
            "Verbalized continuation in %.0fms (context %r): %r -> %r",
            (time.perf_counter() - t0) * 1000, context[-40:], text[:60], result[:60],
        )
        return result

    async def _verbalize_rest(self, context: str, rest: str) -> str:
        """The rest of a sentence, read after ``context``; see verbalize_continuation."""
        if context and not ends_sentence(context):
            try:
                rest_prepared = prepare(rest)
                context_text = prepare(context).text
                output = await self._generate(f"{context_text} {rest_prepared.text}")
                part = split_continuation(context_text, rest_prepared.text, output) if output is not None else None
                if part is None:
                    reason = "no output" if output is None else "no certain split"
                else:
                    part = match_first_case(rest, part, context)
                    ok, reason = _guard(rest_prepared, part, self.growth_guard)
                    if ok:
                        COUNTS["context_used"] += 1
                        return part
            except Exception as exc:   # the joint reading is an extra; any failure falls back to the rest alone
                COUNTS["context_error"] += 1
                reason = f"error {exc!r}"
                logger.warning("Continuation context failed, verbalizing the rest alone: %r", exc)
            COUNTS["context_not_used"] += 1
            logger.info("Continuation context not used (%s): %r | %r", reason, context[-40:], rest[:60])
        return match_first_case(rest, await self._verbalize_plain(rest), context)

    async def verbalize(self, text: str) -> str:
        """Convert written text to spoken form, preserving <<tags>> as-is."""
        t0 = time.perf_counter()

        parts = _TAG_RE.split(text)
        result_parts = []
        for part in parts:
            if _TAG_RE.fullmatch(part):
                result_parts.append(part)
            elif part.strip():
                result_parts.append(await self._verbalize_plain(part))
            else:
                result_parts.append(part)

        result = ''.join(result_parts)

        t_elapsed = (time.perf_counter() - t0) * 1000
        changed = result != text
        logger.debug(
            "Verbalized in %.0fms%s: %r -> %r",
            t_elapsed,
            " (changed)" if changed else "",
            text[:60],
            result[:60],
        )
        return result

    @staticmethod
    def _extract_tagged(decoded: str) -> str | None:
        """The non-empty text between <tool_call> and </tool_call>, or None."""
        start = decoded.find(BEGIN_TAG)
        end = decoded.find(END_TAG)
        if start != -1 and end != -1 and end > start:
            result = decoded[start + len(BEGIN_TAG):end].strip()
            if result:
                return result
        return None

    @staticmethod
    def _extract_output(decoded: str, original: str) -> str:
        """Extract text between <tool_call> and </tool_call> tags.

        Returns the original text unchanged if tags are not found (safe fallback).
        """
        result = Verbalizer._extract_tagged(decoded)
        # Tags not found or empty content -> return original text unchanged
        return original if result is None else result

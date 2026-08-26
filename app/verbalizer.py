"""Text verbalization: convert written text to spoken form using a fine-tuned Qwen model.

Runs a vLLM AsyncLLMEngine with ngram speculative decoding for fast inference.
The model was trained to normalize numbers, symbols, abbreviations, etc. into
their natural spoken equivalents (e.g. "$14.75" -> "fourteen dollars seventy five cents").
"""

import re
import time
import uuid
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer
from app.logging_utils import get_logger

logger = get_logger("verbalizer")


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

class Verbalizer:
    """Wraps a fine-tuned Qwen verbalization model served by vLLM."""

    def __init__(
        self,
        model_path: str,
        gpu_util: float = 0.18,
        max_num_seqs: int = 1,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            fix_mistral_regex=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        engine_args = AsyncEngineArgs(
            model=model_path,
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 10,
                "prompt_lookup_max": 8,
                "prompt_lookup_min": 3,
            },
            max_model_len=VERBALIZE_MAX_SEQ_LEN,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=VERBALIZE_MAX_SEQ_LEN * max_num_seqs,
            gpu_memory_utilization=gpu_util,
            enforce_eager=False,
            dtype="bfloat16",
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1024,
        )

        # Pre-encode fixed prompt parts
        self._prefix_ids = self.tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=True)
        self._think_ids = self.tokenizer.encode(THINKING_TAG, add_special_tokens=False)

    async def _verbalize_plain(self, text: str) -> str:
        """Verbalize a plain text string (no tags). Returns original on failure."""
        text_ids = tokenize_numsplit(text, self.tokenizer)
        prompt_ids = self._prefix_ids + text_ids + self._think_ids

        results = self.engine.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_ids),
            sampling_params=self.sampling_params,
            request_id=str(uuid.uuid4()),
        )
        final_output = None
        async for output in results:
            final_output = output

        if final_output is None or not final_output.outputs:
            return text

        generated_ids = final_output.outputs[0].token_ids

        # Strip leading think tag if the model re-emits it
        think_ids = self._think_ids
        if list(generated_ids[:len(think_ids)]) == think_ids:
            generated_ids = generated_ids[len(think_ids):]

        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._extract_output(decoded, text)

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
    def _extract_output(decoded: str, original: str) -> str:
        """Extract text between <tool_call> and </tool_call> tags.

        Returns the original text unchanged if tags are not found (safe fallback).
        """
        start = decoded.find(BEGIN_TAG)
        end = decoded.find(END_TAG)
        if start != -1 and end != -1 and end > start:
            result = decoded[start + len(BEGIN_TAG):end].strip()
            if result:
                return result
        # Tags not found or empty content -> return original text unchanged
        return original

"""Text verbalization: convert written text to spoken form using a fine-tuned Qwen model.

The model rewrites numbers, symbols, abbreviations, markup and so on into the words a narrator
would say (e.g. "$14.75" -> "fourteen dollars and seventy-five cents").

On the lite backend the output is decoded by draft and verify: a cheap guess of the spoken text
(app/verbalizer_draft.py) is checked by the model many tokens per pass (app/lite/gpu/block.py), and
several inputs share each pass. The result is the model's greedy output. VERBALIZER_DRAFT=0 uses the
lite engine one token at a time; the vLLM backend uses n-gram speculative decoding.

The model copies long numbers and Roman numerals between marker tokens; app/verbalizer_markers.py writes the words.
"""

import asyncio
import json
import re
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from app.engines import build_verbalizer_block_lm, build_verbalizer_engine
from app.logging_utils import get_logger
from app.text import split_long_text
from app.verbalizer_context import context_tail, ends_sentence, match_first_case
from app.verbalizer_draft import DraftFollower, Normalizer
from app.verbalizer_guard import Prepared, check_growth, check_output, prepare, split_continuation
from app.verbalizer_markers import ANY_MARKER_RE, expand_copied

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

VERBALIZE_MAX_SEQ_LEN = 1024

# Size of one model input in characters; a digit counts as four because it becomes a word.
CHUNK_CHARS = 600
CHUNK_MAX_CHARS = 800

LANES = 8                    # inputs decoded side by side in the draft-and-verify path
BLOCK_TOKENS = (8, 32, 96)   # drafted tokens per pass: smallest, first, largest

# Zero-width and directional marks, soft hyphens, variation selectors, tag characters and unpaired surrogates.
_INVISIBLE_RE = re.compile('[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\ufe00-\ufe0f\U000e0000-\U000e01ef\ud800-\udfff]')
_PARAGRAPH_GAP_RE = re.compile(r'(\n[ \t]*\n\s*)')
_SENTENCE_GAP_RE = re.compile(r'((?<=[.!?])\s+)')
_TABLE_ROW_RE = re.compile(r'\t|^\s*\|.*\|\s*$')
_TABLE_DIVIDER_RE = re.compile(r'^[\s|:-]+$')


def _weight(text: str) -> int:
    return len(text) + 3 * sum(char.isdigit() for char in text)


def _is_heading(piece: str) -> bool:
    return "\n" not in piece and 0 < len(piece.strip()) <= 80 and piece.rstrip()[-1] not in ".!?:;,"


def split_for_verbalizer(text: str) -> list[tuple[str, str]]:
    """Split text into model inputs, each paired with the separator that followed it.

    Cuts at blank lines, then line breaks, then sentence ends. A heading stays with the
    text after it, and every part of a split table starts with the table's header row.
    """
    units: list[tuple[str, str, str]] = []  # piece, separator after it, table header to repeat
    fields = _PARAGRAPH_GAP_RE.split(text)
    for para, gap in zip(fields[::2], fields[1::2] + [""]):
        if _weight(para) <= CHUNK_MAX_CHARS:
            units.append((para, gap, ""))
            continue
        lines = para.split("\n")
        header = ""
        if len(lines) > 2 and all(_TABLE_ROW_RE.search(line) for line in lines):
            header = lines[0] + "\n" + (lines[1] + "\n" if _TABLE_DIVIDER_RE.match(lines[1]) else "")
        for index, line in enumerate(lines):
            sep = "\n" if index < len(lines) - 1 else gap
            if _weight(line) <= CHUNK_MAX_CHARS:
                units.append((line, sep, "" if header.startswith(line + "\n") else header))
                continue
            parts = _SENTENCE_GAP_RE.split(line)
            for sentence, space in zip(parts[::2], parts[1::2] + [sep]):
                pieces = split_long_text([sentence], CHUNK_MAX_CHARS, digit_weight=4) or [sentence]
                units += [(piece, " ", "") for piece in pieces[:-1]] + [(pieces[-1], space, "")]

    chunks: list[tuple[str, str]] = []
    current: list[tuple[str, str, str]] = []

    def flush(carry_heading: bool) -> None:
        carried = []
        if carry_heading and len(current) > 1 and _is_heading(current[-1][0]) and not _is_heading(current[-2][0]):
            carried.append(current.pop())
        body = "".join(piece + sep for piece, sep, _ in current[:-1]) + current[-1][0]
        chunks.append((current[0][2] + body, current[-1][1]))
        current[:] = carried

    for unit in units:
        size = _weight("".join(piece + sep for piece, sep, _ in current) + unit[0])
        only_heading = len(current) == 1 and _is_heading(current[0][0])
        if current and size > (CHUNK_MAX_CHARS if only_heading else CHUNK_CHARS):
            flush(carry_heading=True)
        current.append(unit)
    if current:
        flush(carry_heading=False)
    return chunks


def _plausible(source: str, spoken: str) -> bool:
    """Reject outputs that lost most of the text or ran away. Digits are not checked: the model leaves
    them on purpose in text it copies unchanged (other languages, hashes, identifiers)."""
    # A table row repeats every header, so the reading can be several times as long as the input.
    return 0.3 * len(source) <= len(spoken) <= 3 * _weight(source) + 40


class _Lane:
    """One input being decoded by draft and verify."""

    def __init__(self, index: int, lane: int, cached: int, pending: list[int], follower: DraftFollower, limit: int):
        self.index, self.lane, self.cached, self.pending, self.follower = index, lane, cached, pending, follower
        self.limit, self.window, self.out = limit, BLOCK_TOKENS[1], []


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
        draft: bool = True,
        guard: bool = False,
    ):
        self.growth_guard = growth_guard
        self.guard = guard   # VERBALIZER_GUARD: the pre-pass and word checks built for the first verbalizer model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            fix_mistral_regex=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Pre-encode fixed prompt parts
        self._prefix_ids = self.tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=True)
        self._think_ids = self.tokenizer.encode(THINKING_TAG, add_special_tokens=False)
        self._suffix_ids = self._think_ids + self.tokenizer.encode(BEGIN_TAG, add_special_tokens=False)

        self.block = None
        if backend == "lite" and draft:
            self.block = build_verbalizer_block_lm(model_path, lite_quant, VERBALIZE_MAX_SEQ_LEN, LANES, self._prefix_ids)
            stop = {self.tokenizer.eos_token_id, self.tokenizer.encode(END_TAG, add_special_tokens=False)[0]}
            generation = Path(model_path) / "generation_config.json"
            if generation.exists():
                eos = json.loads(generation.read_text()).get("eos_token_id")
                stop.update(eos if isinstance(eos, list) else [eos])
            self._stop = {token for token in stop if token is not None}
            self._template = set(self._suffix_ids)   # prompt tags: one inside an output means the model lost its place
            self._pieces: dict[int, str] = {}
            self._lock = threading.Lock()
            self._queue: list[tuple[str, asyncio.Future]] = []
            self._draining = False
        else:
            self.engine, self.sampling_params, self._tokens_prompt = build_verbalizer_engine(
                backend, model_path, VERBALIZE_MAX_SEQ_LEN, max_num_seqs, gpu_util, lite_quant)

    def _piece(self, token: int) -> str:
        piece = self._pieces.get(token)
        if piece is None:
            piece = self._pieces[token] = self.tokenizer.decode([token])
        return piece

    def _run(self, texts: list[str]) -> list[str | None]:
        """Draft-and-verify decoding of each text; None where the model did not close its output."""
        lm, stop, encode = self.block, self._stop, lambda text: tokenize_numsplit(text, self.tokenizer)
        halt = stop | self._template
        w_min, _, w_max = BLOCK_TOKENS
        normalizer = Normalizer()   # also remembers the respellings made within this batch of texts
        results: list[str | None] = [None] * len(texts)
        waiting = list(range(len(texts)))[::-1]
        free = list(range(lm.lanes))[::-1]
        active: list[_Lane] = []
        hold = False   # after cache blocks ran short, no new input starts until one has finished
        with self._lock:
            while waiting or active:
                while waiting and free and not hold:
                    index = waiting.pop()
                    try:
                        pending = encode(texts[index]) + self._suffix_ids
                        text_draft = normalizer.draft(texts[index])
                    except Exception:   # this text only: the others in the batch must not fail with it
                        logger.exception("Verbalizer could not prepare %r", texts[index][:60])
                        continue
                    room = lm.max_len - lm.prefix_len - len(pending)
                    if room < 8 or len(pending) > lm.max_rows:
                        continue   # too long for the model: left as None
                    if not lm.reserve(free[-1], lm.prefix_len + len(pending)):
                        waiting.append(index)
                        break      # the cache pool is in use: this input starts when blocks come back
                    follower = DraftFollower(text_draft, encode, self._piece)
                    active.append(_Lane(index, free.pop(), lm.prefix_len, pending, follower, min(room, 3 * len(pending) + 48)))
                parts, chosen, rows, heads = [], [], 0, 0
                for lane in active:
                    room = min(lane.window, lane.limit - len(lane.out) - 1,
                               lm.max_rows - rows - len(lane.pending), lm.max_heads - heads - 1)
                    if room < 0:
                        continue   # no space left in this pass: the input waits one round
                    draft = lane.follower.propose(room) if room > 0 else []
                    used = lane.cached + len(lane.pending)
                    if draft and not lm.reserve(lane.lane, used + len(draft)):
                        draft = []
                    if not lm.reserve(lane.lane, used):
                        continue   # no cache block free for its next token: the input waits
                    parts.append((lane.lane, lane.pending + draft, lane.cached, len(lane.pending) - 1))
                    chosen.append((lane, draft))
                    rows += len(lane.pending) + len(draft)
                    heads += len(draft) + 1
                if not parts:
                    if not active:
                        break
                    # Every active input waits for cache blocks: the one started last gives its blocks back and
                    # starts over later.
                    victim = active.pop()
                    lm.release(victim.lane)
                    free.append(victim.lane)
                    waiting.append(victim.index)
                    hold = bool(active)   # alone, an input always fits
                    continue
                if len(chosen) < len(active):
                    active = active[len(chosen):] + active[:len(chosen)]
                for (lane, draft), predicted in zip(chosen, lm.forward(parts)):
                    accepted = 0
                    while accepted < len(draft) and draft[accepted] == predicted[accepted] and draft[accepted] not in halt:
                        accepted += 1
                    lane.cached += len(lane.pending) + accepted
                    lane.pending = [predicted[accepted]]
                    lane.out += draft[:accepted]
                    done = predicted[accepted] in stop
                    lost = predicted[accepted] in self._template
                    if not done and not lost:
                        lane.out.append(predicted[accepted])
                    if done or lost or len(lane.out) >= lane.limit:
                        if done:
                            results[lane.index] = self.tokenizer.decode(lane.out, skip_special_tokens=True).strip() or None
                            if results[lane.index]:
                                normalizer.learn(texts[lane.index], results[lane.index])
                        lm.release(lane.lane)
                        free.append(lane.lane)
                        active.remove(lane)
                        hold = False
                        continue
                    lane.follower.advance(lane.out)
                    if draft:   # blocks grow while the draft holds and shrink where it fails
                        lane.window = (min(w_max, 2 * lane.window) if accepted == len(draft)
                                       else max(w_min, min(lane.window, 2 * accepted + w_min)))
        return results

    async def _drain(self) -> None:
        """Decode everything queued by _generate in shared passes (callers of one event-loop turn go together)."""
        try:
            while self._queue:
                await asyncio.sleep(0)
                batch, self._queue = self._queue, []
                try:
                    outputs = await asyncio.to_thread(self._run, [text for text, _ in batch])
                except Exception as exc:
                    for _, future in batch:
                        if not future.done():
                            future.set_exception(exc)
                    continue
                for (_, future), output in zip(batch, outputs):
                    if not future.done():
                        future.set_result(output)
        finally:
            self._draining = False

    async def _generate(self, text: str) -> str | None:
        """One model call on ``text``: the spoken text, or None when the model gave no output, did not close it,
        or copied a number or Roman numeral wrongly into a marker."""
        if self.block is not None:
            future = asyncio.get_running_loop().create_future()
            self._queue.append((text, future))
            if not self._draining:
                self._draining = True
                asyncio.ensure_future(self._drain())
            output = await future
            return None if output is None else expand_copied(text, output)
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
        output = self._extract_tagged(decoded)
        return None if output is None else expand_copied(text, output)

    async def _verbalize_plain(self, text: str) -> str:
        """Verbalize a plain text string (no tags).

        The guard's pre-pass first rewrites patterns the model misreads (clock
        times, file extensions, identifiers, codes) into spoken words; its
        post-check rejects an output that drops or changes them or a plain
        word, and (growth_guard) one that adds words or runs far longer than
        the input. On failure or rejection the pre-pass text is returned.
        """
        if not self.guard:
            return await self._verbalize_checked(text)
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

    async def _verbalize_checked(self, text: str) -> str:
        """The model's reading of ``text`` if it is plausible; else sentence by sentence, each falling back to
        its written form."""
        result = await self._generate(text)
        if result is not None and _plausible(text, result):
            return result
        COUNTS["no_output" if result is None else "rejected"] += 1
        logger.info("Verbalizer output %s: %r", "missing" if result is None else "rejected", text[:60])
        parts = _SENTENCE_GAP_RE.split(text)
        sentences = [part for part in parts[::2] if part.strip()]
        if len(sentences) < 2:
            return text
        readings = iter(await asyncio.gather(*[self._generate(sentence) for sentence in sentences]))
        out = []
        for index, part in enumerate(parts):
            if index % 2 or not part.strip():
                out.append(part)
                continue
            reading = next(readings)
            out.append(reading if reading is not None and _plausible(part, reading) else part)
        return "".join(out)

    def _prepare(self, text: str) -> tuple[str, Prepared | None]:
        """The text the model reads and what the post-check needs (the guard's pre-pass, when it is on)."""
        if not self.guard:
            return text, None
        prepared = prepare(text)
        return prepared.text, prepared

    def _accept(self, prepared: Prepared | None, source: str, output: str) -> tuple[bool, str]:
        if prepared is None:
            return _plausible(source, output), "implausible"
        return _guard(prepared, output, self.growth_guard)

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
        prefix_text, text = ANY_MARKER_RE.sub("", prefix_text), ANY_MARKER_RE.sub("", text)
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
                rest_text, rest_prepared = self._prepare(rest)
                context_text, _ = self._prepare(context)
                output = await self._generate(f"{context_text} {rest_text}")
                part = split_continuation(context_text, rest_text, output) if output is not None else None
                if part is None:
                    reason = "no output" if output is None else "no certain split"
                else:
                    part = match_first_case(rest, part, context)
                    ok, reason = self._accept(rest_prepared, rest_text, part)
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
        """Convert written text of any length to spoken form, preserving <<tags>> as-is."""
        t0 = time.perf_counter()
        text = ANY_MARKER_RE.sub("", _INVISIBLE_RE.sub("", text))

        # Long text is cut at paragraphs; all pieces are read together, so they share model passes.
        layout: list[str | list[tuple[int, str]]] = []
        pieces: list[str] = []
        for part in _TAG_RE.split(text):
            if _TAG_RE.fullmatch(part) or not part.strip():
                layout.append(part)
                continue
            chunks = split_for_verbalizer(part)
            layout.append([(len(pieces) + index, sep) for index, (_, sep) in enumerate(chunks)])
            pieces += [chunk for chunk, _ in chunks]
        spoken = await asyncio.gather(*[self._verbalize_plain(piece) for piece in pieces])
        result = ''.join(
            entry if isinstance(entry, str) else ''.join(spoken[index] + sep for index, sep in entry)
            for entry in layout
        )

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

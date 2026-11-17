"""Word and character timestamps for /predict (``include_timestamps``; English only).

Alignment. Each text piece the model spoke (after verbalization, ``sanitize_text_for_tts`` and the split) is
force-aligned against its own CB0 codes with the CB0-CTC recogniser, the same machinery as the quality gate
(app/quality_gate.py): ``CB0CTC.forward_fp32`` (fp32 log-probs, 50 Hz) and ``cb0_ctc.ctc_viterbi`` on
``encode_text(normalize_text(piece))``. ``normalize_with_map`` maps every aligned label back to the character of the
piece it came from (a spelled-out number's labels point at the number's first character).

Timeline. The pieces' codes are contiguous in the decoded audio (``TTSEngine.generate(return_piece_frames=True)``):
CB0 frame k starts at k / 12.5 s, CTC output frame t of a piece starting at frame k0 at (4 k0 + t) * 20 ms.
``Timeline`` then maps those times through every later step that moves audio: the edge trim
(``trim_silence_padding_ms``), the pause cap's cuts (the ones ``clip_long_pauses`` actually applied, reported through
``cuts_out``), and the final duration. Resampling to 48 kHz does not move audio in seconds. The prefix of a
continuation is not in the codes or the audio, so it gets no timings.

Output (``assemble``):
  words       whitespace tokens of the joined text with at least one letter; start / end = earliest start / latest
              end of the aligned labels inside the token (letters, and the spelled-out labels of a number inside
              it, as in "21st"), the end extended over the following frames where the recogniser's VAD head gives
              speech p > 0.8, up to the next letter's first frame (``_word_end_frame``); confidence = exp(mean
              log-prob of those labels over their frames on the Viterbi path), in [0, 1]. A word with no aligned label (its piece did not align, or nothing in it is
              alignable) has start = end = None and confidence 0.
  characters  one entry per letter / apostrophe of the joined text that an aligned label covers. A character's
              start is its first emission frame; its end is the next covered character's start inside the same
              word (so the characters of a word tile it), the word's end for its last one.

Accuracy (measured on 6 local int4 generations, 15 word edges next to a pause, against the 10 ms energy envelope):
word starts after a pause median +0 ms (mean |error| 9 ms, max 50 ms); word ends before a pause median +50 ms late
(mean |error| 54 ms, max 80 ms) with the VAD extension, median 140 ms early without it (the last letter is a short
CTC spike). Inside continuous speech a boundary is where the next word's first letter is emitted (20 ms frames).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from app import cb0_ctc as C
from app.logging_utils import get_logger

logger = get_logger("timestamps")

CODE_HZ = 12.5                       # CB0 frames per second
CTC_UPSAMPLE = 4                     # CB0-CTC output frames per CB0 frame
FRAME_S = 1.0 / (CODE_HZ * CTC_UPSAMPLE)   # 20 ms per CTC output frame
APOSTROPHES = frozenset("'’‘ʼ`´")
VAD_EXTEND = 0.8                     # word ends extend over frames with VAD speech probability above this
_TOKEN_RE = re.compile(r"\S+")


@dataclass
class PieceAlignment:
    """One piece's forced alignment. ``labels[k]`` (CTC label id) came from ``piece text[src[k]]``; ``spans[k]`` =
    (first, last + 1) CTC output frame of its emission, relative to the piece start; ``lp_sum[k]`` = sum of its
    log-probs on those frames. ``frame_offset``: CB0 frame where the piece starts in the decoded audio."""

    text: str
    frame_offset: int = 0
    ok: bool = False
    labels: list = field(default_factory=list)
    src: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    spans: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int64))
    lp_sum: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vad: np.ndarray | None = None      # the recogniser's speech probability per output frame (None: no VAD head)
    n_frames: int = 0                   # CTC output frames of the piece


def align_pieces(recognizer, semantic_codes, piece_frames, pieces, batch_size: int = 16) -> list[PieceAlignment]:
    """Forced alignment of each text piece against its own CB0 codes (fp32, batched by length).

    ``piece_frames[i] = (start, end)``: piece i's CB0 frames in ``semantic_codes``. Pieces without frames (after a CB0
    early stop) or without alignable text come back with ``ok=False``."""
    import torch

    out = []
    for i, text in enumerate(pieces):
        start = int(piece_frames[i][0]) if i < len(piece_frames) else 0
        out.append(PieceAlignment(text=text, frame_offset=start))
    jobs = []
    for i, text in enumerate(pieces):
        if i >= len(piece_frames):
            continue
        start, end = (int(v) for v in piece_frames[i])
        norm, src = C.normalize_with_map(text)
        labels = C.encode_text(norm)
        if end <= start or not labels or len(labels) != len(src):
            continue
        out[i].labels, out[i].src = labels, src
        jobs.append((i, list(semantic_codes[start:end])))
    jobs.sort(key=lambda job: -len(job[1]))
    for b in range(0, len(jobs), batch_size):
        batch = jobs[b:b + batch_size]
        lp, ol, vad = recognizer.forward_fp32([codes for _, codes in batch])
        results = C.ctc_viterbi(lp, ol, [out[i].labels for i, _ in batch])
        lp_np = lp.float().cpu().numpy()
        ol_np = ol.cpu().numpy()
        vad_np = vad.float().cpu().numpy() if vad is not None else None
        for row, ((i, _), (spans, _score, ok)) in enumerate(zip(batch, results)):
            piece = out[i]
            if not ok:
                continue
            spans = np.asarray(spans, dtype=np.int64)
            lab = np.asarray(piece.labels, dtype=np.int64)
            # sum of each label's log-prob over its frames on the path (cumulative sums per label column)
            cum = np.concatenate([np.zeros((1, lp_np.shape[2])), np.cumsum(lp_np[row], axis=0)], axis=0)
            piece.lp_sum = cum[spans[:, 1], lab] - cum[spans[:, 0], lab]
            piece.spans = spans
            piece.n_frames = int(ol_np[row])
            piece.vad = vad_np[row, :piece.n_frames] if vad_np is not None else None
            piece.ok = True
    return out


def align_pieces_safe(recognizer, semantic_codes, piece_frames, pieces) -> list[PieceAlignment]:
    """``align_pieces``, never raising: on any error every piece is unaligned (its words get null times)."""
    try:
        return align_pieces(recognizer, semantic_codes, piece_frames, pieces)
    except Exception:  # noqa: BLE001
        logger.exception("Timestamp alignment failed; returning words without times")
        return [PieceAlignment(text=text) for text in pieces]


class Timeline:
    """Maps a time on the decoded (pre-post-processing) audio to the returned audio, in seconds.

    Steps, in order: ``trim(start, end, rate)`` (the edge trim kept samples [start, end)), ``cut(cuts, rate)`` (the
    pause cap's (left, right, width) cuts on the trimmed audio, as applied by ``clip_long_pauses``), and
    ``finish(duration_s)`` (the returned audio's length). A cut keeps [0, left), crossfades [left, left + width) into
    [right, right + width) and continues there: it removes right - left samples, and a time in the removed span
    [left, right) maps to the cut point ``left``. The crossfade window belongs to the incoming side ([right, right +
    width) -> [left, left + width)), so times at or after ``right`` shift by exactly the removed length. Times are
    clamped to the kept audio (a time inside trimmed-off audio maps to the nearest edge).
    """

    def __init__(self):
        self.lead_s = 0.0
        self.kept_s: float | None = None
        self.cuts_s: list[tuple[float, float]] = []
        self.duration_s: float | None = None

    def trim(self, start: int, end: int, sample_rate: int) -> "Timeline":
        self.lead_s = start / sample_rate
        self.kept_s = (end - start) / sample_rate
        return self

    def cut(self, cuts, sample_rate: int) -> "Timeline":
        self.cuts_s = sorted((left / sample_rate, right / sample_rate) for left, right, _width in cuts)
        return self

    def finish(self, duration_s: float) -> "Timeline":
        self.duration_s = duration_s
        return self

    def map(self, t: float) -> float:
        t = max(0.0, t - self.lead_s)
        if self.kept_s is not None:
            t = min(t, self.kept_s)
        removed = 0.0
        for left, right in self.cuts_s:
            if t >= right:
                removed += right - left
            elif t > left:
                t = left
                break
            else:
                break
        t -= removed
        if self.duration_s is not None:
            t = min(t, self.duration_s)
        return max(0.0, t)


def _ms(t: float) -> float:
    return round(t, 3)


def _word_end_frame(piece: PieceAlignment, last_label: int) -> int:
    """End frame of a word whose last aligned label is ``last_label``: the end of that label's emission, extended
    over the following frames while the recogniser's VAD head says speech (p > VAD_EXTEND), up to the next
    letter's first frame (or the piece end). CTC emits a word's last letter as a short spike, typically ~140 ms
    before the sound ends; the VAD extension ends the word ~50 ms after it instead (measured, see the module doc)."""
    end = int(piece.spans[last_label][1])
    if piece.vad is None:
        return end
    limit = piece.n_frames
    for k in range(last_label + 1, len(piece.labels)):
        if piece.labels[k] != C.SPACE_ID:
            limit = int(piece.spans[k][0])
            break
    while end < limit and piece.vad[end] > VAD_EXTEND:
        end += 1
    return end


def assemble(alignments: list[PieceAlignment], timeline: Timeline) -> dict:
    """The response's ``timestamps`` object from the pieces' alignments (pieces joined with one space)."""
    text = " ".join(a.text for a in alignments)
    # per character of `text` covered by an aligned label of an ok piece: (piece, first frame, last frame + 1, last
    # label index), absolute CTC frames on the decoded audio, and the label log-prob mass [sum, frames]
    cover: dict[int, list] = {}
    char_lp: dict[int, list] = {}
    offset = 0
    for p, a in enumerate(alignments):
        if a.ok:
            base = a.frame_offset * CTC_UPSAMPLE
            for k, lab in enumerate(a.labels):
                if lab == C.SPACE_ID:
                    continue
                idx = offset + int(a.src[k])
                f0, f1 = base + int(a.spans[k][0]), base + int(a.spans[k][1])
                c = cover.get(idx)
                cover[idx] = [p, f0, f1, k] if c is None else [p, min(c[1], f0), max(c[2], f1), max(c[3], k)]
                acc = char_lp.setdefault(idx, [0.0, 0])
                acc[0] += float(a.lp_sum[k])
                acc[1] += int(a.spans[k][1]) - int(a.spans[k][0])
        offset += len(a.text) + 1

    def sec(frame: int) -> float:
        return _ms(timeline.map(frame * FRAME_S))

    words, characters = [], []
    for m in _TOKEN_RE.finditer(text):
        token = m.group(0)
        if not any(ch.isalpha() for ch in token):
            continue
        lo, hi = m.start(), m.end()
        covered = [i for i in range(lo, hi) if i in cover]
        if not covered:
            words.append({"text": token, "start": None, "end": None, "char_start": lo, "char_end": hi,
                          "confidence": 0.0})
            continue
        piece = alignments[cover[covered[0]][0]]
        w_start = min(cover[i][1] for i in covered)
        last = max(covered, key=lambda i: cover[i][3])
        w_end = max(max(cover[i][2] for i in covered),
                    piece.frame_offset * CTC_UPSAMPLE + _word_end_frame(piece, cover[last][3]))
        lp_total = sum(char_lp[i][0] for i in covered)
        n_frames = sum(char_lp[i][1] for i in covered)
        confidence = math.exp(lp_total / n_frames) if n_frames else 0.0
        words.append({"text": token, "start": sec(w_start), "end": sec(w_end), "char_start": lo, "char_end": hi,
                      "confidence": round(min(1.0, max(0.0, confidence)), 3)})
        letters = [i for i in covered if text[i].isalpha() or text[i] in APOSTROPHES]
        for n, i in enumerate(letters):
            start = cover[i][1]
            end = max(cover[i][2], cover[letters[n + 1]][1]) if n + 1 < len(letters) else w_end
            characters.append({"char": text[i], "index": i, "start": sec(start), "end": sec(end)})
    return {"text": text, "words": words, "characters": characters}

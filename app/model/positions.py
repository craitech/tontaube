"""TontaubeV1 serialization constants and logical position scheme.

Shared by the vLLM model adapter and the lite backend; importing this module
does not require vLLM.

Input layout for a target_codebook=K model (no SOS):
  - [cb0 prompt][cb1 prompt]...[cbK prompt]: each block at positions 1..P
  - [text with text markers]: from P+1 via the aligned marker recurrence
    (marker_pos = cur + max(n_text + chunk_realign_offset, n_audio))
  - [PAD][cb0 audio (+markers) (+EOS)]: PAD at P, audio from P+1
  - ([PAD][cb_i audio]) x K: audio aligned to cb0 non-marker positions

Since RoPE only depends on relative position differences, all complex
position calculations are done during prefill and shifted so that the
last raw prefill position = seq_len - 1. A fixed coordinate translation is
then applied to every prefill and decode position.
"""
from __future__ import annotations

from dataclasses import dataclass

SEMANTIC_VOCAB_SIZE = 16384
ACOUSTIC_VOCAB_SIZE = 4096

AUDIO_MARKER_LOCAL_IDX = SEMANTIC_VOCAB_SIZE + 1


def get_codebook_vocab_size(codebook_idx: int) -> int:
    if codebook_idx == 0:
        return SEMANTIC_VOCAB_SIZE + 2  # semantic + EOS + audio_marker
    return ACOUSTIC_VOCAB_SIZE


def get_codebook_offset(codebook_idx: int) -> int:
    if codebook_idx == 0:
        return 0
    return SEMANTIC_VOCAB_SIZE + (codebook_idx - 1) * ACOUSTIC_VOCAB_SIZE


@dataclass(frozen=True)
class LayoutIds:
    """Token IDs that drive the position scheme (from the model config)."""

    dc_base_id: int
    pad_token_id: int
    text_marker_id: int
    audio_marker_id: int
    eos_id: int
    chunk_realign_offset: int
    target_codebook: int


def front_prompt_positions(ids: list[int], layout: LayoutIds) -> tuple[list[int], int]:
    """Positions for the TontaubeV1 front-prompt scheme.

    Aligned marker recurrence:
        marker_pos = cur + max(n_text + chunk_realign_offset, n_audio)
    Audio lengths of not-yet-generated segments count as 0, so their text
    markers land at cur + n_text + offset. The realignment offset reserves
    the corresponding position range.

    Returns (positions, gap): gap > 0 when the next decoded token must skip
    a marker position (acoustic rows are aligned to cb0 NON-marker positions).
    """
    n = len(ids)
    dc = layout.dc_base_id
    pad = layout.pad_token_id
    tm = layout.text_marker_id
    am = layout.audio_marker_id
    eos = layout.eos_id
    off = layout.chunk_realign_offset

    def section_of(t):
        d = t - dc
        if d < SEMANTIC_VOCAB_SIZE:
            return 0
        return 1 + (d - SEMANTIC_VOCAB_SIZE) // ACOUSTIC_VOCAB_SIZE

    positions = [0] * n
    i = 0

    # 1) Front prompt blocks: leading dc-range tokens; every codebook section
    #    restarts at position 1. P = length of the cb0 block.
    prompt_len = 0
    cur_sec, k = None, 0
    while i < n and ids[i] >= dc:
        sec = section_of(ids[i])
        if sec != cur_sec:
            cur_sec, k = sec, 0
        k += 1
        positions[i] = k
        if sec == 0:
            prompt_len = k
        i += 1

    content_base = 1 + prompt_len
    pad_pos = content_base - 1

    # 2) Text region: up to the first PAD.
    text_start = i
    while i < n and ids[i] != pad:
        i += 1
    text_end = i  # index of the cb0-row PAD (or n)

    # 3) cb0 audio row: PAD + audio(+markers), ends at EOS / next PAD / n.
    cb0_audio_start = min(i + 1, n)
    j = cb0_audio_start
    while j < n and ids[j] != pad and ids[j] != eos:
        j += 1
    cb0_audio_end = j
    eos_idx = j if (j < n and ids[j] == eos) else None

    # Segment lengths (markers excluded), leading-aligned pairing.
    text_seg_lens, n_text_markers = [], 0
    seg_len = 0
    for x in range(text_start, text_end):
        if ids[x] == tm:
            text_seg_lens.append(seg_len)
            n_text_markers += 1
            seg_len = 0
        else:
            seg_len += 1
    text_seg_lens.append(seg_len)

    audio_seg_lens, n_audio_markers = [], 0
    seg_len = 0
    for x in range(cb0_audio_start, cb0_audio_end):
        if ids[x] == am:
            audio_seg_lens.append(seg_len)
            n_audio_markers += 1
            seg_len = 0
        else:
            seg_len += 1
    audio_seg_lens.append(seg_len)

    # 4) Marker recurrence -> per-segment start positions + marker positions.
    n_bounds = max(n_text_markers, n_audio_markers)
    seg_starts, marker_positions = [], []
    cur = content_base
    for si in range(n_bounds + 1):
        seg_starts.append(cur)
        if si < n_bounds:
            n_t = text_seg_lens[si] if si < len(text_seg_lens) else 0
            n_a = audio_seg_lens[si] if si < len(audio_seg_lens) else 0
            marker_pos = cur + max(n_t + off, n_a)
            marker_positions.append(marker_pos)
            cur = marker_pos + 1

    # Text positions.
    seg_idx, in_seg = 0, 0
    for x in range(text_start, text_end):
        if ids[x] == tm:
            positions[x] = marker_positions[seg_idx]
            seg_idx += 1
            in_seg = 0
        else:
            positions[x] = seg_starts[seg_idx] + in_seg
            in_seg += 1

    # cb0 row positions: PAD, audio (+markers), EOS.
    cb0_nonmarker_pos = []
    if text_end < n:
        positions[text_end] = pad_pos
    seg_idx, in_seg = 0, 0
    last_audio_pos = pad_pos
    for x in range(cb0_audio_start, cb0_audio_end):
        if ids[x] == am:
            positions[x] = marker_positions[seg_idx]
            last_audio_pos = positions[x]
            seg_idx += 1
            in_seg = 0
        else:
            positions[x] = seg_starts[seg_idx] + in_seg
            cb0_nonmarker_pos.append(positions[x])
            last_audio_pos = positions[x]
            in_seg += 1
    if eos_idx is not None:
        positions[eos_idx] = last_audio_pos + 1

    # 5) Codebook rows after the cb0 row: [PAD][audio at cb0 non-marker positions].
    i = eos_idx + 1 if eos_idx is not None else cb0_audio_end
    n_last_row_audio = len(cb0_nonmarker_pos) if eos_idx is None else 0
    while i < n:
        if ids[i] == pad:
            positions[i] = pad_pos
            n_last_row_audio = 0
        else:
            if n_last_row_audio < len(cb0_nonmarker_pos):
                positions[i] = cb0_nonmarker_pos[n_last_row_audio]
            else:
                positions[i] = positions[i - 1] + 1
            n_last_row_audio += 1
        i += 1

    # 6) Decode-continuation gap. cb0 decode continues linearly (PAD -> P+1,
    # marker -> marker_pos+1, audio -> +1): gap 0. Acoustic rows continue at
    # the next cb0 non-marker position, which may skip a marker slot.
    gap = 0
    if layout.target_codebook > 0 and cb0_nonmarker_pos:
        if n_last_row_audio < len(cb0_nonmarker_pos):
            gap = cb0_nonmarker_pos[n_last_row_audio] - positions[-1] - 1

    return positions, gap


def prefill_positions(ids: list[int], layout: LayoutIds) -> list[int]:
    """Raw prefill positions shifted so decoding continues at index len(ids).

    The chunk-realignment translation is not included; callers add it to
    prefill and decode positions alike.
    """
    positions, gap = front_prompt_positions(ids, layout)
    shift = (len(ids) - 1 - gap) - positions[-1]
    return [p + shift for p in positions]

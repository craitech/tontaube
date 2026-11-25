"""What the GPU and CPU engines share around a request: the weights of the release, the start at the first sound,
and the whole utterance from a stream of pieces."""
import json
from pathlib import Path

import numpy as np

REPO, REVISION = "TontaubeAI/TontaubeV1.1-Flash", "v1.1.0"      # the weights on Hugging Face
SR, HOP = 24000, 3200
LEAD_T, LEAD_PAD, LEAD_WAIT = 4 / 32768, 120, 2 * HOP   # start at the first sound: -78 dBFS, 5 ms before it, wait 2 frames
OUT_GAIN = np.float32(0.84139514)  # -1.5 dB on what the engine hands out (the decoder's peaks reach about 1.1)


def release(weights=None):
    """(config, get): the release's config.json as a dict and get(name) -> local path of one of its files. weights: a
    local folder laid out as the release (or its gpu/ or web/ folder), or None: the release on Hugging Face, each file
    downloaded once into the Hugging Face cache. config.json is always asked for first (it counts the download)."""
    if weights is None:
        from huggingface_hub import hf_hub_download
        get = lambda name: Path(hf_hub_download(REPO, name, revision=REVISION))
    else:
        root = Path(weights)
        if not (root / "config.json").exists() and (root.parent / "config.json").exists():
            root = root.parent                          # the release's gpu/ or web/ folder
        get = lambda name: root / name
    cfg_path = get("config.json")
    return (json.loads(cfg_path.read_text()) if cfg_path.exists() else {}), get


def cut_lead(pieces, on_cut=None):
    """(start, audio) pieces in utterance samples -> the same in player samples, the request starting at its first
    sound: s = the first sample with |x| > 4 / 32768 (-78 dBFS), cut point c = max(0, s - 120) (5 ms before it), no
    fade; nothing goes out until s is found or 2 frames are there (then c = 0). Player sample p is utterance sample
    p + c; on_cut(c) is called once c is known. Only the start of a request is cut, not chunk starts or pauses."""
    cut = LeadCut(on_cut)
    for start, audio in pieces:
        yield from cut.push(start, audio)
    yield from cut.finish()


class LeadCut:
    """cut_lead one piece at a time: push(start, audio) and finish() return the pieces that go out."""

    def __init__(self, on_cut=None):
        self.buf, self.c, self.on_cut = np.zeros(0, np.float32), None, on_cut

    def push(self, start, audio):
        c = self.c
        if c is None:
            self.buf = np.concatenate([self.buf[:start], audio])
            loud = np.flatnonzero(np.abs(self.buf) > LEAD_T)
            if not len(loud) and len(self.buf) < LEAD_WAIT:
                return []
            c = self.c = max(0, int(loud[0]) - LEAD_PAD) if len(loud) else 0
            if self.on_cut is not None:
                self.on_cut(c)
            start, audio = c, self.buf[c:]
        elif start < c:                                 # not with the redraw (it never touches frame 0)
            audio, start = audio[c - start:], c
        return [(start - c, audio)]

    def finish(self):
        if self.c is None and len(self.buf):            # shorter than the wait and silent: nothing cut
            self.c = 0
            if self.on_cut is not None:
                self.on_cut(0)
            return [(0, self.buf)]
        return []


def assemble(pieces) -> np.ndarray:
    """The whole utterance as the player gets it: every piece written from its start on, a start before the end
    replacing what came after it."""
    out, n = np.zeros(HOP * 64, np.float32), 0
    for start, audio in pieces:
        if start + len(audio) > len(out):
            out = np.concatenate([out[:n], np.zeros(max(len(out), start + len(audio)), np.float32)])
        out[start:start + len(audio)] = audio
        n = start + len(audio)
    return out[:n].copy()


class Speaker:
    """The request API both engines share; an engine provides stream() and a front end (self.front)."""

    def chunks(self, text, verbalize=True, first_max=0):
        return self.front.chunks(text, verbalize, first_max)

    def speak(self, text, seed=0, verbalize=True, chunks=None, first_max=0, redraw=None, played=None, cut=None,
              boundary=None) -> np.ndarray:
        """The whole utterance as the player gets it (stream() with its replacements applied)."""
        return assemble(self.stream(text, seed, verbalize, chunks, first_max, redraw, played, cut, boundary))

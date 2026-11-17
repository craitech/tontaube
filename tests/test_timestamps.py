"""/predict word and character timestamps (app/timestamps.py, include_timestamps).

The timeline remap through the edge trim and the pause cap (synthetic and the real clip_long_pauses cuts), the word /
character assembly from alignment spans, the forced alignment on a scripted recogniser, TTSEngine.generate's piece
frames, the request validation, and one /predict run end to end (fake recogniser, real trim, real pause cap,
identity enhancer).
"""
import math
from threading import Lock
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest
import torch

import app.audio_pauses as ap
import app.main as app_main
import app.mossformer2_postprocessor as mossformer_module
from app import cb0_ctc as C
from app import timestamps as ts
from app.config import ServerConfig
from app.inference import TTSEngine
from app.main import TTSRequest, TTSRequestOptionError, create_app

SR = 24000
FRAME = 1920          # samples per CB0 frame at 24 kHz


# ---------------------------------------------------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------------------------------------------------
def test_timeline_identity_without_steps():
    tl = ts.Timeline()
    assert tl.map(0.0) == 0.0 and tl.map(1.234) == pytest.approx(1.234)


def test_timeline_trim_shifts_and_clamps():
    tl = ts.Timeline().trim(int(0.5 * SR), int(3.5 * SR), SR).finish(3.0)
    assert tl.map(0.5) == pytest.approx(0.0)
    assert tl.map(1.75) == pytest.approx(1.25)
    assert tl.map(0.2) == 0.0                 # inside the trimmed lead: the first sample
    assert tl.map(9.0) == pytest.approx(3.0)  # inside the trimmed tail: the last sample


def test_timeline_synthetic_cuts():
    # two cuts on the trimmed timeline: [2, 5) and [8, 10) seconds removed
    tl = ts.Timeline().cut([(2 * SR, 5 * SR, 480), (8 * SR, 10 * SR, 480)], SR)
    assert tl.map(1.0) == pytest.approx(1.0)
    assert tl.map(2.0) == pytest.approx(2.0)
    assert tl.map(3.5) == pytest.approx(2.0)   # inside a removed span: the cut point
    assert tl.map(5.0) == pytest.approx(2.0)   # the cut's right edge joins the left edge
    assert tl.map(6.0) == pytest.approx(3.0)
    assert tl.map(9.0) == pytest.approx(5.0)   # in the second removed span: its cut point (8 - 3)
    assert tl.map(11.0) == pytest.approx(6.0)  # 11 - 3 - 2
    # unsorted input is sorted; monotone
    tl2 = ts.Timeline().cut([(8 * SR, 10 * SR, 480), (2 * SR, 5 * SR, 480)], SR)
    xs = np.linspace(0, 12, 241)
    ys = [tl2.map(x) for x in xs]
    assert ys == sorted(ys) and ys == [tl.map(x) for x in xs]


def test_timeline_trim_then_cuts_compose():
    tl = ts.Timeline().trim(SR, 11 * SR, SR).cut([(2 * SR, 5 * SR, 480)], SR).finish(7.0)
    assert tl.map(1.0) == 0.0
    assert tl.map(3.0) == pytest.approx(2.0)   # trimmed time 2 = cut point
    assert tl.map(4.5) == pytest.approx(2.0)
    assert tl.map(7.0) == pytest.approx(3.0)   # trimmed 6 -> 6 - 3
    assert tl.map(20.0) == pytest.approx(7.0)


def _bursts(spec, rate=SR, seed=0):
    """Audio of noise bursts and digital silence: spec = [(seconds, loud?)]."""
    rng = np.random.default_rng(seed)
    parts = [(rng.uniform(-0.3, 0.3, round(s * rate)) if loud else np.zeros(round(s * rate))) for s, loud in spec]
    return np.concatenate(parts).astype(np.float32)


def test_timeline_matches_the_real_pause_cap():
    """Marked samples after real clip_long_pauses cuts land where the Timeline says."""
    audio = _bursts([(0.5, True), (4.0, False), (0.5, True), (3.1, False), (0.4, True), (1.0, False), (0.3, True)])
    marks = [100, int(0.7 * SR), int(4.6 * SR), int(5.2 * SR), int(8.2 * SR), len(audio) - 50]
    for k, m in enumerate(marks):
        audio[m] = 0.9 - 0.01 * k             # unique values
    cuts = []
    out, removed, n = ap.clip_long_pauses(audio, SR, cuts_out=cuts)
    assert n == len(cuts) == 2 and removed == sum(r - l for l, r, _ in cuts)
    tl = ts.Timeline().cut(cuts, SR).finish(len(out) / SR)
    for k, m in enumerate(marks):
        if any(l <= m < r for l, r, _ in cuts):
            continue
        hits = np.flatnonzero(np.isclose(out, 0.9 - 0.01 * k))
        assert len(hits) == 1, k
        assert round(tl.map(m / SR) * SR) == hits[0]
    # a time inside a removed span maps to its cut point
    left, right, _ = cuts[0]
    assert tl.map((left + right) / 2 / SR) == pytest.approx(left / SR)


def test_clip_long_pauses_reports_no_cuts_by_default_and_same_audio():
    audio = _bursts([(0.5, True), (4.0, False), (0.5, True)])
    plain = ap.clip_long_pauses(audio, SR)
    cuts = []
    reported = ap.clip_long_pauses(audio, SR, cuts_out=cuts)
    assert np.array_equal(plain[0], reported[0]) and plain[1:] == reported[1:]
    assert cuts == ap.find_pause_cuts(audio, SR)


# ---------------------------------------------------------------------------------------------------------------------
# assembly from spans
# ---------------------------------------------------------------------------------------------------------------------
def _fake_alignment(text, frame_offset=0, ok=True, start=10, step=3, lp=math.log(0.8)):
    """Each label emitted for one frame, every `step` frames from `start`, with log-prob `lp`."""
    norm, src = C.normalize_with_map(text)
    labels = C.encode_text(norm)
    n = len(labels)
    spans = np.stack([start + step * np.arange(n), start + step * np.arange(n) + 1], 1)
    return ts.PieceAlignment(text=text, frame_offset=frame_offset, ok=ok, labels=labels, src=src,
                             spans=spans if ok else np.zeros((0, 2), np.int64),
                             lp_sum=np.full(n, lp) if ok else np.zeros(0))


def test_assemble_words_and_characters_with_punctuation_and_apostrophes():
    a = _fake_alignment("Hello, it's 21st St. & “Bob”!")
    out = ts.assemble([a], ts.Timeline())
    text = out["text"]
    assert text == "Hello, it's 21st St. & “Bob”!"
    words = out["words"]
    assert [w["text"] for w in words] == ["Hello,", "it's", "21st", "St.", "“Bob”!"]   # "&" has no letter
    assert [(w["char_start"], w["char_end"]) for w in words] == [(0, 6), (7, 11), (12, 16), (17, 20), (23, 29)]
    for w in words:
        assert text[w["char_start"]:w["char_end"]] == w["text"]
        assert w["start"] is not None and w["start"] < w["end"]
        assert w["confidence"] == pytest.approx(0.8)
    # label k is emitted at frame 10 + 3k: "hello" = labels 0..4 -> 0.200 .. (22 + 1) * 0.02
    assert words[0]["start"] == pytest.approx(0.2) and words[0]["end"] == pytest.approx(0.46)
    chars = out["characters"]
    # letters and the apostrophe only; no digits, punctuation, spaces or "&"; original case and characters
    assert "".join(c["char"] for c in chars) == "Helloit'sStBob"
    assert all(text[c["index"]] == c["char"] for c in chars)
    # characters tile their word: each ends where the next letter of the word starts, the last at the word end
    hello = [c for c in chars if c["index"] < 6]
    assert [c["start"] for c in hello] == pytest.approx([0.2, 0.26, 0.32, 0.38, 0.44])
    assert [c["end"] for c in hello[:-1]] == [c["start"] for c in hello[1:]]
    assert hello[-1]["end"] == words[0]["end"]
    # "21st" is spoken as "twenty first": the word gets the number's label times, but no character entries
    first_label_of_21 = len("hello it's ")
    assert words[2]["start"] == pytest.approx((10 + 3 * first_label_of_21) * 0.02)
    # monotone over the text
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)


def test_assemble_unaligned_piece_keeps_words_with_null_times():
    a = _fake_alignment("First piece here.", frame_offset=0)
    b = _fake_alignment("Second one, failed.", frame_offset=100, ok=False)
    c = _fake_alignment("Third.", frame_offset=200)
    out = ts.assemble([a, b, c], ts.Timeline())
    assert out["text"] == "First piece here. Second one, failed. Third."
    by_text = {w["text"]: w for w in out["words"]}
    for name in ("Second", "one,", "failed."):
        assert by_text[name]["start"] is None and by_text[name]["end"] is None
        assert by_text[name]["confidence"] == 0.0
    assert by_text["Third."]["char_start"] == out["text"].index("Third.")
    # the third piece's times are offset by its frame offset (200 CB0 frames = 16 s)
    assert by_text["Third."]["start"] == pytest.approx(16.0 + 10 * 0.02)
    assert not any(18 <= ch["index"] < 37 for ch in out["characters"])     # nothing for the failed piece


def test_assemble_maps_through_timeline_and_rounds_to_ms():
    a = _fake_alignment("Go now.", frame_offset=0, start=50, step=7)       # labels at 1.0, 1.14, ...
    tl = ts.Timeline().trim(int(0.3 * SR), 10 * SR, SR).finish(5.0)
    out = ts.assemble([a], tl)
    go = out["words"][0]
    assert go["start"] == pytest.approx(0.7) and go["end"] == pytest.approx(1.0 + 0.14 + 0.02 - 0.3)
    for w in out["words"]:
        for key in ("start", "end"):
            assert w[key] == round(w[key], 3)


def test_assemble_confidence_is_exp_mean_frame_log_prob():
    a = _fake_alignment("ab cd")
    a.lp_sum = np.array([math.log(0.9), math.log(0.1), 0.0, math.log(0.5), math.log(0.5)])
    a.spans = np.array([[0, 1], [1, 2], [2, 3], [3, 5], [5, 6]])               # "c" covers two frames
    words = ts.assemble([a], ts.Timeline())["words"]
    assert words[0]["confidence"] == pytest.approx(round(math.exp((math.log(0.9) + math.log(0.1)) / 2), 3))
    assert words[1]["confidence"] == pytest.approx(round(math.exp(2 * math.log(0.5) / 3), 3))


def test_assemble_extends_word_ends_over_vad_speech_up_to_the_next_letter():
    a = _fake_alignment("ab cd", start=10, step=3)        # labels a=10, b=13, " "=16, c=19, d=22 (one frame each)
    a.n_frames = 40
    a.vad = np.zeros(40)
    a.vad[14:30] = 0.95                                     # speech after "b" and after "d" until frame 30
    words = ts.assemble([a], ts.Timeline())["words"]
    assert words[0]["end"] == pytest.approx(19 * 0.02)      # stops at the next letter ("c" at frame 19)
    assert words[1]["end"] == pytest.approx(30 * 0.02)      # stops where the VAD drops
    assert words[1]["start"] == pytest.approx(19 * 0.02)    # starts are not moved
    chars = ts.assemble([a], ts.Timeline())["characters"]
    assert chars[1]["end"] == words[0]["end"] and chars[3]["end"] == words[1]["end"]
    a.vad[:] = 0.5                                          # at or below the 0.8 threshold: no extension
    words = ts.assemble([a], ts.Timeline())["words"]
    assert words[1]["end"] == pytest.approx(23 * 0.02)


def test_assemble_word_with_letters_but_nothing_alignable():
    a = _fake_alignment("Hi 42.")
    out = ts.assemble([a], ts.Timeline())
    assert [w["text"] for w in out["words"]] == ["Hi"]
    a2 = _fake_alignment("Ok αβγ.")
    out2 = ts.assemble([a2], ts.Timeline())
    words = {w["text"]: w for w in out2["words"]}
    assert words["αβγ."]["start"] is None and words["αβγ."]["confidence"] == 0.0


# ---------------------------------------------------------------------------------------------------------------------
# forced alignment on a scripted recogniser
# ---------------------------------------------------------------------------------------------------------------------
class ScriptedRecognizer:
    """forward_fp32: CB0 code 0 = silence (blank), any other code = speech (every label equally likely, blank
    impossible). The Viterbi path must then cover exactly the speech frames with the labels."""

    def __init__(self):
        self.calls = []

    def forward_fp32(self, codes_list):
        self.calls.append([list(c) for c in codes_list])
        T = max(len(c) for c in codes_list) * 4
        lp = torch.full((len(codes_list), T, C.N_OUT), -20.0)
        for b, codes in enumerate(codes_list):
            speech = np.repeat(np.asarray(codes) != 0, 4)
            for t in range(len(speech)):
                if speech[t]:
                    lp[b, t, 1:] = math.log(1.0 / (C.N_OUT - 1))
                else:
                    lp[b, t, 0] = 0.0
        return lp, torch.tensor([4 * len(c) for c in codes_list]), None


def test_align_pieces_puts_labels_on_speech_frames_and_flags_bad_pieces():
    codes = [0] * 3 + [5] * 10 + [0] * 7 + [0] * 2 + [5] * 6 + [0] * 2 + [5] * 1
    frames = [(0, 20), (20, 30), (30, 31)]
    rec = ScriptedRecognizer()
    out = ts.align_pieces(rec, codes, frames, ["Hello there.", "Hi.", "Much too long for one frame.", "Lost."])
    assert [a.ok for a in out] == [True, True, False, False]
    assert out[1].frame_offset == 20 and out[3].frame_offset == 0
    for a, (lo, hi) in zip(out[:2], [(12, 52), (8, 32)]):
        assert len(a.spans) == len(a.labels)
        assert a.spans[:, 0].min() >= lo and a.spans[:, 1].max() <= hi       # inside the speech frames
        assert np.all(np.diff(a.spans[:, 0]) > 0)
        assert np.all(a.lp_sum < 0)
    # one batch, longest first
    assert len(rec.calls) == 1 and [len(c) for c in rec.calls[0]] == [20, 10, 1]
    words = ts.assemble(out, ts.Timeline())["words"]
    assert words[0]["start"] >= 3 * 0.08 and words[1]["end"] <= 13 * 0.08
    assert words[2]["start"] >= (20 + 2) * 0.08 and words[2]["end"] <= (20 + 8) * 0.08
    assert all(w["start"] is None for w in words[3:])


def test_align_pieces_safe_never_raises():
    class Broken:
        def forward_fp32(self, codes_list):
            raise RuntimeError("boom")

    out = ts.align_pieces_safe(Broken(), [1] * 10, [(0, 10)], ["Hello."])
    assert len(out) == 1 and not out[0].ok
    assert ts.assemble(out, ts.Timeline())["words"][0]["start"] is None


# ---------------------------------------------------------------------------------------------------------------------
# TTSEngine.generate: piece frames
# ---------------------------------------------------------------------------------------------------------------------
def _engine(semantic, markers, prefix=None):
    config = ServerConfig()
    config.trailing_silence_tokens = 3
    engine = TTSEngine(config)
    engine.vv_tokenizer = object()
    engine.dc_inference = object()
    engine._resolve_params = MagicMock(return_value=(dict(temperature=0.8, top_p=0.95, top_k=50, seed=1),
                                                     (None, None), "", 0.0, -1))
    engine._resolve_prefix = AsyncMock(return_value=(prefix, {}))
    engine._n_codebooks = MagicMock(return_value=4)
    engine._generate_semantic = AsyncMock(side_effect=lambda *a, **k: (list(semantic), list(markers)))
    engine._generate_acoustic = AsyncMock(side_effect=lambda *a, **k: [2] * len(semantic))
    return engine


@pytest.mark.asyncio
async def test_generate_returns_piece_frames_only_when_asked():
    engine = _engine([1] * 50, [20, 35])
    with patch("app.inference.free_memory"), patch(
            "app.inference.decode_chunked_with_context", return_value=_codes_audio([5] * 53)):
        two = await engine.generate(text=["a.", "b.", "c."])
        three = await engine.generate(text=["a.", "b.", "c."], return_piece_frames=True)
    assert len(two) == 2
    assert three[2] == [(0, 20), (20, 35), (35, 50)]       # the 3 trailing silence frames are in no piece
    assert len(three[1][0]) == 53


@pytest.mark.asyncio
async def test_generate_piece_frames_after_the_prefix_is_cut():
    prefix = {"semantic_codes": [7] * 12}
    # _generate_semantic returns prefix + pieces, markers [prefix end, start of piece 1]
    engine = _engine([7] * 12 + [1] * 30, [12, 30], prefix=prefix)
    engine.config.prefix_joint_decode = False
    with patch("app.inference.free_memory"), patch(
            "app.inference.decode_chunked_with_context", return_value=_codes_audio([5] * 33)):
        _, tokens, frames = await engine.generate(text=["a.", "b."], prefix_text="Before.", return_piece_frames=True)
    assert frames == [(0, 18), (18, 30)]
    assert len(tokens[0]) == 33


# ---------------------------------------------------------------------------------------------------------------------
# request validation
# ---------------------------------------------------------------------------------------------------------------------
def test_require_timestamps():
    english = TTSRequest(text="Hello.", include_timestamps=True)
    loaded = SimpleNamespace(cb0_ctc=object())
    app_main._require_timestamps(english, loaded)
    app_main._require_timestamps(TTSRequest(text="Hallo.", language="german"), loaded)   # not requested: fine
    with pytest.raises(TTSRequestOptionError, match="^include_timestamps is only available for English$") as exc:
        app_main._require_timestamps(TTSRequest(text="Hallo.", language="german", include_timestamps=True), loaded)
    assert exc.value.code == "timestamps_unsupported_language"
    with pytest.raises(TTSRequestOptionError, match="^timestamps are not available on this server$") as exc:
        app_main._require_timestamps(english, SimpleNamespace(cb0_ctc=None))
    assert exc.value.code == "timestamps_unavailable"
    assert TTSRequest(text="x", language="en", include_timestamps=True).language == "english"
    assert TTSRequest(text="x").include_timestamps is False


def test_streaming_refuses_include_timestamps():
    for transport in ("mp3", "opus"):
        assert app_main._streaming_option_error(TTSRequest(text="Hi.", include_timestamps=False), transport) is None
        message = app_main._streaming_option_error(TTSRequest(text="Hi.", include_timestamps=True), transport)
        assert message == "include_timestamps is supported only by non-streaming /predict"
        assert app_main._error_body(message) == {
            "error": "include_timestamps is supported only by non-streaming /predict",
            "code": "timestamps_not_streamable",
        }
    # other refusals keep the plain {"error": ...} body
    other = app_main._streaming_option_error(TTSRequest(text="Hi.", vllm_priority=1), "mp3")
    assert app_main._error_body(other) == {"error": other}


async def _post(monkeypatch, path, body, engine):
    async def get_engine():
        return engine

    monkeypatch.setattr(app_main, "_get_engine", get_engine)
    monkeypatch.setattr(app_main, "_unload_engine", lambda: None)
    application = create_app()
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        async with application.router.lifespan_context(application):
            return await client.post(path, json=body)


@pytest.mark.asyncio
async def test_http_422_for_non_english_missing_recogniser_and_stream(monkeypatch):
    engine = SimpleNamespace(cb0_ctc=object(), verbalizer=None, vv_tokenizer=object(), generate=AsyncMock())
    r = await _post(monkeypatch, "/predict", {"text": "Hallo.", "language": "german", "include_timestamps": True},
                    engine)
    assert r.status_code == 422 and r.json() == {"error": "include_timestamps is only available for English",
                                                 "code": "timestamps_unsupported_language"}
    r = await _post(monkeypatch, "/predict", {"text": "Hello.", "include_timestamps": True},
                    SimpleNamespace(cb0_ctc=None, verbalizer=None, vv_tokenizer=object(), generate=AsyncMock()))
    assert r.status_code == 422 and r.json() == {"error": "timestamps are not available on this server",
                                                 "code": "timestamps_unavailable"}
    r = await _post(monkeypatch, "/stream", {"text": "Hello.", "include_timestamps": True}, engine)
    assert r.status_code == 422
    assert r.json() == {"error": "include_timestamps is supported only by non-streaming /predict",
                        "code": "timestamps_not_streamable"}
    # another option refusal on /predict keeps its plain body
    r = await _post(monkeypatch, "/predict", {"text": "Hallo.", "language": "german", "use_verbalization": True},
                    engine)
    assert r.status_code == 422 and set(r.json()) == {"error"}
    engine.generate.assert_not_called()


# ---------------------------------------------------------------------------------------------------------------------
# /predict end to end: fake recogniser, real trim, real pause cap (identity enhancer, librosa resample to 48 kHz)
# ---------------------------------------------------------------------------------------------------------------------
def _postprocessor():
    processor = mossformer_module.MossFormer2Postprocessor.__new__(mossformer_module.MossFormer2Postprocessor)
    processor._process_lock = Lock()
    processor._resident = None
    processor._enhance_in_chunks = lambda source: source.copy()
    return processor


def _codes_audio(codes, seed=1):
    rng = np.random.default_rng(seed)
    audio = np.zeros(len(codes) * FRAME, np.float32)
    for k, c in enumerate(codes):
        if c:
            audio[k * FRAME:(k + 1) * FRAME] = rng.uniform(-0.3, 0.3, FRAME)
    return torch.from_numpy(audio).reshape(1, 1, -1)


async def _predict(monkeypatch, codes, frames, texts, **req):
    observed = {}
    config = ServerConfig()
    config.output_headroom_db = 0.0
    engine = SimpleNamespace(
        cb0_ctc=ScriptedRecognizer(), verbalizer=None, vv_tokenizer=object(), config=config,
        generate=AsyncMock(return_value=(_codes_audio(codes), [list(codes)] * 4, frames)),
    )

    def encode_audio(audio, _format, _bitrate, sample_rate):
        observed["audio"], observed["sample_rate"] = audio.clone(), sample_rate
        return b"audio", "audio/wav"

    async def get_engine():
        return engine

    monkeypatch.setattr(app_main, "_get_engine", get_engine)
    monkeypatch.setattr(app_main, "_encode_audio", encode_audio)
    monkeypatch.setattr(mossformer_module, "get_mossformer2_postprocessor", _postprocessor)
    monkeypatch.setattr(app_main, "_split_with_controls", lambda text, controls, **_: (texts, None))
    result = await app_main._synthesize(TTSRequest(text=" ".join(texts), **req))
    observed["result"] = result
    observed["generate_kwargs"] = engine.generate.call_args.kwargs
    return observed


def _loud_spans(audio, rate):
    x = np.abs(audio.reshape(-1).numpy()) > 1e-3
    edges = np.flatnonzero(np.diff(np.r_[0, x.astype(np.int8), 0]))
    spans = [(a / rate, b / rate) for a, b in zip(edges[::2], edges[1::2])]
    merged = []                                  # join gaps under 10 ms (noise zero crossings)
    for a, b in spans:
        if merged and a - merged[-1][1] < 0.01:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


@pytest.mark.asyncio
async def test_predict_timestamps_follow_trim_and_pause_cap(monkeypatch):
    # piece 0: 5 silent frames, 20 speech, 65 silent (5.2 s); piece 1: 5 silent, 25 speech, 10 silent
    codes = [0] * 5 + [5] * 20 + [0] * 65 + [0] * 5 + [5] * 25 + [0] * 10
    frames = [(0, 90), (90, 130)]
    texts = ["Hello there.", "General Kenobi, you are a bold one."]
    capped = await _predict(monkeypatch, codes, frames, texts, include_timestamps=True,
                            mossformer2_postprocess=True, trim_silence_padding_ms=250)
    out = capped["result"]["timestamps"]
    assert capped["sample_rate"] == 48000
    assert out["text"] == " ".join(texts)
    assert capped["generate_kwargs"]["return_piece_frames"] is True
    duration = capped["audio"].shape[-1] / 48000
    loud = _loud_spans(capped["audio"], 48000)
    assert len(loud) == 2
    # the 5.6 s pause was capped to 2.5 s and the 0.4 s lead trimmed to about 0.25 s (10 ms trim hops)
    assert loud[0][0] == pytest.approx(0.25, abs=0.015)
    assert loud[1][0] - loud[0][1] == pytest.approx(2.5, abs=0.02)
    words = out["words"]
    assert [w["text"] for w in words] == "Hello there. General Kenobi, you are a bold one.".split()
    first, second = words[:2], words[2:]
    tol = 0.021                                   # one CTC output frame
    for w in first:
        assert loud[0][0] - tol <= w["start"] < w["end"] <= loud[0][1] + tol
    for w in second:
        assert loud[1][0] - tol <= w["start"] < w["end"] <= loud[1][1] + tol
    assert first[0]["start"] == pytest.approx(loud[0][0], abs=tol)
    assert second[-1]["end"] == pytest.approx(loud[1][1], abs=tol)
    assert all(0 <= c["start"] <= c["end"] <= duration for c in out["characters"])

    # the same request uncapped (24 kHz, no pause cap): the timings differ exactly by the applied cut
    plain = await _predict(monkeypatch, codes, frames, texts, include_timestamps=True,
                           mossformer2_postprocess=False, trim_silence_padding_ms=250)
    p_words = plain["result"]["timestamps"]["words"]
    shift = (5.6 - 2.5)
    for a, b in zip(words, p_words):
        expect = (b["start"], b["end"]) if b["end"] < 2.0 else (b["start"] - shift, b["end"] - shift)
        assert a["start"] == pytest.approx(expect[0], abs=0.002) and a["end"] == pytest.approx(expect[1], abs=0.002)


@pytest.mark.asyncio
async def test_predict_without_timestamps_is_unchanged(monkeypatch):
    codes = [0] * 5 + [5] * 20 + [0] * 5
    observed = await _predict(monkeypatch, codes, [(0, 30)], ["Hello there."], mossformer2_postprocess=False)
    assert observed["result"]["timestamps"] is None
    assert "return_piece_frames" not in observed["generate_kwargs"]

"""Flash on the CPU (flash/cpu): the GPU engine's stream contract on the native engine. Needs a C++17 compiler and the
release's CPU weights in the Hugging Face cache; skipped otherwise. The first run builds the engine (about a minute)."""
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from flash.common.stream import OUT_GAIN, REPO, REVISION, assemble, cut_lead

TEXT = "Here's your shopping list: One, Milk, Two, Eggs, and Three, Bread. It is 5:30 pm."
NEVER = lambda: -10 ** 9                                # a player that never plays: every redraw is in time


@pytest.fixture(scope="module")
def weights():
    if shutil.which("g++") is None and shutil.which("c++") is None:
        pytest.skip("no C++ compiler")
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(REPO, "web/lm.bin", revision=REVISION)
    if not isinstance(p, str):
        pytest.skip("the Flash CPU weights are not in the Hugging Face cache")
    return Path(p).parent.parent


@pytest.fixture(scope="module")
def eng(weights):
    from flash.cpu.engine import Engine
    e = Engine(weights=weights, threads=4)
    yield e
    e.close()


@pytest.fixture(scope="module")
def eng24(weights):
    from flash.cpu.engine import Engine
    e = Engine(weights=weights, threads=2, sample_rate=24000)
    yield e
    e.close()


def test_48_khz_by_default_24_khz_on_request(eng, eng24):
    assert eng.sample_rate == eng.info["rate"] == 48000 and eng24.sample_rate == eng24.info["rate"] == 24000
    a, b = eng.speak("Short.", seed=1, redraw=False), eng24.speak("Short.", seed=1, redraw=False)
    assert abs(len(a) - 2 * len(b)) <= 2 * 3200 and eng.stats["cut"] % 2 == 0


def _check_pieces(pieces):
    """Appends start at the end so far; a replacement starts before it."""
    end, reps = 0, 0
    for start, audio in pieces:
        assert audio.dtype == np.float32 and 0 <= start <= end
        reps += start < end
        end = start + len(audio)
    return reps


def test_stream_and_speak(eng):
    pieces = list(eng.stream(TEXT, seed=3, played=NEVER))
    _check_pieces(pieces)
    st = eng.stats
    assert st["chunks"] == 1 and st["frames"][0] > 10 and st["t_first"] > 0 and st["cut"] > 0 and "redraw" in st
    wave = eng.speak(TEXT, seed=3, played=NEVER)
    assert np.array_equal(assemble(pieces), wave)               # the same seed, the same audio
    assert not np.array_equal(eng.speak(TEXT, seed=4, played=NEVER), wave)
    plain = list(eng.stream(TEXT, seed=3, redraw=False))
    assert _check_pieces(plain) == 0 and "redraw" not in eng.stats
    cps = eng.chunks(TEXT)
    assert np.array_equal(eng.speak(None, seed=3, chunks=cps, redraw=False), assemble(plain))


def test_lead_cut_as_in_python(eng24):
    """The native start cut is where flash.common.stream.cut_lead cuts the uncut stream (found on the audio before the
    output gain, as the engine does), and the cut output is the uncut output from there on (24 kHz: at 48 kHz the
    upsampler starts on other samples)."""
    eng = eng24
    pieces, first_cut = [], None
    for piece in eng.stream(TEXT, seed=5, redraw=False, cut=True):
        first_cut = eng.stats["cut"] if first_cut is None else first_cut          # known with the first piece
        pieces.append(piece)
    cut, c_native = assemble(pieces), eng.stats["cut"]
    assert first_cut == c_native
    got = {}
    uncut = eng.speak(TEXT, seed=5, redraw=False, cut=False)
    list(cut_lead([(0, uncut / OUT_GAIN)], lambda c: got.update(c=c)))
    assert eng.stats["cut"] == 0 and got["c"] == c_native > 0
    assert np.array_equal(cut, uncut[c_native:])


@pytest.fixture(scope="module")
def redraw_seed(eng):
    """A seed whose take has a replacement when nothing stops the redraw."""
    seed = next((s for s in range(8) if _check_pieces(eng.stream(TEXT, seed=s, played=NEVER)) > 0), None)
    if seed is None:
        pytest.skip("no redraw in these takes")
    return seed


def test_replacements_stay_ahead_of_the_player(eng, redraw_seed):
    seed = redraw_seed
    t0 = [None]

    def played():                                       # a slow player (a tenth of real time) from the first piece
        return 0 if t0[0] is None else int((time.perf_counter() - t0[0]) * 2400)
    end, reps = 0, 0
    for start, audio in eng.stream(TEXT, seed=seed, played=played):
        t0[0] = t0[0] or time.perf_counter()
        if start < end:
            reps += 1
            assert start >= played()                    # never into audio the player has taken
        end = start + len(audio)
    assert reps > 0


def test_boundary_of_a_caller_that_holds_back(eng, redraw_seed):
    """A boundary five frames behind the newest sample (a server that sends audio on late): the take of a player that
    never plays, and no replacement before the boundary."""
    ref = eng.speak(TEXT, seed=redraw_seed, played=NEVER)
    buf, sent, reps = np.zeros(0, np.float32), 0, 0
    for start, audio in eng.stream(TEXT, seed=redraw_seed, boundary=lambda: sent):
        assert start >= sent
        reps += start < len(buf)
        buf = np.concatenate([buf[:start], audio])
        sent = max(sent, len(buf) - 5 * 3200 * eng.sample_rate // 24000)
    assert reps > 0 and np.array_equal(buf, ref)
    late = list(eng.stream(TEXT, seed=redraw_seed, boundary=lambda: 10 ** 9))     # everything final: no replacement
    assert _check_pieces(late) == 0
    with pytest.raises(ValueError):
        next(eng.stream(TEXT, played=NEVER, boundary=NEVER))


def test_threads_cut_to_what_can_run(eng, weights):
    from flash.cpu.engine import Engine
    with pytest.warns(RuntimeWarning):
        big = Engine(weights=weights, threads=10000)
    try:
        assert big.info["asked"] == 10000 and big.info["threads"] == big.info["usable"] <= big.info["physical_cores"]
        assert not set(big.info["cpus"]) & set(eng.info["cpus"])
    finally:
        big.close()


def test_stop_and_busy(eng):
    g = eng.stream(TEXT * 3, seed=1)
    for _ in range(3):
        next(g)
    with pytest.raises(RuntimeError):
        next(eng.stream(TEXT, seed=1))
    t = time.perf_counter()
    g.close()                                           # stops the request at the next frame
    assert time.perf_counter() - t < 5
    assert len(eng.speak("Hello.", seed=1)) > 3200


def test_chunk_too_long(eng):
    with pytest.raises(ValueError):
        eng.speak(None, chunks=[np.full(3000, ord("a"), np.int32)])


def test_two_engines_in_one_process(eng, weights):
    from flash.cpu.engine import Engine
    other = Engine(weights=weights, threads=2)
    try:
        if eng.info["cpus"] and other.info["cpus"]:
            assert not set(eng.info["cpus"]) & set(other.info["cpus"])
        ref = [eng.speak(t, seed=2, redraw=False) for t in ("One engine.", "And another one.")]
        out = [None, None]

        def run(k, e, t):
            out[k] = e.speak(t, seed=2, redraw=False)
        th = [threading.Thread(target=run, args=(0, eng, "One engine.")), threading.Thread(target=run, args=(1, other, "And another one."))]
        for x in th:
            x.start()
        for x in th:
            x.join()
        assert np.array_equal(out[0], ref[0]) and np.array_equal(out[1], ref[1])
    finally:
        other.close()


def test_set_sampler(eng):
    ref = eng.speak("Short.", seed=1, redraw=False)
    solver, passes = eng.sampler
    eng.set_sampler(solver, passes, 0.5)
    assert not np.array_equal(eng.speak("Short.", seed=1, redraw=False), ref)
    eng.set_sampler(solver, passes, 0.7)
    assert np.array_equal(eng.speak("Short.", seed=1, redraw=False), ref)

"""TontaubeV1.1 Flash from the command line (voice: Marcus, English), on an NVIDIA GPU or on the CPU.

  python flash/speak.py "Some text." --wav out.wav [--device gpu|cpu] [--variant int4|int8|fp16] [--weights DIR] [--seed 0]
         [--no-redraw] [--cutoff 0.85] [--max-takes 4] [--no-cut] [--no-verbalize] [--first-chunk N] [--repeat R] [--threads N]
         [--rate 48000|24000]
  python flash/speak.py --lines texts.txt --wav out_dir      one text per line, the model loaded once

The weights are downloaded from Hugging Face on the first run (or taken from --weights DIR). Prints the time to the
first audio, the speed, the silence skipped at the start and what the redraw did. The audio is streamed to a simulated
player that plays at real time, so a redraw only replaces audio a live player would not have played yet.
"""
import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

if __package__ in (None, ""):                         # run as a script: the repository root on the import path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ap = argparse.ArgumentParser(description="Speak English text with TontaubeV1.1 Flash and write a wav file (CPU 48 kHz, GPU 24 kHz).")
ap.add_argument("text", nargs="?", help="the text to speak (written English: numbers, dates, money and so on are read out)")
ap.add_argument("--wav", required=True, help="output wav file (with --lines: an output folder)")
ap.add_argument("--lines", default=None, help="a text file with one text per line instead of TEXT")
ap.add_argument("--device", choices=["gpu", "cpu"], default=None, help="gpu (an NVIDIA GPU) or cpu; default: gpu if there is one")
ap.add_argument("--variant", default="int4", help="int4 (default, smallest), int8 or fp16 (GPU only)")
ap.add_argument("--weights", default=None, help="a local folder with the weights laid out as on Hugging Face, instead of downloading them")
ap.add_argument("--seed", type=int, default=0, help="random seed: another seed, another take")
ap.add_argument("--temp", type=float, default=0.7, help="sampling temperature (default 0.7)")
ap.add_argument("--passes", type=int, default=8, help="flow head passes per frame (default 8)")
ap.add_argument("--solver", default="midpoint", choices=["midpoint", "euler"], help="flow solver (default midpoint)")
ap.add_argument("--no-redraw", action="store_true", help="turn the redraw off: keep pitch jumps the detector hears as they were drawn")
ap.add_argument("--cutoff", type=float, default=0.85, help="how sure the detector must be before a spot is drawn again (default 0.85)")
ap.add_argument("--max-takes", type=int, default=4, help="takes of a spot, the first one included (default 4: up to 3 redraws)")
ap.add_argument("--no-cut", action="store_true", help="keep the short silence before the first sound (by default the audio starts 5 ms before it)")
ap.add_argument("--no-verbalize", action="store_true", help="the text is already written as it is spoken (no reading of numbers etc.)")
ap.add_argument("--first-chunk", type=int, default=0, help="a first chunk of about N characters: faster first audio for long texts")
ap.add_argument("--repeat", type=int, default=1, help="say each text R times with seeds seed, seed + 1, ... (timings of a warm model)")
ap.add_argument("--threads", type=int, default=None, help="CPU: threads (default: automatic)")
ap.add_argument("--rate", type=int, choices=[48000, 24000], default=48000, help="output sample rate (default 48000)")
a = ap.parse_args()
if not a.text and not a.lines:
    ap.error("give a TEXT or --lines FILE")


CU126_HINT = ("for the GPU, install the CUDA 12.6 PyTorch build (README.md, \"GPUs the default PyTorch build does "
              "not support\")")


def gpu_unsupported():
    """The installed PyTorch build has no kernels for the GPU (e.g. the default CUDA 12.8 wheels on a GTX 10 series card):
    its name, else None. The same rule as `tontaube serve` (app/torch_variant.py)."""
    try:
        import torch
        from app.torch_variant import build_supports
        cap = torch.cuda.get_device_capability(0)
        if build_supports(cap, torch.cuda.get_arch_list()):
            return None
        return f"{torch.cuda.get_device_name(0)} (compute capability {cap[0]}.{cap[1]})"
    except Exception:                      # never block a run because of the check itself
        return None


def has_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except ImportError:
        return False
    bad = gpu_unsupported()
    if bad:
        print(f"{bad} is not supported by this PyTorch build: using the CPU; {CU126_HINT}.", file=sys.stderr)
        return False
    return True


def save_wav(path, x, sr):
    """float32 audio at sr -> 16-bit mono wav."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(np.asarray(x, np.float32), -1, 1) * 32767).astype(np.int16).tobytes())


device = a.device or ("gpu" if has_gpu() else "cpu")
if a.device == "gpu" and gpu_unsupported():
    sys.exit(f"{gpu_unsupported()} is not supported by this PyTorch build; use --device cpu, or {CU126_HINT}.")
common = dict(solver=a.solver, passes=a.passes, temp=a.temp, redraw=not a.no_redraw, cutoff=a.cutoff, max_takes=a.max_takes,
              lead_cut=not a.no_cut, weights=a.weights)
t0 = time.perf_counter()
if device == "gpu":
    from flash.gpu.engine import Engine
    eng = Engine(a.variant, sample_rate=a.rate, **common)
    where = "GPU"
else:
    from flash.cpu.engine import Engine
    eng = Engine(a.variant, threads=a.threads, sample_rate=a.rate, **common)
    where = f"CPU, {eng.info['threads']} threads"
print(f"engine {eng.variant} ({where}) ready in {time.perf_counter() - t0:.2f} s; sampler {a.solver} {a.passes} passes, temperature {a.temp}; "
      + ("no redraw" if a.no_redraw or eng.judge is None else f"redraw at {a.cutoff}, up to {a.max_takes} takes"), file=sys.stderr)
texts = [l.strip() for l in open(a.lines, encoding="utf-8") if l.strip()] if a.lines else [a.text]
if a.lines:
    Path(a.wav).mkdir(parents=True, exist_ok=True)
for k, text in enumerate(texts):
    for r in range(a.repeat):
        t0 = time.perf_counter()
        out = eng.speak(text, seed=a.seed + r, verbalize=not a.no_verbalize, first_max=a.first_chunk)
        total = time.perf_counter() - t0
        st = eng.stats
        secs = len(out) / eng.sample_rate
        if not len(out):
            print("nothing to say", file=sys.stderr)
            continue
        rd = st.get("redraw")
        print(f"{st['chars']} characters in {st['chunks']} chunk(s): first audio {1e3 * st['t_first']:.1f} ms"
              + (f" ({1e3 * st['cut'] / eng.sample_rate:.0f} ms of silence skipped)" if st["cut"] else "")
              + f", {secs:.1f} s of audio in {total:.3f} s = {secs / total:.1f}x real time"
              + (f"; redraw: {rd['spots']} spot(s), {rd['attempts']} redraw(s), clean after 1/2/3: {rd['clean_after']}, kept a non-clean "
                 f"take {rd['kept_best_unclean']}, given up {rd['given_up_played'] + rd['given_up_previous_chunk']}" if rd else "")
              + (f"; characters the model does not know, read as spaces: {st['unknown']!r}" if st["unknown"] else ""), file=sys.stderr)
    if len(out):
        path = Path(a.wav) / f"{k + 1:04d}.wav" if a.lines else a.wav
        save_wav(path, out, eng.sample_rate)
        print(path)

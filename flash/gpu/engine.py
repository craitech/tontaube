"""TontaubeV1.1 Flash on one GPU (voice: Marcus): text -> rule verbalizer -> chunker -> frame model (Qwen3-0.6B backbone
with a flow head) -> autoencoder decoder -> VibeVoice decoder, one 133 ms frame at a time, streaming.

    from flash.gpu.engine import Engine
    eng = Engine("int4")                            # downloads the weights once; compiles, captures, warms up
    for start, audio in eng.stream("Some text."):   # float32 audio (48 kHz) for the player's samples from `start` on
        ...
    wave = eng.speak("Some text.")

Every weight is read straight from its stored format (dense fp16, int8 or int4) by one Triton kernel with the bias /
activation / residual fused in; the key/value cache and every buffer are static; the text prefix runs in one batch;
the frame step, the head's passes, the autoencoder and the VibeVoice decoder are captured as CUDA graphs and replayed,
so a frame costs a handful of host calls and one wait.

Generation: per chunk, the previous chunk's text and frames and the first 50 characters of the next chunk are the
context; a chunk ends when the stop value turns positive (after at least 2 frames). Two features on top, both on by
default: the redraw (redraw.py: a small detector listens to the audio as it is made, and where the pitch jumps the
frames are drawn again before they are played) and the start at the first sound (cut_lead: the silence before the
first sound is not played).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import kernels as K
from ..common.frontend import FrontEnd
from ..common.stream import HOP, OUT_GAIN, SR, Speaker, cut_lead, release
from .bwe import BWE, DELAY, Stream as Up48
from .detector import Judge
from .redraw import Redraw

HERE = Path(__file__).parent
# backbone, head + autoencoder ("mix": head int8, autoencoder fp16), VibeVoice decoder
VARIANTS = {"fp16": ("fp16", "fp16", "fp16"), "int8": ("int8", "mix", "int8"), "int4": ("int4", "mix", "int8")}
FEW = 64                       # up to this many rows a matrix is read in its stored format; above: dense + cuBLAS
MLP_SCALE = 1 / 16             # the MLP's inner activation is kept as fp16 / 16 (it reaches several thousand)
RING = 16                      # decoder-state snapshots kept (a redraw goes back at most a few frames)

_TUNE = {}                     # launch configurations per matrix shape, read by Engine()


def tuning_path(device="cuda"):
    """This GPU's own table if there is one, else tuning.json (tuned on an RTX 3090)."""
    p = HERE / f"tuning_{torch.cuda.get_device_name(device).replace(' ', '_')}.json"
    return p if p.exists() else HERE / "tuning.json"


def _heuristic(n, k, bits):
    """(BLOCK_N, SPLIT_K, warps) for a matrix without a tuned entry."""
    split = 1
    while (n // 32) * split < 128 and k // (128 * split * 2) >= 2:
        split *= 2
    return 32, split, 4


class Lin:
    """A linear layer in its stored format. scratch: the engine's buffer for the partial sums of split launches (one
    per engine: its launches run one after another)."""

    def __init__(self, f, meta, name, dev, scratch):
        m = meta["linears"][name]
        self.name, self.n, self.k, self.bits, self.group = name, m["n"], m["k"], m["bits"], m["group"]
        get = lambda s: f.get_tensor(name + s).to(dev)
        if self.bits == 16:
            self.w, self.scales, self.mins = get(".w"), None, None
        else:
            self.w, self.scales, self.mins = get(".codes"), get(".scales"), get(".mins")
        self.bias = get(".b") if name + ".b" in f.keys() else None
        self.bias16, self.scratch = None, scratch
        self.keep_dense, self.w16 = False, None        # keep_dense: many() decodes packed weights once and keeps them

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.w, self.scales, self.mins, self.bias) if t is not None)

    def config(self, pair=False, M=1):
        """Tuned (BLOCK_N, SPLIT_K, warps) for this shape and format; more than 16 rows have their own entry."""
        key = f"{self.bits},{self.group},{self.n // 2 if pair else self.n},{self.k},{int(pair)}"
        for k in ((f"{key},{M}", f"{key},64", key) if M > 16 else (key,)):
            if k in _TUNE:
                return tuple(_TUNE[k])
        return _heuristic(self.n // 2 if pair else self.n, self.k, self.bits)

    def few(self, x, out, epi=K.EPI_NONE, res=None, gate=None, pair=False, out_scale=1.0, cfg=None, norm=None, eps=0.0):
        """x [M <= 64, K] fp16 -> out [M, N] (pair: [M, N / 2] = SiLU(first half) * second half * out_scale).
        norm: x is one fp32 row of a residual stream; RMSNorm with this weight runs inside the same kernel."""
        block_n, split, warps = cfg or self.config(pair, x.shape[0])
        n = self.n // 2 if pair else self.n
        if split * 2 * x.shape[0] * n > self.scratch.numel():
            split = 1
        part = self.scratch if split > 1 else None
        K.launch_linear(x, self.w, self.scales, self.mins, self.bias, res, gate, out, part, self.bits, self.group, n, epi, pair,
                        out_scale, block_n, split, warps, norm, eps)
        return out

    def dense(self):
        """Dense fp16 weights (packed formats: decoded for this call)."""
        if self.bits == 16:
            return self.w
        return K.dequantize(self.w, self.scales, self.mins, self.bits, self.group, self.n, self.k)

    def many(self, x):
        """x [M, K] fp16 -> fp16 [M, N] through cuBLAS."""
        if self.bias is not None and self.bias16 is None:
            self.bias16 = self.bias.half()
        if self.keep_dense and self.w16 is None:
            self.w16 = self.dense()
        return F.linear(x, self.w16 if self.w16 is not None else self.dense(), self.bias16)


def weight_files(formats, weights=None, bwe=False):
    """Paths of the three weight files (backbone, head + autoencoder, VibeVoice decoder: formats as in VARIANTS) and of
    the detector (and of the 48 kHz upsampler, web/bwe.bin, with bwe). weights: a local folder laid out as the release
    (config.json, gpu/...), or None: the release on Hugging Face, downloaded once into the Hugging Face cache."""
    kinds = ("lm", "misc", "vv")
    names = [f"gpu/{k}_{f}.safetensors" for k, f in zip(kinds, formats)]
    cfg, get = release(weights)
    for v, files in cfg.get("gpu_variants", {}).items():
        if tuple(VARIANTS.get(v, ())) == tuple(formats):
            names = files
    if bwe:
        kinds, names = kinds + ("bwe",), names + [next((n for n in cfg.get("web", []) if Path(n).name == "bwe.bin"), "web/bwe.bin")]
    paths = dict(zip(kinds, map(get, names)))
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("weight files not found: " + ", ".join(missing) + " (weights= is the release folder: config.json, gpu/, web/"
                                + ("; bwe.bin is for 48 kHz, sample_rate=24000 runs without it" if bwe else "") + ")")
    det = cfg.get("detector", "gpu/detector.safetensors")
    try:
        paths["detector"] = get(det)
    except Exception:                                   # no detector in the folder: the engine runs without the redraw
        paths["detector"] = None
    return paths


def _open(path):
    f = safe_open(str(path), "pt", device="cpu")
    return f, json.loads(f.metadata()["meta"])


class Engine(Speaker):
    def __init__(self, variant="int4", lm=None, misc=None, vv=None, device="cuda", max_rows=1536, first="default",
                 solver="midpoint", passes=8, temp=0.7, prefill_buckets="default", graphs=True, verbose=False, few_max=None,
                 redraw=True, cutoff=0.85, max_takes=4, margin_ms=50.0, lead_cut=True, weights=None, sample_rate=48000):
        """variant: int4 (default: smallest) | int8 | fp16 (or the three parts' formats one by one). weights: a local
        folder laid out as the release; None = download the release from Hugging Face once. first: "default" = the
        fixed first frames stored in the weights (none), None = none, or an array [K, 16]. prefill_buckets: text
        prefixes up to the largest bucket replay a captured graph (padded to the bucket), longer ones run op by op.
        redraw: flagged pitch jumps are drawn again (redraw.py; per request too: stream(redraw=...)); cutoff: the
        detector's spot measure that flags a spot; max_takes: takes of a spot counting the original; margin_ms: audio
        the player may take before a replacement can reach it. lead_cut: every request starts at its first sound
        (cut_lead; per request too: stream(cut=...)). sample_rate: 48000 (the model's 24 kHz raised to 48 kHz by an
        upsampler, web/bwe.bin) or 24000; self.sample_rate is the output rate, and every position (pieces, played,
        boundary, stats["cut"]) counts samples at it."""
        if sample_rate not in (48000, 24000):
            raise ValueError(f"sample_rate: 48000 or 24000 (not {sample_rate!r})")
        t0 = time.perf_counter()
        self.sample_rate, self.delay = sample_rate, DELAY if sample_rate == 48000 else 0   # delay: output samples the upsampler holds back
        self.dev = torch.device(device)
        tp = tuning_path(self.dev)
        _TUNE.clear()
        _TUNE.update(json.loads(tp.read_text()) if tp.exists() else {})
        self.tuning = tp.name
        fl, fm_, fv = VARIANTS[variant] if variant in VARIANTS else variant.split("+")       # or "lm+misc+vv" formats
        self.formats = (lm or fl, misc or fm_, vv or fv)
        self.variant = variant if (lm, misc, vv) == (None, None, None) else "+".join(self.formats)
        self.files = weight_files(self.formats, weights, bwe=sample_rate == 48000)
        self.timing = {}
        # prefixes up to this length read the packed weights directly (64 rows per launch, faster than decoding them
        # for cuBLAS up to 96 rows); dense fp16 weights go to cuBLAS above 64
        self.few_max = few_max or (FEW if self.formats[0] == "fp16" else 96)
        with torch.inference_mode():
            self._load(first)
            self.bwe = BWE(self.files["bwe"], self.dev, graph=graphs, batch_sizes=getattr(self, "sizes", (1,)),
                           half=getattr(self, "bwe_half", False)) if sample_rate == 48000 else None
            self.timing["load_weights_s"] = time.perf_counter() - t0
            self.front = FrontEnd()
            self._buffers(max_rows)
            t1 = time.perf_counter()
            self.use_graphs = graphs
            self.graphs, self.head_graphs = {}, {}
            self.pool = torch.cuda.graph_pool_handle() if graphs else None
            self.temp = temp
            # A first chunk's prefix has at most 350 + 1 + 52 + 1 + 3 fixed frames = 407 positions. Above 96 the buckets
            # end where the time of the cuBLAS path jumps (it is flat in between).
            self.buckets = (16, 32, 48, 64, 80, 96, 128, 168, 256, 304, 352, 384, 416) if prefill_buckets == "default" else tuple(prefill_buckets)
            self.set_sampler(solver, passes, temp)
            self._capture()
            self.redraw, self.cutoff, self.max_takes, self.margin_ms, self.lead_cut = redraw, cutoff, max_takes, margin_ms, lead_cut
            det = self.files["detector"]
            self.judge = Judge(det, self.dev, graphs=graphs) if det is not None and det.exists() else None
            self.rd = None
            self.timing["compile_capture_s"] = time.perf_counter() - t1
            t2 = time.perf_counter()
            self._start_state()
            self._warm_up()
            torch.cuda.synchronize()
            self.timing["warm_up_s"] = time.perf_counter() - t2
        self.timing["total_s"] = time.perf_counter() - t0
        if verbose:
            print(f"engine {self.variant}: ready in {self.timing['total_s']:.2f} s (weights {self.timing['load_weights_s']:.2f}, kernels and graphs "
                  f"{self.timing['compile_capture_s']:.2f}, warm-up {self.timing['warm_up_s']:.2f})", flush=True)

    def _warm_up(self):
        self.frame_s = 0.005                     # time per frame for the redraw budget: measured from here on
        for _ in range(2):
            self.speak("Warm up. This is the second sentence.", seed=0)
        st = self.stats
        self.frame_s = (st["t_total"] - st["t_first_frame"]) / max(1, sum(st["frames"]) - 1)

    # ------------------------------------------------------------------------------------------------------- load
    def _load(self, first):
        dev = self.dev
        self.scratch = torch.empty(8 * 2 * FEW * 4096, device=dev, dtype=torch.float32)   # see Lin
        L = lambda name: Lin(f, meta, name, dev, self.scratch)                             # noqa: E731  (the file being read)
        f, meta = _open(self.files["lm"])
        c = self.cfg = meta["cfg"]
        self.H, self.NL, self.NQ, self.NKV, self.HD = c["hidden"], c["layers"], c["heads"], c["kv_heads"], c["head_dim"]
        self.eps = c["eps"]
        g = lambda name: f.get_tensor(name).to(dev)
        self.layers = [dict(in_norm=g(f"L{i}.in_norm"), post_norm=g(f"L{i}.post_norm"), q_norm=g(f"L{i}.q_norm"), k_norm=g(f"L{i}.k_norm"),
                            qkv=L(f"L{i}.qkv"), o=L(f"L{i}.o"), gu=L(f"L{i}.gu"), down=L(f"L{i}.down")) for i in range(self.NL)]
        self.norm = g("norm")
        self.inv_freq = 1.0 / (c["theta"] ** (torch.arange(0, self.HD, 2, dtype=torch.float32, device=dev) / self.HD))

        f, meta = _open(self.files["misc"])
        m = self.mcfg = meta["cfg"]
        g = lambda name: f.get_tensor(name).to(dev)
        self.D, self.FW, self.NB = m["d"], m["head_width"], m["head_blocks"]
        self.rate, self.text_off, self.lookahead, self.off, self.min_frames = m["rate"], m["text_off"], m["lookahead"], m["off"], m["min_frames"]
        self.chars = f.get_tensor("chars").numpy()
        self.char_row = {int(cp): i for i, cp in enumerate(self.chars)}
        self.char_lut = np.full(int(self.chars.max()) + 2, -1, np.int64)
        self.char_lut[self.chars] = np.arange(len(self.chars))
        self.char_emb, self.start, self.tmark, self.amark = g("char_emb"), g("start"), g("text_marker"), g("audio_marker")
        self.fin0_w, self.fin0_b, self.fin2 = g("frame_in0.w"), g("frame_in0.b"), L("frame_in2")
        self.x_in_w, self.x_in_b = g("fl.x_in.w"), g("fl.x_in.b")
        self.cn_w, self.cn_b, self.c_in = g("fl.c_norm.w"), g("fl.c_norm.b"), L("fl.c_in")
        self.t0_w, self.t0_b, self.t2_w, self.t2_b = g("fl.t0.w"), g("fl.t0.b"), g("fl.t2.w"), g("fl.t2.b")
        self.mods = L("fl.mods")
        self.mlp = [(L(f"fl.mlp{k}a"), L(f"fl.mlp{k}b")) for k in range(self.NB)]
        self.out_w, self.out_b, self.stop_w, self.stop_b = g("fl.out.w"), g("fl.out.b"), g("stop.w").flatten(), g("stop.b")
        self.ae_in_w, self.ae_in_b, self.ae_k = g("ae.in.w"), g("ae.in.b"), m["ae_k"]
        self.ae_blocks = [dict(nw=g(f"ae.b{k}.norm.w"), nb=g(f"ae.b{k}.norm.b"), conv=L(f"ae.b{k}.conv"),
                               mix=L(f"ae.b{k}.mix")) for k in range(len(self.ae_k))]
        self.ae_nw, self.ae_nb, self.ae_out_w, self.ae_out_b = g("ae.norm.w"), g("ae.norm.b"), g("ae.out.w"), g("ae.out.b")
        self.AW = m["ae_width"]
        if isinstance(first, str):
            self.first = g("first")
        else:
            self.first = torch.zeros((0, self.D), device=dev) if first is None else torch.as_tensor(np.asarray(first, np.float32), device=dev).reshape(-1, self.D)
        self.KF = len(self.first)
        # VibeVoice silence latents decoded in front of every utterance (their audio is dropped)
        self.prime = g("prime") if "prime" in f.keys() else torch.zeros((0, 64), device=dev)

        f, meta = _open(self.files["vv"])
        v = self.vcfg = meta["cfg"]
        g = lambda name: f.get_tensor(name).to(dev)
        self.vv_stem = L("vv.stem")
        self.vv_stages, T = [], 1
        for i, (depth, C, s) in enumerate(zip(v["depths"], v["dims"], v["strides"])):
            up = L(f"vv.up{i}") if i else None
            T_in, T = T, T * s
            blocks = [dict(taps=g(f"vv.s{i}.b{j}.taps"), dwb=g(f"vv.s{i}.b{j}.dwb"), fw=g(f"vv.s{i}.b{j}.fw"), phi=g(f"vv.s{i}.b{j}.phi"),
                           l1=L(f"vv.s{i}.b{j}.l1"), l2=L(f"vv.s{i}.b{j}.l2")) for j in range(depth)]
            self.vv_stages.append(dict(C=C, T=T, T_in=T_in, s=s, up=up, blocks=blocks))
        self.vv_head_w, self.vv_head_b, self.vv_eps = g("vv.head.w"), g("vv.head.b"), v["eps"]
        assert T == HOP

    def weight_bytes(self) -> dict:
        lm = sum(L[k].nbytes() for L in self.layers for k in ("qkv", "o", "gu", "down"))
        head = sum(l.nbytes() for l in [self.fin2, self.c_in, self.mods] + [x for ab in self.mlp for x in ab])
        ae = sum(b[k].nbytes() for b in self.ae_blocks for k in ("conv", "mix"))
        vv = self.vv_stem.nbytes() + sum((s["up"].nbytes() if s["up"] else 0) + sum(b["l1"].nbytes() + b["l2"].nbytes() for b in s["blocks"])
                                         for s in self.vv_stages)
        return {"backbone": lm, "head_and_frame_input": head, "autoencoder": ae, "vibevoice": vv,
                "kv_cache": self.kc.numel() * 2 * 2, "frame_embeddings_and_table": self.src.numel() * 4}

    # ---------------------------------------------------------------------------------------------------- buffers
    def _buffers(self, S):
        dev, H = self.dev, self.H
        z = lambda *shape, dtype=torch.float32: torch.zeros(shape, device=dev, dtype=dtype)
        self.S = S
        self.kc, self.vc = z(self.NL, S, self.NKV, self.HD, dtype=torch.half), z(self.NL, S, self.NKV, self.HD, dtype=torch.half)
        # embedding source: characters, text marker, START, audio marker, padding (zeros), the fixed first frames; then
        # the frames' embeddings of the previous / current chunk
        nch = len(self.chars)
        self.R_TM, self.R_START, self.R_AM, self.R_PAD = nch, nch + 1, nch + 2, nch + 3
        self.FE0, self.MAXF = nch + 4 + 16, 512          # up to 16 fixed first frames sit right before FE0
        self.R_FIRST = self.FE0 - self.KF
        self.src = z(self.FE0 + self.MAXF, H)
        self.src[:nch] = self.char_emb
        self.src[self.R_TM], self.src[self.R_START], self.src[self.R_AM] = self.tmark, self.start, self.amark
        # control values, written by one host copy each per chunk: [row, len, frame-embedding row, noise index, last
        # prefix token, 0, -, -, prefix ids...] and [next frame position, -, ..., prefix positions...]
        self.MAXT = S
        self.ctl_i, self.ctl_f = z(8 + self.MAXT, dtype=torch.long), z(8 + self.MAXT)
        self.ctl_i_host, self.ctl_f_host = np.zeros(8 + self.MAXT, np.int64), np.zeros(8 + self.MAXT, np.float32)
        self.row, self.len, self.fe_row, self.ti, self.last, self.zero = (self.ctl_i[k:k + 1] for k in range(6))
        self.pos = self.ctl_f[0:1]
        # frame step
        self.x, self.xm = z(1, self.D), z(1, self.D)
        self.h1, self.res = z(1, H, dtype=torch.half), z(1, H)
        self.qkv = z(1, (self.NQ + 2 * self.NKV) * self.HD)
        self.q, self.att = z(1, self.NQ, self.HD, dtype=torch.half), z(1, self.NQ * self.HD, dtype=torch.half)
        self.act = z(1, self.cfg["inter"], dtype=torch.half)
        self.hn, self.stop = z(1, H), z(1)
        self.attn = K.DecodeAttention(self.NQ, self.NKV, self.HD, S, dev)
        # head
        self.y16, self.c0, self.hx, self.z16 = z(1, H, dtype=torch.half), z(1, self.FW), z(1, self.FW), z(1, self.FW, dtype=torch.half)
        self.noise = z(self.MAXF, self.D)
        self.gen = torch.Generator(device=dev)
        # autoencoder and VibeVoice states in one arena: [slot flag, autoencoder rings, VibeVoice slot 0, slot 1]
        off = 1
        self.ae_off = []
        for k in self.ae_k:
            self.ae_off.append(off)
            off += (k - 1) * self.AW
        self.vv_off0 = off
        o = 0
        self.vv_stem_off = o; o += 6 * self.vcfg["latent"]
        for st in self.vv_stages:
            if st["up"] is not None:
                st["up_off"] = o; o += st["up"].k // 2
            for b in st["blocks"]:
                b["off"] = o; o += 6 * st["C"]
        self.vv_head_off = o; o += 6 * self.vv_stages[-1]["C"]
        self.vv_slot = o
        self.arena = z(self.vv_off0 + 2 * self.vv_slot)
        self.arena_start = self.arena.clone()
        # redraw: per frame of the chunk the state that predicts it; per recent frame the decoders' state before it
        self.hn_hist, self.arena_hist, self._rd_audio = z(self.MAXF + 1, H), z(RING, self.arena.numel()), None
        self.one = torch.ones(1, device=dev)
        self.ax, self.az16, self.af16, self.lat = z(1, self.AW), z(1, 2 * self.AW, dtype=torch.half), z(self.AW, dtype=torch.half), z(self.vcfg["latent"])
        self.ae_u = [z(1, k * self.AW, dtype=torch.half) for k in self.ae_k]
        self.vv_u0 = z(1, 7 * self.vcfg["latent"], dtype=torch.half)
        for st in self.vv_stages:
            T, C = st["T"], st["C"]
            st["few"] = T <= FEW
            st["xa"] = z(T, C) if st["up"] is None or st["T_in"] <= FEW else None      # the stage's input where a kernel writes it
            st["xm"], st["hn"] = (z(T, C), z(T, C)), z(T, C, dtype=torch.half)     # a block never writes the buffer it reads
            st["g"] = z(T, 4 * C, dtype=torch.half) if st["few"] else None
            if st["up"] is not None:
                st["u"] = z(st["T_in"], st["up"].k, dtype=torch.half)
        self.audio = z(HOP)
        self.audio_host = torch.zeros(HOP, dtype=torch.float32).pin_memory()
        self.stop_host = torch.zeros(1, dtype=torch.float32).pin_memory()
        self.ev = torch.cuda.Event()

    # ------------------------------------------------------------------------------------------------------ sampler
    def set_sampler(self, solver="midpoint", passes=6, temp=0.8):
        """solver euler | midpoint, head passes per frame (midpoint: 2 per step), temperature of the start noise."""
        assert solver in ("euler", "midpoint")
        with torch.inference_mode():
            self.temp = float(temp)
            euler = solver == "euler"
            P = passes if euler else 2 * max(1, passes // 2)
            steps = P if euler else P // 2
            key = (solver, P)
            if getattr(self, "sampler", None) == key:
                return
            self.sampler, self.P, self.steps, self.euler = key, P, steps, euler
            if key not in self.head_graphs:
                ts = [p / steps if euler else (p // 2) / steps + (p % 2) / (2 * steps) for p in range(P)]
                t = torch.tensor(ts, dtype=torch.float32, device=self.dev)
                half = self.mcfg["t_dim"] // 2
                a = 1000 * t[:, None] * torch.exp(-np.log(10000) * torch.arange(half, device=self.dev) / half)
                tm = F.linear(F.mish(F.linear(torch.cat([a.cos(), a.sin()], -1), self.t0_w, self.t0_b)), self.t2_w, self.t2_b).contiguous()
                bufs = dict(tm=tm, ws=torch.zeros((P, self.FW), device=self.dev, dtype=torch.half),
                            modb=torch.zeros((P, self.mods.n), device=self.dev))
                self.head_graphs[key] = [bufs, None]
            self.hb = self.head_graphs[key][0]
            if self.use_graphs and getattr(self, "g_lm", None) is not None and self.head_graphs[key][1] is None:
                self.head_graphs[key][1] = self._graph(self._head)

    # ------------------------------------------------------------------------------------------------- the pieces
    def _head(self):
        """Final state hn -> stop value and the next frame x (start noise: row ti of the chunk's noise)."""
        b, FW = self.hb, self.FW
        K.head_pre(self.hn, self.cn_w, self.cn_b, self.stop_w, self.stop_b, self.y16, self.stop)
        self.c_in.few(self.y16, self.c0)
        K.head_ws(self.c0, b["tm"], b["ws"])
        self.mods.few(b["ws"], b["modb"])
        torch.index_select(self.noise, 0, self.ti, out=self.x)
        nb3, coef = 3 * FW, 1.0 / self.steps
        for p in range(self.P):
            mid = (not self.euler) and p % 2 == 1
            src = self.xm if mid else self.x
            mod = b["modb"][p]
            K.head_in(src, self.x_in_w, self.x_in_b, mod, self.hx, self.y16)
            for k, (la, lb) in enumerate(self.mlp):
                if k:
                    K.head_mid(self.hx, mod[k * nb3:], self.y16)
                la.few(self.y16, self.z16, epi=K.EPI_MISH)
                lb.few(self.z16, self.hx, epi=K.EPI_GATED, res=self.hx, gate=b["modb"][p:p + 1, k * nb3 + 2 * FW:(k + 1) * nb3])
            if self.euler or mid:
                K.head_out(self.hx, mod[self.NB * nb3:], self.out_w, self.out_b, self.x, self.x, coef)
            else:
                K.head_out(self.hx, mod[self.NB * nb3:], self.out_w, self.out_b, self.x, self.xm, coef / 2)

    def _lm_step(self):
        """The frame x through the backbone at the next cache row: its embedding (kept for the next chunk's prefix),
        28 layers reading the stored weights, the final state hn; the counters move on."""
        K.small_linear(self.x, self.fin0_w, self.fin0_b, self.h1, act=1)
        self.fin2.few(self.h1, self.res)
        self.src.index_copy_(0, self.fe_row, self.res)
        att2 = self.att.view(self.NQ, self.HD)
        for li, L in enumerate(self.layers):                    # both RMSNorms of a layer run inside the linear that follows
            L["qkv"].few(self.res, self.qkv, norm=L["in_norm"], eps=self.eps)
            K.qk_norm_rope_store(self.qkv, L["q_norm"], L["k_norm"], self.pos, self.inv_freq, self.kc[li], self.vc[li], self.row, self.q,
                                 self.NQ, self.NKV, self.HD, self.eps)
            self.attn(self.q[0], self.kc[li], self.vc[li], self.len, att2)
            L["o"].few(self.att, self.res, epi=K.EPI_RES, res=self.res)
            L["gu"].few(self.res, self.act, pair=True, out_scale=MLP_SCALE, norm=L["post_norm"], eps=self.eps)
            L["down"].few(self.act, self.res, epi=K.EPI_RES, res=self.res, out_scale=1 / MLP_SCALE)
        K.add_rmsnorm(self.res, None, self.norm, self.eps, self.hn)
        self.ctl_i[0:4].add_(1)
        self.pos.add_(self.rate)

    def _prefill(self, T):
        """The first T entries of the prefix (ids / positions in the control buffers) in one batch into cache rows
        0..T-1; hn = the final state of entry `last`. Up to few_max entries every matrix is read in its stored format
        (64 rows per launch), longer prefixes go through dense weights and cuBLAS. Of the last layer only the keys and
        values of all entries are needed; its attention output and MLP are computed for entry `last` alone."""
        ids, pos = self.ctl_i[8:8 + T], self.ctl_f[8:8 + T]
        dev, few = self.dev, T <= self.few_max
        res = self.src.index_select(0, ids)
        h = torch.empty((T, self.H), device=dev, dtype=torch.half)
        K.add_rmsnorm(res, None, self.layers[0]["in_norm"], self.eps, h)
        q = torch.empty((T, self.NQ, self.HD), device=dev, dtype=torch.half)
        a3 = torch.empty((T, self.NQ, self.HD), device=dev, dtype=torch.half)
        a = a3.view(T, self.NQ * self.HD)
        act = torch.empty((T, self.act.shape[1]), device=dev, dtype=torch.half)
        qkv = torch.empty((T, self.qkv.shape[1]), device=dev) if few else None
        rows = [slice(r, min(r + FEW, T)) for r in range(0, T, FEW)]
        for li, L in enumerate(self.layers):
            if few:
                for s in rows:
                    L["qkv"].few(h[s], qkv[s])
            else:
                qkv = L["qkv"].many(h)
            K.qk_norm_rope_store(qkv, L["q_norm"], L["k_norm"], pos, self.inv_freq, self.kc[li], self.vc[li], self.zero, q,
                                 self.NQ, self.NKV, self.HD, self.eps)
            K.causal_attention(q, self.kc[li], self.vc[li], a3, self.NKV)
            if li + 1 == self.NL:
                break
            nxt = self.layers[li + 1]["in_norm"]
            if few:
                for s in rows:
                    L["o"].few(a[s], res[s], epi=K.EPI_RES, res=res[s])
                K.add_rmsnorm(res, None, L["post_norm"], self.eps, h)
                for s in rows:
                    L["gu"].few(h[s], act[s], pair=True, out_scale=MLP_SCALE)
                    L["down"].few(act[s], res[s], epi=K.EPI_RES, res=res[s], out_scale=1 / MLP_SCALE)
                K.add_rmsnorm(res, None, nxt, self.eps, h)
            else:
                K.add_rmsnorm(L["o"].many(a), res, L["post_norm"], self.eps, h)
                K.silu_mul(L["gu"].many(h), act, MLP_SCALE)
                K.add_rmsnorm(L["down"].many(act), res, nxt, self.eps, h, 1 / MLP_SCALE)
        r1, a1 = res.index_select(0, self.last), a.index_select(0, self.last)
        L["o"].few(a1, r1, epi=K.EPI_RES, res=r1)
        L["gu"].few(r1, self.act, pair=True, out_scale=MLP_SCALE, norm=L["post_norm"], eps=self.eps)
        L["down"].few(self.act, r1, epi=K.EPI_RES, res=r1, out_scale=1 / MLP_SCALE)
        K.add_rmsnorm(r1, None, self.norm, self.eps, self.hn)

    def _ae(self):
        """The frame x (model units) -> one VibeVoice latent; the blocks' convolutions read their own last frames."""
        K.small_linear(self.x, self.ae_in_w, self.ae_in_b, self.ax)
        for b, k, off, u in zip(self.ae_blocks, self.ae_k, self.ae_off, self.ae_u):
            K.ae_pre(self.ax, b["nw"], b["nb"], self.arena[off:] if k > 1 else None, u, k)
            b["conv"].few(u, self.az16, epi=K.EPI_MISH)
            b["mix"].few(self.az16, self.ax, epi=K.EPI_RES, res=self.ax)
        K.ae_pre(self.ax, self.ae_nw, self.ae_nb, None, self.af16, 1)
        K.wide_out(self.af16, self.ae_out_w, self.ae_out_b, self.lat)

    def _vv(self):
        """One latent -> 3,200 samples; every convolution's past frames are in the arena (exact streaming)."""
        A, base, slot, eps = self.arena, self.vv_off0, self.vv_slot, self.vv_eps
        K.vv_stem(self.lat, A, base + self.vv_stem_off, slot, self.vv_u0, self.vcfg["latent"])
        xa, xb, phi = None, None, None
        for i, st in enumerate(self.vv_stages):
            T, C = st["T"], st["C"]
            if i == 0:
                xa = self.vv_stem.few(self.vv_u0, st["xa"])
            else:
                if xb is None:
                    K.vv_upcat(xa, None, None, A, base + st["up_off"], slot, st["u"])
                else:
                    K.vv_upcat(xa, xb, phi, A, base + st["up_off"], slot, st["u"])
                if st["T_in"] <= FEW:
                    xa = st["up"].few(st["u"], st["xa"].view(st["T_in"], st["s"] * C)).view(T, C)
                else:
                    xa = st["up"].many(st["u"]).view(T, C)
                xb = None
            for j, b in enumerate(st["blocks"]):
                xm = st["xm"][j % 2]
                K.vv_pre(xa, xb, phi, b["taps"], b["dwb"], b["fw"], A, base + b["off"], slot, xm, st["hn"], eps)
                if st["few"]:
                    b["l1"].few(st["hn"], st["g"], epi=K.EPI_GELU)
                    b["l2"].few(st["g"], xm, epi=K.EPI_GATED, res=xm, gate=b["phi"])
                    xa, xb = xm, None
                else:
                    xa, xb, phi = xm, b["l2"].many(F.gelu(b["l1"].many(st["hn"]))), b["phi"]
        K.vv_head(xa, xb, phi, self.vv_head_w, self.vv_head_b, A, base + self.vv_head_off, slot, self.audio)
        torch.sub(self.one, A[0:1], out=A[0:1])

    # ------------------------------------------------------------------------------------------------------- graphs
    def _graph(self, fn, *args):
        fn(*args)                                    # compile and warm outside the capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            fn(*args)
        return g

    def _capture(self):
        self.ctl_i_host[:] = 0
        self.ctl_i_host[8:] = self.R_PAD
        self.ctl_i_host[0:5] = (16, 17, self.FE0, 0, 15)
        self.ctl_i.copy_(torch.from_numpy(self.ctl_i_host))
        if not self.use_graphs:
            for fn in (self._lm_step, self._head, self._ae, self._vv):
                fn()
            self.g_lm = self.g_ae = self.g_vv = self.g_head = None
            self.arena.zero_()
            return
        self.g_lm = self._graph(self._lm_step)
        self.head_graphs[self.sampler][1] = self._graph(self._head)
        self.g_ae = self._graph(self._ae)
        self.g_vv = self._graph(self._vv)
        for T in self.buckets:
            self.graphs[T] = self._graph(self._prefill, T)
        self.arena.zero_()

    def _run_head(self):
        g = self.head_graphs[self.sampler][1]
        g.replay() if g is not None else self._head()

    def _run(self, g, fn):
        g.replay() if g is not None else fn()

    def _run_prefill(self, n):
        for T in self.buckets:
            if T >= n:
                if T in self.graphs:
                    self.graphs[T].replay()
                else:
                    self._prefill(T)
                return T
        T = min(-(-n // 64) * 64, self.MAXT)
        self._prefill(T)
        return T

    @torch.inference_mode()
    def set_first(self, first):
        """Other fixed first frames [K, 16] (model units), or None for none."""
        self.first = torch.zeros((0, self.D), device=self.dev) if first is None else torch.as_tensor(np.asarray(first, np.float32), device=self.dev).reshape(-1, self.D)
        self.KF = len(self.first)
        self.R_FIRST = self.FE0 - self.KF
        self._start_state()

    def _start_state(self):
        """The decoders' state at the start of every request: VibeVoice primed with the silence latents, then the
        fixed first frames if any (their embeddings too; they are never decoded again)."""
        self.arena.zero_()
        for p in self.prime:
            self.lat.copy_(p)
            self._run(self.g_vv, self._vv)
        for t in range(self.KF):
            self.x.copy_(self.first[t:t + 1])
            K.small_linear(self.x, self.fin0_w, self.fin0_b, self.h1, act=1)
            self.fin2.few(self.h1, self.res)
            self.src[self.R_FIRST + t] = self.res[0]
            self._run(self.g_ae, self._ae)
            self._run(self.g_vv, self._vv)
        self.arena_start.copy_(self.arena)

    # --------------------------------------------------------------------------------------------------------- text
    def _segments(self, chunks, c):
        """lm_chunks.segments on code points: a leading space after the item's first chunk, a newline after its last."""
        n = len(chunks)

        def cue(i, s):
            return np.concatenate(([32] if i > 0 else [], s, [10] if i == n - 1 else [])).astype(np.int32)
        segs = [cue(c - 1, chunks[c - 1])] if c > 0 else []
        segs.append(cue(c, chunks[c]))
        if c + 1 < n:
            s = chunks[c + 1]
            if len(s) > self.lookahead:
                sp = np.flatnonzero(s[:self.lookahead] == 32)
                cut = int(sp[-1]) if len(sp) else -1
                s = s[:cut] if cut > 0 else s[:self.lookahead]
            segs.append(cue(c + 1, s))
        return segs, (1 if c > 0 else 0)

    def _rows(self, cps):
        """Code points -> rows of the embedding source; characters the model does not know are read as spaces."""
        r = self.char_lut[np.minimum(cps, len(self.char_lut) - 1)]
        if (r < 0).any():
            self.unknown.update(chr(c) for c in cps[r < 0].tolist())
            r = np.where(r < 0, self.char_row[32], r)
        return r

    def _layout(self, segs, cur, n_prev, prev_row, k0, r_first):
        """lm_chunks.layout: fills the control buffers with the prefix (as rows of the embedding source) and its
        positions; returns (prefix length, position of the chunk's first frame)."""
        I, Fp = self.ctl_i_host, self.ctl_f_host
        starts, p = [], 1.0
        for s, seg in enumerate(segs):
            starts.append(p)
            p += max(len(seg) + self.text_off, self.rate * n_prev if s < cur else 0) + 1
        n = 8
        for s, seg in enumerate(segs):
            L = len(seg)
            I[n:n + L] = self._rows(seg)
            Fp[n:n + L] = starts[s] + np.arange(L, dtype=np.float32)
            n += L
            if s < len(segs) - 1:
                I[n], Fp[n] = self.R_TM, starts[s + 1] - 1
                n += 1
        I[n], Fp[n] = self.R_START, 0.0
        n += 1
        if cur:
            I[n:n + n_prev] = prev_row + np.arange(n_prev)
            Fp[n:n + n_prev] = starts[0] + self.rate * np.arange(n_prev, dtype=np.float32)
            n += n_prev
            I[n], Fp[n] = self.R_AM, starts[1] - 1
            n += 1
        if k0:
            I[n:n + k0] = r_first + np.arange(k0)
            Fp[n:n + k0] = starts[cur] + self.rate * np.arange(k0, dtype=np.float32)
            n += k0
        return n - 8, starts[cur]

    def _begin_chunk(self, chunks, c, n_prev):
        """Host side of a chunk: prefix layout, control values, noise. Returns (frames at most, fixed frames)."""
        n, max_f, k0 = self._layout_chunk(chunks, c, n_prev, self.FE0)
        nz = torch.randn((max_f - k0, self.D), generator=self.gen, device=self.dev)
        torch.mul(nz, self.temp, out=self.noise[:max_f - k0])
        return n, max_f, k0

    def _layout_chunk(self, chunks, c, n_prev, fe0):
        """The prefix of chunk c in the control buffers (host and device); fe0: the embedding source's row of the
        request's frame 0 (its fixed first frames right before it). Returns (prefix length, frames at most, fixed
        frames); self.chunk_start: the position of the chunk's frame 0."""
        segs, cur = self._segments(chunks, c)
        k0 = self.KF if c == 0 else 0
        r_first = fe0 - self.KF
        # frames' embeddings: the fixed ones in rows r_first.. (right before fe0), each chunk's generated ones from fe0 on
        n, start = self._layout(segs, cur, n_prev, r_first if c == 1 else fe0, k0, r_first)
        self.chunk_start = start                 # position of the chunk's frame 0 (frame t sits at start + rate * t)
        max_f =min(len(segs[cur]) + self.off, self.MAXF - 1, self.S - n - 1)     # the cache bounds a chunk's frames (never reached by real text)
        if max_f <= k0:
            raise ValueError(f"chunk {c}: {n} prefix positions do not fit the cache ({self.S})")
        T = next((b for b in self.buckets if b >= n), min(-(-n // 64) * 64, self.MAXT))
        I, Fp = self.ctl_i_host, self.ctl_f_host
        I[8 + n:8 + T], Fp[8 + n:8 + T] = self.R_PAD, 0.0
        I[0:6] = (n, n + 1, fe0, 0, n - 1, 0)
        Fp[0] = start + self.rate * k0
        self.ctl_i[:8 + T].copy_(torch.from_numpy(I[:8 + T]))
        self.ctl_f[:8 + T].copy_(torch.from_numpy(Fp[:8 + T]))
        return n, max_f, k0

    # ----------------------------------------------------------------------------------------------------- generate
    @torch.inference_mode()
    def _chunk(self, chunks, c, n_prev):
        """Prefix of chunk c in one batch, then the first stop value and frame; nothing waits for the GPU."""
        n, max_f, k0 = self._begin_chunk(chunks, c, n_prev)
        self._run_prefill(n)
        self._run_head()
        return n, max_f, k0

    @torch.inference_mode()
    def _stopped(self):
        torch.cuda.current_stream().synchronize()
        return bool(self.stop_host[0] > 0)

    @torch.inference_mode()
    def _frame(self, first_check, ch=None, t=0):
        """Decode the frame at hand, start the next state / stop value / frame, return the audio as soon as it is
        there (the GPU works on while the caller has it). first_check: the request's first frame is decoded before its
        stop value is read; None comes back if the model stopped right there (never seen). With a redraw in the
        request (ch, t: the chunk and its frame index): the audio goes to the detector, which scores the steps it makes
        final, and the decoders' state after the frame and the state that predicts the next frame are kept."""
        rd = self.rd
        if first_check:
            self.stop_host.copy_(self.stop, non_blocking=True)
        self._run(self.g_ae, self._ae)
        self._run(self.g_vv, self._vv)
        self.audio_host.copy_(self.audio, non_blocking=True)
        self.ev.record()
        if rd is not None:
            g = ch["G"] + t - ch["k0"]
            self.arena_hist[(g + 1) % RING].copy_(self.arena)
            rd.after_decode(g)
        self._run(self.g_lm, self._lm_step)
        if rd is not None:
            self.hn_hist[t + 1 - ch["k0"]].copy_(self.hn[0])
        self._run_head()
        self.ev.synchronize()
        if first_check and self.stop_host[0] > 0:
            torch.cuda.current_stream().synchronize()
            self.arena.copy_(self.arena_start)
            return None
        self.stop_host.copy_(self.stop, non_blocking=True)
        return self.audio_host.numpy().copy()

    # ------------------------------------------------------------------------- redraw: the generator's state by frame
    def snapshot(self, t, g):
        """Copies of the state just before the chunk's frame t (utterance frame g): the final state that predicts it
        and the autoencoder / VibeVoice decoders' state."""
        return {"hn": self.hn_hist[t - self.ch["k0"]].clone(), "arena": self.arena_hist[g % RING].clone()}

    def restore(self, snap, t, ch):
        """The generator just before the chunk's frame t: the snapshot's states, and the counters of the key/value
        cache (rows from frame t on are written again), frame embedding row, noise row and position."""
        j = t - ch["k0"]
        self.hn[0].copy_(snap["hn"])
        self.arena.copy_(snap["arena"])
        n = ch["n"]
        self.ctl_i[0:4].copy_(torch.tensor([n + j, n + j + 1, self.FE0 + j, j]))
        self.ctl_f[0:1].fill_(ch["start"] + self.rate * t)

    def fresh_noise(self, t, count):
        """New start noise for the chunk's frames t .. t + count - 1 (the random generator just continues)."""
        j = t - self.ch["k0"]
        count = min(count, self.MAXF - j)
        nz = torch.randn((count, self.D), generator=self.gen, device=self.dev)
        torch.mul(nz, self.temp, out=self.noise[j:j + count])
        return self.noise[j:j + count].clone()

    def head_again(self):
        self._run_head()
        self.stop_host.copy_(self.stop, non_blocking=True)

    def _next_frame(self, first, ch, t):
        """The audio of the frame at hand (_frame). A generator (that yields nothing here): the batched engine
        suspends a request at this point until its frame comes out of the batch's step."""
        return self._frame(first, ch, t)
        yield

    def _sync(self):
        torch.cuda.synchronize()

    def run_frames(self, ch, t, t_to, check=None):
        """Frames t .. t_to - 1 of the chunk with the stop rule (and the stop value at t_to read): (next frame,
        stopped, audio per frame, why). check(): called after every frame; a non-empty answer stops here (why).
        A generator (yield from): see _next_frame."""
        audio, stopped, why = [], False, None
        while t < ch["max_f"]:
            if t >= self.min_frames and self._stopped():
                stopped = True
                break
            if t >= t_to:
                break
            audio.append((yield from self._next_frame(False, ch, t)))
            t += 1
            why = check() if check is not None else None
            if why:
                stopped = t < ch["max_f"] and t >= self.min_frames and self._stopped()
                break
        return t, stopped, audio, why

    @torch.inference_mode()
    def _begin(self, seed):
        self.unknown = set()
        self.gen.manual_seed(seed)
        self.arena.copy_(self.arena_start)
        self.arena_hist[0].copy_(self.arena_start)

    @torch.inference_mode()
    def stream(self, text, seed=0, verbalize=True, chunks=None, first_max=0, redraw=None, played=None, cut=None, boundary=None):
        """Yields (start, audio): float32 audio (self.sample_rate) for the utterance's samples from `start` on, as soon
        as it exists: normally one frame (3,200 samples at 24 kHz, 6,400 at 48 kHz) right after the previous piece.
        After a redraw `start` lies before the end of what came so far: the audio from `start` on is replaced (only
        audio not yet played, by the budget rule; at 48 kHz from up to self.delay samples before the redrawn frame).
        chunks: code-point arrays instead of text (no front end). first_max > 0: a first chunk of about that
        many characters (faster first audio for long texts, but shorter than the chunks the model usually reads).
        redraw: None = the engine's setting. played: a function giving the samples the player has consumed so far
        (None: a player that starts with the first piece and plays at real time). cut: None = the engine's setting;
        on, the request starts at its first sound and every position (pieces, played) is in player samples, player
        sample p = utterance sample p + self.stats["cut"]. boundary (instead of played): a function giving the samples
        that can no longer be replaced, for a caller that holds audio back before it passes it on; a redraw may then
        replace from any point at or after it (nothing is added for the time the redraw takes, the hold-back is the
        margin). self.stats holds the request's numbers afterwards (redraw counters in self.stats["redraw"]; t_first =
        the first piece out, t_first_frame = the first frame decoded)."""
        if played is not None and boundary is not None:
            raise ValueError("played or boundary, not both")
        t0 = time.perf_counter()
        if chunks is None:
            chunks = self.front.chunks(text, verbalize, first_max)
        st = self.stats = {"chunks": len(chunks), "chars": int(sum(len(c) for c in chunks)), "frames": [], "t_front": time.perf_counter() - t0,
                           "t_first": None, "t_first_frame": None, "prefix": [], "t_chunk_first": [], "gap_max": 0.0}
        if not chunks:
            return
        self._begin(seed)
        use = self.judge is not None and (self.redraw if redraw is None else redraw)
        up = Up48(self.bwe) if self.bwe is not None else None
        if up is not None:      # the redraw counts 24 kHz samples; a replacement at 48 kHz starts up to DELAY earlier
            if boundary is not None:
                b48 = boundary
                boundary = lambda: b48() / 2 + up.lag() - 0.5                # noqa: E731
            else:
                p48 = played or (lambda: (time.perf_counter() - rd.t_first) * self.sample_rate if rd.t_first else 0.0)   # noqa: E731
                played = lambda: p48() / 2 + up.lag()                        # noqa: E731
        rd = self.rd = Redraw(self, sum(min(len(c) + self.off, self.MAXF - 1) for c in chunks), played, boundary) if use else None
        st["cut"] = 0
        try:
            pieces = self._generate(chunks, st, rd, t0)
            if self.lead_cut if cut is None else cut:
                def on_cut(c):
                    st["cut"] = c * self.sample_rate // SR
                    if rd is not None:
                        rd.offset = c                    # the redraw's swap points and playback checks in player samples
                pieces = cut_lead(pieces, on_cut)
            if up is not None:
                pieces = up.run(pieces)
            for start, audio in pieces:
                if st["t_first"] is None:
                    now = time.perf_counter()
                    st["t_first"] = now - t0
                    if rd is not None:
                        rd.t_first = now                 # the simulated player starts with the first piece
                yield start, audio * OUT_GAIN            # the output gain on the pieces only (the redraw keeps its audio)
        finally:
            self.rd, self.last_redraw = None, rd

    def _generate(self, chunks, st, rd, t0):
        n_prev, t_last, G = 0, None, 0
        for c in range(len(chunks)):
            n, max_f, k0 = self._chunk(chunks, c, n_prev)
            ch = self.ch = {"c": c, "n": n, "max_f": max_f, "k0": k0, "G": G, "start": self.chunk_start, "last": c == len(chunks) - 1}
            st["prefix"].append(n)
            if rd is not None:
                self.hn_hist[0].copy_(self.hn[0])
            t, stopped, fresh, held = k0, False, True, 0.0
            while True:
                first = c == 0 and t == k0 and t >= self.min_frames      # decoded before its stop value is read
                if not stopped and t < max_f and t >= self.min_frames and not first and self._stopped():
                    stopped = True
                if rd is not None:
                    pieces, t, stopped = yield from rd.point(ch, t, stopped)
                    for piece in pieces:
                        fresh = True
                        yield piece
                if stopped or t >= max_f:
                    break
                out = yield from self._next_frame(first, ch, t)
                if out is None:
                    break
                now = time.perf_counter()
                if st["t_first_frame"] is None:
                    st["t_first_frame"] = now - t0
                if t == k0:
                    st["t_chunk_first"].append(now - t0)
                if t_last is not None:
                    st["gap_max"] = max(st["gap_max"], now - t_last)
                    if not fresh and held < 1e-3:          # the redraw budget's time per frame: only when the caller
                        self.frame_s += 0.05 * (now - t_last - self.frame_s)   # gave the frame back at once
                t_last, fresh = now, False
                yield HOP * (G + t - k0), out
                held = time.perf_counter() - now           # how long the caller kept the frame (a player may wait)
                t += 1
            n_prev = t
            st["frames"].append(t - k0)
            G += t - k0
        self._sync()
        st["t_total"] = time.perf_counter() - t0
        st["unknown"] = "".join(sorted(self.unknown))
        if rd is not None:
            st["redraw"] = rd.stats

    @torch.inference_mode()
    def profile_first(self, text, seed=0, verbalize=True, first_max=0) -> dict:
        """The way to the first audio stage by stage (seconds), waiting for the GPU after each stage; the sum is
        therefore a little above the time to first audio that stream() reaches."""
        sync, now, r = torch.cuda.synchronize, time.perf_counter, {}
        sync()
        t0 = now()
        chunks = self.front.chunks(text, verbalize, first_max)
        r["verbalizer"], r["chunker"] = float(self.front.times[0]), float(self.front.times[1])
        t1 = now()
        r["front_end_call"] = t1 - t0                      # the two above plus the call into the library
        self._begin(seed)
        n, max_f, k0 = self._begin_chunk(chunks, 0, 0)
        sync()
        t2 = now()
        r["prepare"] = t2 - t1                             # prefix layout, two host copies, noise, decoder start state
        self._run_prefill(n)
        sync()
        t3 = now()
        r["prefill"] = t3 - t2
        self._run_head()
        sync()
        t4 = now()
        r["head"] = t4 - t3
        self._run(self.g_ae, self._ae)
        sync()
        t5 = now()
        r["autoencoder"] = t5 - t4
        self._run(self.g_vv, self._vv)
        self.audio_host.copy_(self.audio)
        self.audio_host.numpy().copy()
        t6 = now()
        r["vibevoice"] = t6 - t5
        r["total"], r["prefix"] = t6 - t0, n
        return r

    @torch.inference_mode()
    def trace(self, chunks, seed=0, forced=None):
        """The same generation, slowly, keeping everything: per chunk the final states h, stop values and sampled
        frames x; all latents and the audio. forced: per chunk the frames to feed back instead of the own ones (the
        float reference's: every state is then comparable)."""
        self._begin(seed)
        out, lats, audio, n_prev = [], [], [], 0
        for c in range(len(chunks)):
            n, max_f, k0 = self._begin_chunk(chunks, c, n_prev)
            self._run_prefill(n)
            hs, stops, xs = [], [], []
            t = k0
            while t < max_f:
                self._run_head()
                hs.append(self.hn[0].clone()); stops.append(self.stop[0].clone())
                if (t >= len(forced[c])) if forced is not None else (t >= self.min_frames and float(self.stop) > 0):
                    break
                xs.append(self.x[0].clone())
                if forced is not None:
                    self.x.copy_(torch.as_tensor(forced[c][t], device=self.dev)[None])
                self._run(self.g_ae, self._ae)
                self._run(self.g_vv, self._vv)
                lats.append(self.lat.clone()); audio.append(self.audio.clone())
                self._run(self.g_lm, self._lm_step)
                t += 1
            n_prev = t
            st = lambda v, w: torch.stack(v).cpu().numpy() if v else np.zeros((0, w), np.float32)
            out.append({"h": st(hs, self.H), "stop": torch.stack(stops).cpu().numpy(), "x": st(xs, self.D), "n": n, "k0": k0})
        return out, torch.stack(lats).cpu().numpy(), torch.cat(audio).cpu().numpy()

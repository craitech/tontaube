"""TontaubeV1.1 Flash on one GPU for many requests at once (continuous batching).

    from flash.gpu.batch import BatchEngine
    eng = BatchEngine("int4", max_batch=64)
    for start, audio in eng.stream("Some text."):     # as Engine.stream, from any number of threads at once
        ...
    req = eng.submit("Some text.", seed=1)           # the same request as a handle: iterate it; req.stats, req.close()

One worker thread drives the GPU. Requests are admitted in (priority, arrival) order when a slot and the key/value
pages for their longest chunk are free. Each request runs the single engine's own generation code (Engine._generate
and the redraw) as a coroutine that stops wherever it needs a frame; every step of the worker gives all of them their
frame at once: the autoencoder and VibeVoice decoders, the backbone step and the flow head of all active requests run
as two replays of CUDA graphs captured per batch size, and the 48 kHz upsampler computes the step's common pieces in
one more. Per request: a chunk's text prefix (the single engine's prefill graphs into a staging cache, its rows then
copied into the request's pages), the stop decision and the redraw's rewinds (they restore the request's slot; its
frames are drawn in the batch like any other). Every request has its own random generator and temperature, so its
audio does not depend on the others in the batch (up to float rounding of other batch sizes).
"""
import heapq
import itertools
import queue
import threading
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from . import kernels as K
from ..common.stream import OUT_GAIN, SR, LeadCut
from .bwe import Stream as Up48
from .detector import CONTEXT, FRAME, HOP as STEP, LAG
from .engine import FEW, MLP_SCALE, RING, Engine
from .redraw import Redraw

PAGE = 64                      # key/value rows per page


def graph_batch_sizes(max_batch):
    """Batch sizes with captured graphs (as the V1.1 lite engine): every size up to 8, then 12 .. 64, and max_batch."""
    sizes = list(range(1, min(max_batch, 8) + 1)) + [s for s in (12, 16, 24, 32, 48, 64) if s < max_batch]
    return sizes if max_batch in sizes else sizes + [max_batch]


def _few_rows(lin, x, out, epi=K.EPI_NONE, res=None, gate=None, pair=False, out_scale=1.0):
    """lin.few for any number of rows (64 per launch)."""
    M = x.shape[0]
    for r in range(0, M, FEW):
        s = slice(r, min(r + FEW, M))
        lin.few(x[s], out[s], epi=epi, res=None if res is None else res[s], gate=gate[s] if gate is not None and gate.dim() == 2 else gate,
                pair=pair, out_scale=out_scale)
    return out


class Request:
    """A request's pieces (start, audio) as an iterator, from any thread; stats: its numbers (as Engine.stats);
    close(): cancel it."""

    def __init__(self, eng, chunks, t0, seed, temp, redraw, played, cut, boundary, priority):
        self.eng, self.chunks, self.t0, self.seed, self.temp = eng, chunks, t0, seed, temp
        self.redraw, self.played, self.cut, self.boundary, self.priority = redraw, played, cut, boundary, priority
        self.q = queue.SimpleQueue()
        self.cancelled = self.done = False
        self.stats = {}

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        item = self.q.get()
        if item is None or isinstance(item, BaseException):
            self.done = True
            if item is None:
                self.eng.stats = self.stats
                raise StopIteration
            raise item
        return item

    def close(self):
        if not self.done:
            self.cancelled = self.done = True
            with self.eng._cond:
                self.eng._cond.notify()

    def __del__(self):
        self.cancelled = True


class _Req:
    """A request inside the batch: what the single engine's generation code and the redraw read and write, here in
    the request's slot of the batched state."""

    _generate = Engine._generate
    run_frames = Engine.run_frames

    def __init__(self, eng, req, slot):
        self.e, self.slot = eng, slot
        self.dev, self.D, self.MAXF, self.KF, self.min_frames = eng.dev, eng.D, eng.MAXF, eng.KF, eng.min_frames
        self.judge, self.cutoff, self.max_takes, self.margin_ms = eng.judge, eng.cutoff, eng.max_takes, eng.margin_ms
        self.FE0 = eng.FE0 + slot * eng.SLOT_ROWS
        self.hn, self.hn_hist, self.noise = eng.hns[slot:slot + 1], eng.hn_hists[slot], eng.noises[slot]
        self.gen = torch.Generator(device=eng.dev)
        self.gen.manual_seed(req.seed)
        self.temp = req.temp
        self.unknown, self._rd_audio, self.rd, self.ch, self.G = set(), None, None, None, 0
        self.stop_val = None           # the stop value of the frame at hand on the host (None: read it from the GPU)
        self.row = None                # the request's row in the last step
        self.audio, self.frame_at = None, None
        self.judge_later = eng.judge_later
        self.pm, self.pm_n, self.pm_to = None, 0, 0   # the detector's pitch map of the utterance: rows valid / final

    frame_s = property(lambda self: self.e.frame_s, lambda self, v: setattr(self.e, "frame_s", v))

    def _chunk(self, chunks, c, n_prev):
        return self.e._prefill_slot(self, chunks, c, n_prev)

    def _next_frame(self, first, ch, t):
        if first and self._stopped():            # (the single engine decodes this frame before it reads the stop)
            return None
        self.frame_at = (ch, t)
        return (yield self)                      # the worker's next step brings the frame's audio

    def _stopped(self):
        if self.stop_val is None:
            self.stop_val = float(self.e.stops[self.slot])
        return self.stop_val > 0

    def _sync(self):
        pass

    # the redraw's access to the generator (Engine.snapshot ... head_again for this slot)
    def snapshot(self, t, g):
        return {"hn": self.hn_hist[t - self.ch["k0"]].clone(), "arena": self.e.arena_hists[self.slot, g % RING].clone()}

    def restore(self, snap, t, ch):
        e, s, j = self.e, self.slot, t - ch["k0"]
        self.hn[0].copy_(snap["hn"])
        e.arenas[s].copy_(snap["arena"])
        e._set_slot(s, ch["n"] + j + 1, self.FE0 + j, j, ch["G"], ch["start"] + e.rate * t)

    def fresh_noise(self, t, count):
        j = t - self.ch["k0"]
        count = min(count, self.MAXF - j)
        nz = torch.randn((count, self.D), generator=self.gen, device=self.dev)
        torch.mul(nz, self.temp, out=self.noise[j:j + count])
        return self.noise[j:j + count].clone()

    def head_again(self):
        self.e._head_one(self.slot)
        self.stop_val = None


class BatchEngine(Engine):
    def __init__(self, variant="int4", max_batch=64, kv_rows=None, **kw):
        """As Engine (the same options; graphs always on), plus max_batch: requests generated at once; kv_rows: rows
        of the key/value pool shared by them (default 640 per request, at least one full sequence; a request takes the
        rows of its longest chunk, e.g. about 300 for a sentence; more wait)."""
        if not kw.get("graphs", True):
            raise ValueError("BatchEngine needs CUDA graphs")
        self.max_batch = int(max_batch)
        self.sizes = graph_batch_sizes(self.max_batch)
        self.kv_rows = kv_rows or 640 * self.max_batch
        self._cond = threading.Condition()
        self._pending, self._active, self._arrival = [], [], itertools.count()
        self._running, self._fatal, self._thread = True, None, None
        self._front_lock = threading.Lock()
        self._asks, self._ending, self._judge_asks = [], [], []
        self.steps, self.wait_s = 0, 0.0         # steps run; seconds the worker waited for them
        self.pm_reach = 0                         # set with the detector: a pitch-map row's samples end at 120 t + this
        self._built = False
        self.bwe_half = True                     # the upsampler's matrix products in fp16 (bwe.BWE)
        super().__init__(variant, **kw)
        self._built = True

    # ---------------------------------------------------------------------------------------------------- buffers
    def _buffers(self, S):
        super()._buffers(S)                      # one request's buffers: the prefill's staging cache, the start state
        dev, H, D, MB = self.dev, self.H, self.D, self.max_batch
        z = lambda *shape, dtype=torch.float32: torch.zeros(shape, device=dev, dtype=dtype)   # noqa: E731
        R = self.SCRATCH = MB                    # slots: one per request; the last one takes the rows that pad a batch
        R += 1
        # embedding source: the shared rows, then per slot 16 rows (fixed first frames) and MAXF (the frames)
        self.SLOT_ROWS = 16 + self.MAXF
        base = self.FE0 - 16
        src = z(base + R * self.SLOT_ROWS, H)
        src[:base].copy_(self.src[:base])
        self.src = src
        # key/value pool in pages (the last page: the padding rows'), the pages of every slot; at least one request's
        # longest chunk (S rows) fits, however small max_batch is
        self.ppr = -(-S // PAGE)
        self.n_pages = max(-(-self.kv_rows // PAGE), self.ppr)
        rows = (self.n_pages + 1) * PAGE
        self.kp = z(self.NL, rows, self.NKV, self.HD, dtype=torch.half)
        self.vp = z(self.NL, rows, self.NKV, self.HD, dtype=torch.half)
        self.table = torch.full((R, self.ppr), self.n_pages, device=dev, dtype=torch.int32)
        self.pattn = K.PagedDecodeAttention(self.NQ, self.NKV, self.HD, S, max(self.sizes), dev, PAGE)
        self.arange_s = torch.arange(S, device=dev)
        # per slot: counters (key/value length, frame-embedding row, frame index in the chunk, the chunk's first
        # frame in the utterance), position, frame x, final state, stop value, noise, decoder state, redraw rings
        self.s_len, self.s_fe = torch.ones(R, device=dev, dtype=torch.long), z(R, dtype=torch.long)
        self.s_ti, self.s_g, self.s_pos = z(R, dtype=torch.long), z(R, dtype=torch.long), z(R)
        self.s_fe[self.SCRATCH] = self.FE0 + self.SCRATCH * self.SLOT_ROWS
        self.xs, self.hns, self.stops, self.noises = z(R, D), z(R, H), z(R), z(R, self.MAXF, D)
        A = self.A = self.arena.numel()
        self.arenas, self.arena_hists, self.hn_hists = z(R, A), z(R, RING, A), z(R, self.MAXF + 1, H)
        # a batch: its rows' slots (padding: the scratch slot) and which rows move on
        G = max(self.sizes)
        self.sel, self.adv = torch.full((G,), self.SCRATCH, device=dev, dtype=torch.long), z(G, dtype=torch.long)
        self.sel1 = torch.full((1,), self.SCRATCH, device=dev, dtype=torch.long)
        self.sel_host, self.adv_host = np.full(G, self.SCRATCH, np.int64), np.zeros(G, np.int64)
        self.batch_key = None
        # the step's buffers, G rows
        self.bx, self.bxm = z(G, D), z(G, D)
        self.bax, self.baz16, self.baf16, self.blat = z(G, self.AW), z(G, 2 * self.AW, dtype=torch.half), z(G, self.AW, dtype=torch.half), z(G, self.vcfg["latent"])
        self.bae_u = [z(G, k * self.AW, dtype=torch.half) for k in self.ae_k]
        self.bvv_u0 = z(G, 7 * self.vcfg["latent"], dtype=torch.half)
        self.bvv = []
        for st in self.vv_stages:
            T, C = st["T"], st["C"]
            b = {"xm": (z(G * T, C), z(G * T, C)), "hn": z(G * T, C, dtype=torch.half)}
            b["xa"] = z(G * T, C) if st["up"] is None or st["T_in"] <= FEW else None
            b["g"] = z(FEW, 4 * C, dtype=torch.half) if T <= FEW else None
            if st["up"] is not None:
                b["u"] = z(G * st["T_in"], st["up"].k, dtype=torch.half)
            self.bvv.append(b)
        self.baudio = z(G, self.vv_stages[-1]["T"])
        self.bh1, self.bres, self.bh16 = z(G, H, dtype=torch.half), z(G, H), z(G, H, dtype=torch.half)
        self.bqkv, self.bq = z(G, self.qkv.shape[1]), z(G, self.NQ, self.HD, dtype=torch.half)
        self.batt, self.bact, self.bhn = z(G, self.NQ * self.HD, dtype=torch.half), z(G, self.cfg["inter"], dtype=torch.half), z(G, H)
        self.by16, self.bstop, self.bc0 = z(G, self.FW, dtype=torch.half), z(G), z(G, self.FW)
        self.bhx, self.bz16 = z(G, self.FW), z(G, self.FW, dtype=torch.half)
        self.audio_host = torch.zeros(G, self.baudio.shape[1], dtype=torch.float32).pin_memory()
        self.stop_host = torch.zeros(G, dtype=torch.float32).pin_memory()
        self.step_ev = torch.cuda.Event()
        self.free_slots = list(range(self.max_batch - 1, -1, -1))
        self.free_pages = list(range(self.n_pages - 1, -1, -1))

    def _head_buffers(self):
        """The batched head's pass buffers for the sampler in use (G rows of P passes)."""
        G, P = max(self.sizes), self.P
        if getattr(self, "_hb_key", None) != (G, P):
            self.bws = torch.zeros((G * P, self.FW), device=self.dev, dtype=torch.half)
            self.bmodb = torch.zeros((G * P, self.mods.n), device=self.dev)
            self._hb_key = (G, P)

    def set_sampler(self, solver="midpoint", passes=6, temp=0.8):
        """The temperature is per request (stream(temp=...)); this sets the default. Solver and passes are fixed
        when the engine is built (its graphs)."""
        if self._built and (solver, passes if solver == "euler" else 2 * max(1, passes // 2)) != self.sampler:
            raise ValueError("BatchEngine: solver and passes are set when it is built")
        super().set_sampler(solver, passes, temp)

    # ------------------------------------------------------------------------------------------------- the step
    def _dec_b(self, G):
        """The decoders for rows 0..G-1: each row's frame x (slot sel[row]) -> its audio; the decoders' state after
        the frame goes to the slot's redraw ring."""
        s, AS = self.sel[:G], self.A
        x = self.bx[:G]
        torch.index_select(self.xs, 0, s, out=x)
        ax = self.bax[:G]
        K.small_linear(x, self.ae_in_w, self.ae_in_b, ax)
        for b, k, off, u in zip(self.ae_blocks, self.ae_k, self.ae_off, self.bae_u):
            K.ae_pre(ax, b["nw"], b["nb"], self.arenas[:, off:] if k > 1 else None, u[:G], k, sel=s if k > 1 else None, astride=AS)
            _few_rows(b["conv"], u[:G], self.baz16[:G], epi=K.EPI_MISH)
            _few_rows(b["mix"], self.baz16[:G], ax, epi=K.EPI_RES, res=ax)
        K.ae_pre(ax, self.ae_nw, self.ae_nb, None, self.baf16[:G], 1)
        K.wide_out(self.baf16[:G], self.ae_out_w, self.ae_out_b, self.blat[:G])
        # VibeVoice
        A, base, slot, eps = self.arenas, self.vv_off0, self.vv_slot, self.vv_eps
        K.vv_stem(self.blat[:G], A, base + self.vv_stem_off, slot, self.bvv_u0[:G], self.vcfg["latent"], sel=s, astride=AS)
        xa, xb, phi = None, None, None
        for i, (st, bb) in enumerate(zip(self.vv_stages, self.bvv)):
            T, C = st["T"], st["C"]
            M = G * T
            if i == 0:
                xa = _few_rows(self.vv_stem, self.bvv_u0[:G], bb["xa"][:G])
            else:
                u = bb["u"][:G * st["T_in"]]
                K.vv_upcat(xa, xb, phi, A, base + st["up_off"], slot, u, B=G, sel=s, astride=AS)
                if G * st["T_in"] <= FEW:
                    xa = st["up"].few(u, bb["xa"][:M].view(G * st["T_in"], st["s"] * C)).view(M, C)
                else:
                    xa = st["up"].many(u).view(M, C)
                xb = None
            for j, b in enumerate(st["blocks"]):
                xm, hn = bb["xm"][j % 2][:M], bb["hn"][:M]
                K.vv_pre(xa, xb, phi, b["taps"], b["dwb"], b["fw"], A, base + b["off"], slot, xm, hn, eps, B=G, sel=s, astride=AS)
                if M <= FEW:
                    b["l1"].few(hn, bb["g"][:M], epi=K.EPI_GELU)
                    b["l2"].few(bb["g"][:M], xm, epi=K.EPI_GATED, res=xm, gate=b["phi"])
                    xa, xb = xm, None
                else:
                    xa, xb, phi = xm, b["l2"].many(F.gelu(b["l1"].many(hn))), b["phi"]
        K.vv_head(xa, xb, phi, self.vv_head_w, self.vv_head_b, A, base + self.vv_head_off, slot, self.baudio[:G], B=G, sel=s, astride=AS)
        K.vv_flip(A, s, G)
        ring = s * RING + (self.s_g.index_select(0, s) + self.s_ti.index_select(0, s) + 1) % RING
        self.arena_hists.view(-1, AS).index_copy_(0, ring, A.index_select(0, s))

    def _lm_b(self, G):
        """The backbone step for rows 0..G-1: each row's frame x at its slot's next cache row -> the slot's final state;
        the counters of the real rows move on."""
        s, adv = self.sel[:G], self.adv[:G]
        res = self.bres[:G]
        K.small_linear(self.bx[:G], self.fin0_w, self.fin0_b, self.bh1[:G], act=1)
        _few_rows(self.fin2, self.bh1[:G], res)
        self.src.index_copy_(0, self.s_fe.index_select(0, s), res)
        ln = self.s_len.index_select(0, s)
        pos = self.s_pos.index_select(0, s)
        tab = self.table.index_select(0, s)
        r = ln - 1
        rows = tab.gather(1, (r // PAGE)[:, None])[:, 0].long() * PAGE + r % PAGE
        h16, qkv, q, att, act = self.bh16[:G], self.bqkv[:G], self.bq[:G], self.batt[:G], self.bact[:G]
        for li, L in enumerate(self.layers):
            if G == 1:                           # one row: the RMSNorms inside the linears, as the single engine
                L["qkv"].few(res, qkv, norm=L["in_norm"], eps=self.eps)
            else:
                K.add_rmsnorm(res, None, L["in_norm"], self.eps, h16)
                _few_rows(L["qkv"], h16, qkv)
            K.qk_norm_rope_store(qkv, L["q_norm"], L["k_norm"], pos, self.inv_freq, self.kp[li], self.vp[li], rows, q,
                                 self.NQ, self.NKV, self.HD, self.eps, per_token=True)
            self.pattn(q, self.kp[li], self.vp[li], tab, ln, att)
            _few_rows(L["o"], att, res, epi=K.EPI_RES, res=res)
            if G == 1:
                L["gu"].few(res, act, pair=True, out_scale=MLP_SCALE, norm=L["post_norm"], eps=self.eps)
            else:
                K.add_rmsnorm(res, None, L["post_norm"], self.eps, h16)
                _few_rows(L["gu"], h16, act, pair=True, out_scale=MLP_SCALE)
            _few_rows(L["down"], act, res, epi=K.EPI_RES, res=res, out_scale=1 / MLP_SCALE)
        hn = self.bhn[:G]
        K.add_rmsnorm(res, None, self.norm, self.eps, hn)
        self.hns.index_copy_(0, s, hn)
        self.hn_hists.view(-1, self.H).index_copy_(0, s * (self.MAXF + 1) + self.s_ti.index_select(0, s) + 1, hn)
        for c in (self.s_len, self.s_fe, self.s_ti):
            c.index_add_(0, s, adv)
        self.s_pos.index_add_(0, s, adv.float() * self.rate)

    def _head_b(self, G, s):
        """The flow head for rows 0..G-1 (slots s): the slot's final state -> its stop value and next frame x (start
        noise: row ti of the slot's noise)."""
        hb, FW, P = self.hb, self.FW, self.P
        hn = self.bhn[:G]
        torch.index_select(self.hns, 0, s, out=hn)
        y16, c0, stop = self.by16[:G], self.bc0[:G], self.bstop[:G]
        K.head_pre(hn, self.cn_w, self.cn_b, self.stop_w, self.stop_b, y16, stop)
        _few_rows(self.c_in, y16, c0)
        ws, modb = self.bws[:G * P], self.bmodb[:G * P]
        K.head_ws(c0, hb["tm"], ws)
        _few_rows(self.mods, ws, modb)
        mods = modb.view(G, P, -1)
        x, xm, hx, z16 = self.bx[:G], self.bxm[:G], self.bhx[:G], self.bz16[:G]
        torch.index_select(self.noises.view(-1, self.D), 0, s * self.MAXF + self.s_ti.index_select(0, s), out=x)
        nb3, coef = 3 * FW, 1.0 / self.steps
        for p in range(P):
            mid = (not self.euler) and p % 2 == 1
            mod = mods[:, p]
            K.head_in(xm if mid else x, self.x_in_w, self.x_in_b, mod, hx, y16)
            for k, (la, lb) in enumerate(self.mlp):
                if k:
                    K.head_mid(hx, mod[:, k * nb3:], y16)
                _few_rows(la, y16, z16, epi=K.EPI_MISH)
                _few_rows(lb, z16, hx, epi=K.EPI_GATED, res=hx, gate=mods[:, p, k * nb3 + 2 * FW:(k + 1) * nb3])
            if self.euler or mid:
                K.head_out(hx, mod[:, self.NB * nb3:], self.out_w, self.out_b, x, x, coef)
            else:
                K.head_out(hx, mod[:, self.NB * nb3:], self.out_w, self.out_b, x, xm, coef / 2)
        self.xs.index_copy_(0, s, x)
        self.stops.index_copy_(0, s, stop)

    def _lm_head_b(self, G):
        self._lm_b(G)
        self._head_b(G, self.sel[:G])

    def _capture(self):
        for L in self.layers:                    # dense copies of what cuBLAS reads (prefill, the larger decoder stages)
            for k in ("qkv", "o", "gu", "down"):
                L[k].keep_dense = True
        for st in self.vv_stages:
            for lin in [st["up"]] + [b[k] for b in st["blocks"] for k in ("l1", "l2")]:
                if lin is not None:
                    lin.keep_dense = True
        super()._capture()                       # the single request's graphs (prefill, start state)
        self._head_buffers()
        self.g_dec, self.g_lm_head = {}, {}
        for G in sorted(self.sizes, reverse=True):
            self.g_dec[G] = self._graph(self._dec_b, G)
            self.g_lm_head[G] = self._graph(self._lm_head_b, G)
        self.g_head1 = self._graph(self._head_b, 1, self.sel1)
        self.arenas.zero_()

    def _start_state(self):
        super()._start_state()
        if self.KF:                              # the fixed first frames' embeddings in every slot
            first = self.src[self.R_FIRST:self.FE0]
            for s in range(1, self.SCRATCH + 1):
                self.src[self.R_FIRST + s * self.SLOT_ROWS:self.FE0 + s * self.SLOT_ROWS].copy_(first)

    def _warm_up(self):
        if self.judge is not None:
            self.pm_reach = self.judge.pm.centre - self.judge.pm.win - self.judge.pm.tmax
        torch.cuda.synchronize()
        self._thread = threading.Thread(target=self._worker, name="flash-batch", daemon=True)
        self._thread.start()
        super()._warm_up()

    # ------------------------------------------------------------------------------------------- per request ops
    def _set_slot(self, s, length, fe, ti, g, pos):
        self.s_len[s] = length
        self.s_fe[s] = fe
        self.s_ti[s] = ti
        self.s_g[s] = g
        self.s_pos[s] = pos

    def _head_one(self, s):
        self.sel1.fill_(s)
        self.g_head1.replay()

    def _prefill_slot(self, r, chunks, c, n_prev):
        """Chunk c's prefix for request r (slot r.slot): the single engine's prefill into the staging cache, its rows
        into the slot's pages, the slot's counters and noise, the first frame from the head."""
        self.unknown = r.unknown
        n, max_f, k0 = self._layout_chunk(chunks, c, n_prev, r.FE0)
        r.chunk_start = self.chunk_start
        if c:
            r.G += n_prev - (self.KF if c == 1 else 0)
        nz = torch.randn((max_f - k0, self.D), generator=r.gen, device=self.dev)
        torch.mul(nz, r.temp, out=r.noise[:max_f - k0])
        self._run_prefill(n)
        t = self.arange_s[:n]
        rows = self.table[r.slot].long()[t // PAGE] * PAGE + t % PAGE
        self.kp.index_copy_(1, rows, self.kc[:, :n])
        self.vp.index_copy_(1, rows, self.vc[:, :n])
        self.hns[r.slot].copy_(self.hn[0])
        self.hn_hists[r.slot, 0].copy_(self.hn[0])
        self._set_slot(r.slot, n + 1, r.FE0, 0, r.G, self.chunk_start + self.rate * k0)
        self._head_one(r.slot)
        r.stop_val = None
        return n, max_f, k0

    def _rows_needed(self, chunks):
        """Key/value rows of the request's longest chunk (prefix + frames at most)."""
        need, n_prev = 0, 0
        for c in range(len(chunks)):
            segs, cur = self._segments(chunks, c)
            k0 = self.KF if c == 0 else 0
            n = sum(len(s) for s in segs) + len(segs) + (n_prev + 1 if cur else 0) + k0
            max_f = min(len(segs[cur]) + self.off, self.MAXF - 1, self.S - n - 1)
            need, n_prev = max(need, n + max_f), max_f
        return need

    # -------------------------------------------------------------------------------------------------- the API
    def submit(self, text, seed=0, verbalize=True, chunks=None, first_max=0, redraw=None, played=None, cut=None, boundary=None,
               temp=None, priority=0) -> Request:
        """Takes a request; returns its handle (iterate it for the pieces). Options as Engine.stream; temp: this
        request's temperature (None: the engine's); priority: lower is admitted first."""
        if played is not None and boundary is not None:
            raise ValueError("played or boundary, not both")
        t0 = time.perf_counter()
        if chunks is None:
            with self._front_lock:
                chunks = self.front.chunks(text, verbalize, first_max)
        req = Request(self, chunks, t0, seed, self.temp if temp is None else float(temp), redraw, played, cut, boundary, int(priority))
        req.stats = {"chunks": len(chunks), "chars": int(sum(len(c) for c in chunks)), "frames": [], "t_front": time.perf_counter() - t0,
                     "t_first": None, "t_first_frame": None, "prefix": [], "t_chunk_first": [], "gap_max": 0.0, "cut": 0}
        if not chunks:
            req.q.put(None)
            return req
        req.pages = -(-self._rows_needed(chunks) // PAGE)
        if req.pages > self.n_pages:
            raise ValueError(f"the text needs {req.pages * PAGE} key/value rows, the pool has {self.n_pages * PAGE}")
        with self._cond:
            if self._fatal is not None:
                raise RuntimeError("the batch engine has stopped") from self._fatal
            req.arrival = next(self._arrival)
            heapq.heappush(self._pending, (req.priority, req.arrival, req))
            self._cond.notify()
        return req

    def stream(self, text, seed=0, verbalize=True, chunks=None, first_max=0, redraw=None, played=None, cut=None, boundary=None,
               temp=None, priority=0):
        """As Engine.stream (the request's numbers in .stats of the returned handle, and in self.stats when it ends)."""
        return self.submit(text, seed, verbalize, chunks, first_max, redraw, played, cut, boundary, temp, priority)

    def shutdown(self):
        with self._cond:
            self._running = False
            self._cond.notify()
        if self._thread is not None:
            self._thread.join(timeout=10)

    close = shutdown

    # ------------------------------------------------------------------------------------------------ the worker
    def _worker(self):
        try:
            self._loop()
            exc = RuntimeError("the batch engine was shut down")
        except BaseException as err:             # never leave a caller waiting
            traceback.print_exc()
            exc = err
        with self._cond:
            self._fatal = exc
            failed = [r for _, _, r in self._pending] + self._active
            self._pending, self._active = [], []
        for r in failed:
            r.q.put(exc)

    def _loop(self):
        if self.dev.index is not None:
            torch.cuda.set_device(self.dev)
        stream, self.side = torch.cuda.Stream(self.dev), torch.cuda.Stream(self.dev)
        with torch.cuda.stream(stream), torch.inference_mode():
            while True:
                if not self._active:
                    self._flush()
                with self._cond:
                    while self._running and not self._pending and not self._active:
                        self._cond.wait()
                    if not self._running:
                        return
                    admitted = self._admit()
                for r in admitted:
                    self._start(r)
                for r in [r for r in self._active if r.cancelled]:
                    self._end(r, close=True)
                rows = list(self._active)
                if rows:
                    self._launch(rows)           # the step runs on the GPU while the last step's pieces are upsampled
                self._flush()
                if rows:
                    self._collect(rows)
                elif not admitted:
                    with self._cond:
                        if self._running and not self._active:
                            self._cond.wait(timeout=0.05)

    def _admit(self):
        out = []
        while self._pending and self.free_slots:
            r = self._pending[0][2]
            if r.cancelled:
                heapq.heappop(self._pending)
                continue
            if r.pages > len(self.free_pages):
                break
            heapq.heappop(self._pending)
            r.slot = self.free_slots.pop()
            r.page_ids = [self.free_pages.pop() for _ in range(r.pages)]
            out.append(r)
        return out

    def _start(self, r):
        """Request r into its slot (the setup of Engine.stream), its coroutine up to its first frame."""
        s, st = r.slot, r.stats
        tab = np.full(self.ppr, self.n_pages, np.int32)
        tab[:len(r.page_ids)] = r.page_ids
        self.table[s].copy_(torch.from_numpy(tab))
        self.arenas[s].copy_(self.arena_start)
        self.arena_hists[s, 0].copy_(self.arena_start)
        v = r.view = _Req(self, r, s)
        use = self.judge is not None and (self.redraw if r.redraw is None else r.redraw)
        up = r.up = Up48(self.bwe) if self.bwe is not None else None
        played, boundary = r.played, r.boundary
        if up is not None:                       # the redraw counts 24 kHz samples; a replacement at 48 kHz starts up to DELAY earlier
            if boundary is not None:
                b48 = boundary
                boundary = lambda: b48() / 2 + up.lag() - 0.5              # noqa: E731
            else:
                p48 = played or (lambda: (time.perf_counter() - rd.t_first) * self.sample_rate if rd.t_first else 0.0)   # noqa: E731
                played = lambda: p48() / 2 + up.lag()                      # noqa: E731
        rd = v.rd = Redraw(v, sum(min(len(c) + self.off, self.MAXF - 1) for c in r.chunks), played, boundary) if use else None
        r.cutter = None
        if self.lead_cut if r.cut is None else r.cut:
            def on_cut(c):
                st["cut"] = c * self.sample_rate // SR
                if rd is not None:
                    rd.offset = c
            r.cutter = LeadCut(on_cut)
        r.co = v._generate(r.chunks, st, rd, r.t0)
        self._active.append(r)
        self._advance(r, None)

    def _advance(self, r, audio):
        """Runs r's coroutine on to its next frame (its pieces go out on the way) or to its end."""
        try:
            item = r.co.send(audio)
            while item is not r.view:
                self._out(r, item)
                item = next(r.co)
        except StopIteration:
            self._end(r)
        except Exception as exc:                 # the request fails, the others go on
            self._end(r, exc=exc)

    def _end(self, r, exc=None, close=False):
        if close:
            r.co.close()
        elif exc is None:
            for p in (r.cutter.finish() if r.cutter is not None else []):
                self._emit(r, p)
            if r.up is not None:
                a = r.up.ask_finish()
                if a is not None:
                    self._asks.append((r, a))
            if r.view.rd is not None:
                self.last_redraw = r.view.rd
        self._active.remove(r)
        self.free_slots.append(r.slot)
        self.free_pages.extend(r.page_ids)
        self.table[r.slot].fill_(self.n_pages)
        self._ending.append((r, exc))

    def _launch(self, rows):
        """One frame for every request in rows: the graphs and the copies to the host, not waited for."""
        B = len(rows)
        G = next(g for g in self.sizes if g >= B)
        key = tuple(r.slot for r in rows) + (G,)
        if key != self.batch_key:
            self.sel_host[:G] = self.SCRATCH
            self.sel_host[:B] = key[:B]
            self.adv_host[:G] = 0
            self.adv_host[:B] = 1
            self.sel[:G].copy_(torch.from_numpy(self.sel_host[:G]))
            self.adv[:G].copy_(torch.from_numpy(self.adv_host[:G]))
            self.batch_key = key
        self.g_dec[G].replay()
        self.audio_host[:G].copy_(self.baudio[:G], non_blocking=True)
        self.g_lm_head[G].replay()
        self.stop_host[:G].copy_(self.bstop[:G], non_blocking=True)
        self.step_ev.record()

    def _collect(self, rows):
        """The step's frames to the requests (each runs on to its next frame); the redraw's detector gets them first."""
        t = time.perf_counter()
        self.step_ev.synchronize()
        self.wait_s += time.perf_counter() - t
        self.steps += 1
        audio, stop = self.audio_host.numpy(), self.stop_host.numpy()
        for i, r in enumerate(rows):
            v = r.view
            v.row, v.stop_val = i, float(stop[i])
            if v.rd is not None and not r.cancelled:
                ch, t = v.frame_at
                g = ch["G"] + t - ch["k0"]
                v.pm_n = min(v.pm_n, max(0, (FRAME * g + self.pm_reach) // STEP + 1))   # map rows the frame changes
                v.pm_to = max(0, (FRAME * (g + 1) + self.pm_reach) // STEP + 1)         # rows whose audio is all there
                v.audio = self.baudio[i]
                v.rd.after_decode(g)             # the frame into the request's buffer; its windows to judge_later
        if self._judge_asks:
            self._judge_batch()
        for i, r in enumerate(rows):
            if not r.cancelled:
                self._advance(r, audio[i].copy())

    def judge_later(self, rd, lo, hi, take):
        self._judge_asks.append((rd, lo, hi, take))

    def _judge_batch(self):
        """The detector's windows of every request of the step in one pass (Redraw._judge, one by one, otherwise).
        Each request keeps the pitch map of its utterance: only the rows of new (or redrawn) audio are computed, all
        requests' in one batch; the windows are then scored from the kept maps (the same scores: a step's map depends
        on its own stretch of audio only, and the rows a window takes from before its start or after its last final
        step do not reach the steps it scores)."""
        asks, self._judge_asks = self._judge_asks, []
        J = self.judge
        span = J.W - CONTEXT - LAG
        wins = [(i, s, min(hi, s + span - 1)) for i, (_, lo, hi, _) in enumerate(asks) for s in range(lo, hi + 1, span)]
        w0s = [max(0, e + LAG - J.W) for _, _, e in wins]
        views = list({id(rd.eng): rd.eng for rd, _, _, _ in asks}.values())
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()                               # the audio is in the requests' buffers
        J.stream.wait_event(a)
        with torch.cuda.stream(J.stream):
            work = []
            for v in views:
                if v.pm is None or v.pm.shape[0] < v.rd.audio.numel() // STEP + 1:
                    v.pm, v.pm_n = torch.zeros((v.rd.audio.numel() // STEP + 1, J.pm.f.numel()), device=self.dev), 0
                if v.pm_n < v.pm_to:
                    work.append((v, v.pm_n, v.pm_to))
            if work:
                rows = J.map_rows(torch.cat([J.segments(v.rd.audio, t0, t1) for v, t0, t1 in work]))
                k = 0
                for v, t0, t1 in work:
                    v.pm[t0:t1].copy_(rows[k:k + t1 - t0])
                    k += t1 - t0
                    v.pm_n = t1
            m = torch.stack([asks[i][0].eng.pm[w0:w0 + J.W] for (i, _, _), w0 in zip(wins, w0s)])
            off = torch.tensor([float((-w0 * STEP) % FRAME) for w0 in w0s], dtype=torch.float64, device=self.dev)
            m = J.scores_maps(m, off)
            b.record()
            m = m.cpu().numpy()
        vals = [[] for _ in asks]
        for k, ((i, s, e), w0) in enumerate(zip(wins, w0s)):
            vals[i].append(m[k, s - w0:e - w0 + 1])
        for (rd, lo, hi, take), v in zip(asks, vals):
            rd.pending.append((lo, hi, v[0] if len(v) == 1 else np.concatenate(v), a, b, take))

    def _out(self, r, piece):
        for p in (r.cutter.push(*piece) if r.cutter is not None else [piece]):
            self._emit(r, p)

    def _emit(self, r, p):
        if r.up is not None:
            a = r.up.ask(*p)
            if a is not None:
                self._asks.append((r, a))
        else:
            self._deliver(r, p)

    def _deliver(self, r, p):
        st = r.stats
        if st["t_first"] is None:
            now = time.perf_counter()
            st["t_first"] = now - r.t0
            if r.view.rd is not None:
                r.view.rd.t_first = now          # the simulated player starts with the first piece
        r.q.put((p[0], p[1] * OUT_GAIN))         # the output gain on the pieces only (the redraw keeps its audio)

    def _flush(self):
        """The step's upsampler asks in one batch; then the ends of the requests that ended."""
        if self._asks:
            asks, self._asks = self._asks, []
            with torch.cuda.stream(self.side):   # beside the step on the main stream
                outs = self.bwe.segments([a for _, a in asks])
            for (r, a), out in zip(asks, outs):
                self._deliver(r, (a[1], out))
        ending, self._ending = self._ending, []
        for r, exc in ending:
            r.q.put(exc)                         # None: the end

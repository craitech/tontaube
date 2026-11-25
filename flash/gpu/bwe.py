"""AP-BWE 24 kHz -> 48 kHz on the GPU: the CPU engine's upsampler (flash/engine/bwe.h) with its weights (web/bwe.bin,
the int8 matrices decoded to fp32).

Our audio is upsampled 2x (torchaudio's sinc kernel) and kept as it is; the model's change of the spectrum is added only
from ~11.6 kHz up (g per bin). The network has convolutions only (9 layers of 7-frame filters), so an output sample
depends on a finite stretch of input (look-ahead DELAY): Stream computes each new stretch of output from the input kept
so far, and after a replacement in the input it computes the output again from the first sample the replacement reaches.
The common piece (one frame, 3,200 samples in, 6,400 out, inside the utterance) replays a captured CUDA graph, and
segments() computes the common pieces of several streams in one replay (a graph per batch size).
"""
import struct

import numpy as np
import torch
import torch.nn.functional as F

from . import kernels as K

DELAY = 2494        # output samples held back: output sample i needs input samples up to i / 2 + 1246
PIECE = 6400        # output samples of the piece the CUDA graph computes


def read_bin(path) -> dict:
    """The tensors of a TTLM0001 file (the CPU engine's format) as numpy arrays."""
    b = open(path, "rb").read()
    if b[:8] != b"TTLM0001":
        raise ValueError(f"{path}: not a TTLM0001 file")
    types = {0: np.uint8, 1: np.int8, 2: np.float16, 3: np.float32, 4: np.int32}
    out = {}
    for i in range(struct.unpack_from("<I", b, 8)[0]):
        e = 16 + 104 * i
        t, nd = struct.unpack_from("<II", b, e + 48)
        dims = struct.unpack_from("<4q", b, e + 56)[:nd]
        off, nb = struct.unpack_from("<QQ", b, e + 88)
        dt = np.dtype(types[t])
        out[b[e:e + 48].split(b"\0")[0].decode()] = np.frombuffer(b, dt, nb // dt.itemsize, off).reshape(dims)
    return out


class BWE:
    def __init__(self, path, device="cuda", graph=True, batch_sizes=(1,), half=False):
        """half: the network's matrix products in fp16 (fp32 sums; the residual stream and the output stay fp32)."""
        f = read_bin(path)
        self.dev = dev = torch.device(device)
        self.NF, self.HOP, self.WN, self.C, self.NL, self.KB0, _ = (int(v) for v in f["cfg"])
        self.W0, self.OFF = self.NF // 2 + 1, (self.NF - self.WN) // 2
        self.R = 3 * (self.NL + 1)                     # frames each side the network reads around a frame
        T = lambda a: torch.from_numpy(np.array(a, np.float32)).to(dev)   # noqa: E731

        def w(name):                                   # [N, K] fp32 (int8: code x scale + min per 128 inputs)
            if name + ".w" in f:
                return f[name + ".w"].astype(np.float32)
            c = f[name + ".codes"].astype(np.float32)
            s, m = (f[name + x].astype(np.float32)[..., None] for x in (".scales", ".mins"))
            return (c * s + m).reshape(len(c), -1)

        both = lambda fn: T(np.stack([fn("m."), fn("p.")]))                              # noqa: E731  [2, ...]: the streams
        self.upk, self.win, self.g = T(f["up_k"]).T.contiguous(), T(f["window"]), T(f["g"])
        self.pre_w = both(lambda p: w(p + "pre")[:, :7 * self.W0])                    # inputs: frame j - 3 (j < 7) x bin
        self.pre_b, self.pn_w, self.pn_b = (both(lambda p, x=x: f[p + x]) for x in ("pre.bias", "pre_norm.w", "pre_norm.b"))
        self.qn_w, self.qn_b = (both(lambda p, x=x: f[p + x]) for x in ("post_norm.w", "post_norm.b"))
        self.blocks = []
        for i in range(self.NL):
            q = lambda p, x: f"{p}b{i}.{x}"                                           # noqa: E731
            self.blocks.append(dict(dw=both(lambda p: f[q(p, "dw.w")]), dwb=both(lambda p: f[q(p, "dw.b")]),
                                    nw=both(lambda p: f[q(p, "norm.w")]), nb=both(lambda p: f[q(p, "norm.b")]),
                                    w1=both(lambda p: w(q(p, "pw1"))), b1=both(lambda p: f[q(p, "pw1.bias")]),
                                    w2=both(lambda p: w(q(p, "pw2"))), b2=both(lambda p: f[q(p, "pw2.bias")]),
                                    gam=both(lambda p: f[q(p, "gamma")])))
        self.post = [(T(w(n)), T(f[n + ".bias"])) for n in ("post_m", "post_r", "post_i")]
        self.dt = torch.half if half else torch.float32
        if half:                                       # the matrices of the network in fp16
            self.pre_w, self.pre_b = self.pre_w.half(), self.pre_b.half()
            for B in self.blocks:
                for k in ("w1", "b1", "w2", "b2"):
                    B[k] = B[k].half()
        self.graphs = {}                               # batch size -> (graph, input, output)
        if graph and self.dev.type == "cuda":
            self._capture(batch_sizes)

    def _geometry(self, n24, o0, o1, final):
        """The frames and input samples behind output samples [o0, o1): (P0, P1) reach them, (F0, F1) are read by the
        network, input samples [a, b) feed the upsampling (zeros outside the input)."""
        HOP, h = self.HOP, self.WN // 2
        T = 2 * n24 // HOP + 1 if final else 1 << 62   # STFT frames in all (torch.stft, center=True)
        P0, P1 = max(0, -((h - 1 - o0) // HOP)), min(T, (o1 - 1 + h) // HOP + 1)
        F0, F1 = max(0, P0 - self.R), min(T, P1 + self.R)
        s0, s1 = max(0, min(F0 * HOP - h, o0)), max((F1 - 1) * HOP + h, o1)
        if final:
            s1 = min(s1, 2 * n24)
        return P0, P1, F0, F1, s0 // 2 - 7, (s1 - 1) // 2 + 8

    def _capture(self, sizes):
        """CUDA graphs of the common piece (PIECE output samples inside the utterance, any phase to the STFT hop),
        one per batch size (one memory pool: the largest is captured first)."""
        P0, P1, F0, F1, a, b = self._geometry(1 << 40, 1 << 20, (1 << 20) + PIECE, False)
        self.g_geo = (P1 - P0, F1 - F0, P0 - F0, b - a)
        self.g_idx = torch.arange(F1 - F0, device=self.dev)[:, None] * self.HOP + torch.arange(self.WN, device=self.dev)
        pool = torch.cuda.graph_pool_handle()
        for B in sorted(set(sizes), reverse=True):
            g_in = torch.zeros(B, b - a, device=self.dev)
            side = torch.cuda.Stream(self.dev)
            side.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._graph_body(g_in)
            torch.cuda.current_stream(self.dev).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                g_out = self._graph_body(g_in)
            self.graphs[B] = (graph, g_in, g_out)

    def _graph_body(self, g_in):
        x48, hb = self._body(g_in, self.g_idx, self.R, self.R + self.g_geo[0])
        o = self.R * self.HOP                          # x48[:, o] is the first sample of hb
        return x48[:, o:o + hb.shape[1]] + hb

    @staticmethod
    def _lin(x, w, b):                                 # x [2, B, n, K] -> [2, B, n, N] (each stream its weights)
        return torch.baddbmm(b[:, None, :], x.reshape(2, -1, x.shape[-1]), w.transpose(1, 2)).reshape(*x.shape[:-1], -1)

    def _network(self, feat):
        """feat [2, B, n, W0] (log magnitude, phase) of n frames of B pieces -> [2, B, n, C]; zeros around the n
        frames. The depthwise convolutions and the norms run in one kernel each (kernels.conv_norm)."""
        n, dt = feat.shape[2], self.dt
        xp = F.pad(feat, (0, 0, 3, 3)).to(dt)
        x = self._lin(torch.cat([xp[:, :, j:j + n] for j in range(7)], 3), self.pre_w, self.pre_b).contiguous()
        x = K.conv_norm(x, None, None, self.pn_w, self.pn_b, torch.empty(x.shape, device=x.device))
        for B in self.blocks:
            xm = x[0] + x[1]                           # x_m = x_m + x_p; x_p = x_p + x_m (before every block)
            x = torch.stack([xm, x[1] + xm])
            d = K.conv_norm(x, B["dw"], B["dwb"], B["nw"], B["nb"], torch.empty(x.shape, device=x.device, dtype=dt))
            h = F.gelu(self._lin(d, B["w1"], B["b1"]))
            x = torch.addcmul(x, B["gam"][:, None, None, :], self._lin(h, B["w2"], B["b2"]))
        return K.conv_norm(x, None, None, self.qn_w, self.qn_b, torch.empty(x.shape, device=x.device))

    def _body(self, xin, idx, p0, p1):
        """xin [B, L]: 24 kHz input of B pieces on the GPU (7 samples before the first pair each upsamples); idx [n, WN]:
        the samples of n STFT frames in the upsampled audio; frames p0 .. p1 - 1 of them reach the output -> (x48, hb)
        [B, ...]: the upsampled audio and the model's change overlap-added (hb[:, 0] at frame p0's first sample)."""
        HOP, WN, NF, KB0 = self.HOP, self.WN, self.NF, self.KB0
        Bn = xin.shape[0]
        x48 = (xin.unfold(1, 15, 1) @ self.upk).reshape(Bn, -1)   # x48[2n + i] = sum_k x[n + k - 7] up_k[i, k]
        X = torch.fft.rfft(F.pad(x48[:, idx] * self.win, (self.OFF, NF - WN - self.OFF)))
        la = torch.log(X.abs() + 1e-4)
        y = self._network(torch.stack([la, torch.angle(X)]))[:, :, p0:p1]
        Xs, las = X[:, p0:p1, KB0:], la[:, p0:p1, KB0:]
        n = p1 - p0
        ml, pr, pi = (torch.addmm(b_, y[k].reshape(Bn * n, -1), w_.T).reshape(Bn, n, -1) for k, (w_, b_) in zip((0, 1, 1), self.post))
        nn_ = torch.sqrt(pr * pr + pi * pi)
        ok = nn_ > 0
        c = torch.complex(torch.where(ok, pr / nn_, 1.0), torch.where(ok, pi / nn_, 0.0))
        D = (torch.exp(las + ml) * c - Xs) * self.g
        D = F.pad(D, (KB0, 0))
        D[..., -1] = D[..., -1].real                   # no imaginary part at Nyquist
        fr = torch.fft.irfft(D, n=NF)[..., self.OFF:self.OFF + WN] * self.win
        fold = lambda v: F.fold(v.transpose(1, 2), (1, (n - 1) * HOP + WN), (1, WN), stride=(1, HOP)).reshape(v.shape[0], -1)   # noqa: E731
        ola, env = fold(fr), fold((self.win * self.win)[None, None].expand(1, n, WN))
        return x48, torch.where(env > 1e-11, ola / env, 0.0)

    @torch.inference_mode()
    def segment(self, x, o0, o1, final) -> np.ndarray:
        """48 kHz output samples [o0, o1) of the 24 kHz input x (numpy, all of it so far; final: x ends there)."""
        HOP, WN, h, n24 = self.HOP, self.WN, self.WN // 2, len(x)
        L = 2 * n24
        P0, P1, F0, F1, a, b = self._geometry(n24, o0, o1, final)
        if not final and b > n24:
            raise ValueError("output asked for beyond the look-ahead")
        xin = np.zeros(b - a, np.float32)
        xin[max(0, -a):min(n24, b) - a] = x[max(0, a):min(n24, b)]
        ob = P0 * HOP - h                              # the first sample of hb
        if 1 in self.graphs and not final and a >= 0 and (P1 - P0, F1 - F0, P0 - F0, b - a) == self.g_geo:
            graph, g_in, g_out = self.graphs[1]
            g_in[0].copy_(torch.from_numpy(xin))
            graph.replay()
            return g_out[0, o0 - ob:o1 - ob].cpu().numpy()
        base = 2 * (a + 7)                             # x48[0] is output sample base
        if final and L <= WN:                          # too short for the STFT: the plain upsampled audio
            x48 = (torch.from_numpy(xin).to(self.dev).unfold(0, 15, 1) @ self.upk).reshape(-1)
            return x48[o0 - base:o1 - base].cpu().numpy()
        # STFT frames [F0, F1) (reflected at both ends of the utterance)
        idx = (torch.arange(F0, F1, device=self.dev)[:, None] * HOP - h + torch.arange(WN, device=self.dev)).abs()
        if final:
            idx = torch.where(idx >= L, 2 * (L - 1) - idx, idx)
        x48, hb = self._body(torch.from_numpy(xin).to(self.dev)[None], idx - base, P0 - F0, P1 - F0)
        return (x48[0, o0 - base:o1 - base] + hb[0, o0 - ob:o1 - ob]).cpu().numpy()

    @torch.inference_mode()
    def segments(self, asks) -> list:
        """segment() for several streams at once: asks [(x, o0, o1, final), ...] -> their outputs. The common pieces
        go through one replay of the batch graph (in groups of the largest batch size), the others one by one."""
        out, common = [None] * len(asks), []
        for i, (x, o0, o1, final) in enumerate(asks):
            P0, P1, F0, F1, a, b = self._geometry(len(x), o0, o1, final)
            if self.graphs and not final and a >= 0 and b <= len(x) and (P1 - P0, F1 - F0, P0 - F0, b - a) == self.g_geo:
                common.append((i, a, b, o0 - (P0 * self.HOP - self.WN // 2), o1 - o0))
            else:
                out[i] = self.segment(x, o0, o1, final)
        most = max(self.graphs) if self.graphs else 1
        for k in range(0, len(common), most):
            group = common[k:k + most]
            graph, g_in, g_out = self.graphs[min(s for s in self.graphs if s >= len(group))]
            xin = np.zeros(tuple(g_in.shape), np.float32)
            for j, (i, a, b, _, _) in enumerate(group):
                xin[j] = asks[i][0][a:b]
            g_in.copy_(torch.from_numpy(xin))
            graph.replay()
            res = g_out[:len(group)].cpu().numpy()
            for j, (i, _, _, s0, n) in enumerate(group):
                out[i] = res[j, s0:s0 + n]
        return out

    def upsample(self, x) -> np.ndarray:
        """The whole utterance x (24 kHz) at 48 kHz."""
        return self.segment(np.asarray(x, np.float32), 0, 2 * len(x), True)


class Stream:
    """(start, audio) pieces at 24 kHz, a start before the end replacing what came after it -> the same at 48 kHz,
    assembled equal to the whole utterance's result."""

    def __init__(self, bwe):
        self.bwe, self.x, self.out = bwe, np.zeros(0, np.float32), 0   # input so far; output samples given

    def lag(self) -> float:
        """Input samples (24 kHz) not yet answered by output."""
        return len(self.x) - self.out / 2

    def _ask(self, end, final):
        if end <= self.out:
            return None
        o0, self.out = self.out, end
        return self.x, o0, end, final

    def ask(self, start, audio):
        """A piece in: what the output asks of the upsampler, (x, o0, o1, final) for BWE.segment(s) (the output
        piece starts at o0), or None (all of it still in the look-ahead)."""
        self.x = np.concatenate([self.x[:start], np.asarray(audio, np.float32)])
        self.out = min(self.out, max(0, 2 * start - DELAY))
        return self._ask(2 * len(self.x) - DELAY, False)

    def ask_finish(self):
        return self._ask(2 * len(self.x), True)

    def push(self, start, audio):
        """A piece in: (start, audio) at 48 kHz out, or None (all of it still in the look-ahead)."""
        a = self.ask(start, audio)
        return None if a is None else (a[1], self.bwe.segment(*a))

    def finish(self):
        a = self.ask_finish()
        return None if a is None else (a[1], self.bwe.segment(*a))

    def run(self, pieces):
        for start, audio in pieces:
            if (p := self.push(start, audio)) is not None:
                yield p
        if (p := self.finish()) is not None:
            yield p

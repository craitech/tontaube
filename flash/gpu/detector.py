"""The redraw's detector, streaming on the GPU: it hears where the pitch jumps to another note.

PitchMap: every 5 ms the normalised cross-correlation of a 25 ms window with the same window one candidate period
later, on a log-pitch axis from 70 to 300 Hz in thirds of a semitone (no learned parts). PitchDisc: a small network
(2D convolutions over time x pitch, then 1D over time with the position inside the 133 ms frame) gives one score per
5 ms step. Weights: gpu/detector.safetensors of the release.

Score of a 5 ms step t (centred at sample 120 t of the utterance): the pitch map quantised to 8 bits, clarity = its
maximum over the pitch bins, p = sigmoid(PitchDisc), voiced = clarity > 0.5, m_t = mean of p * voiced over steps
t-10 .. t+9 (zero padding at the utterance's edges). A step is final once its pitch-map window, the detector's view and
the pooling reach only decoded audio: t <= decoded samples / 120 - LAG. `Judge.scores` computes final steps on a window
of the recent audio (at least CONTEXT steps before the first one) and gives exactly the values the whole utterance
gives: the pitch map is computed in float64 (in float32 its rounding would differ between a window and the whole
utterance and now and then move an 8-bit map value) and the convolutions in fp32 without TF32.
"""
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

SR, HOP, FRAME = 24000, 120, 3200
DT = HOP / SR
LAG = 48            # steps a score needs past it (pitch-map window 4.7 + detector view 30 + pooling 9, rounded up)
CONTEXT = 100       # steps of audio before the first step used (0.5 s)


class PitchMap(nn.Module):
    def __init__(self, fmin: float = 70.0, fmax: float = 300.0, per_semitone: int = 3, win: int = 600):
        super().__init__()
        n = int(round(12 * per_semitone * math.log2(fmax / fmin))) + 1
        f = fmin * 2 ** (torch.arange(n, dtype=torch.float64) / (12 * per_semitone))
        lag = SR / f                                            # fractional lags, falling with pitch
        lo = lag.floor().long()
        self.win, self.tmax = win, int(lo.max()) + 1
        self.register_buffer("f", f.float(), persistent=False)
        self.register_buffer("lo", lo, persistent=False)
        self.register_buffer("w", (lag - lo).float(), persistent=False)
        self.nfft = 1 << int(math.ceil(math.log2(2 * win + self.tmax)))
        self.centre = int(round((win + SR / math.sqrt(fmin * fmax)) / 2))    # analysis centre of a step at the middle lag

    def steps(self, n_samples: int) -> int:
        return -(-n_samples // HOP)

    def forward(self, x):
        """x [B, N] -> map [B, T, P] (correlation, clamped at 0), clarity [B, T]; T = ceil(N / HOP)."""
        B, N = x.shape
        T = self.steps(N)
        L = self.win + self.tmax
        x = F.pad(x, (self.centre, (T - 1) * HOP + L - self.centre - N))
        return self.maps(x.unfold(1, L, HOP)[:, :T])           # [B, T, L]

    def maps(self, seg):
        """Analysis segments [..., L] (one per step: samples from 120 t - centre on) -> map [..., P], clarity [...]."""
        a = seg[..., :self.win]
        corr = torch.fft.irfft(torch.fft.rfft(seg, self.nfft) * torch.fft.rfft(a, self.nfft).conj(), self.nfft)[..., :self.tmax + 1]
        c2 = F.pad(seg.pow(2).cumsum(-1), (1, 0))
        eb = c2[..., self.win:self.win + self.tmax + 1] - c2[..., :self.tmax + 1]      # energy of seg[tau : tau + win]
        ea = eb[..., :1]
        nccf = corr / torch.sqrt(ea * eb + (self.win * 1e-7) ** 2)
        m = torch.lerp(nccf[..., self.lo], nccf[..., self.lo + 1], self.w).clamp_min(0)  # [B, T, P]
        return m, m.amax(-1)


def phase_features(T: int, offset, device=None):
    """[B, 4, T]: sin and cos of the position inside the 133 ms frame (and of twice it) at each step centre; offset
    (a number or a tensor [B]) = the sample position of a frame boundary."""
    off = torch.as_tensor(offset, device=device, dtype=torch.float64).reshape(-1, 1)
    ph = ((torch.arange(T, device=device, dtype=torch.float64)[None] * HOP - off) % FRAME / FRAME * 2 * math.pi).float()
    return torch.stack([ph.sin(), ph.cos(), (2 * ph).sin(), (2 * ph).cos()], 1)


class PitchDisc(nn.Module):
    def __init__(self, c1: int = 16, c2: int = 24, c3: int = 32, h: int = 32):
        super().__init__()
        self.map = PitchMap()
        self.conv1 = nn.Conv2d(2, c1, (5, 5), padding=(2, 2))
        self.conv2 = nn.Conv2d(c1, c2, (5, 5), padding=(4, 2), dilation=(2, 1))
        self.conv3 = nn.Conv2d(c2, c3, (5, 5), padding=(8, 2), dilation=(4, 1))
        self.head1 = nn.Conv1d(2 * c3 + 4, h, 5, padding=16, dilation=8)
        self.head2 = nn.Conv1d(h, 1, 1)

    def from_map(self, m, clarity, offset: int = 0):
        """m [B, T, P], clarity [B, T] -> logits [B, T]."""
        B, T, P = m.shape
        x = torch.stack([m, clarity[..., None].expand(B, T, P)], 1)          # [B, 2, T, P]
        x = F.mish(self.conv1(x))
        x = F.max_pool2d(x, (1, 2), ceil_mode=True)                         # pitch axis in steps of 2/3 semitone
        x = F.mish(self.conv2(x))
        x = F.mish(self.conv3(x))
        x = torch.cat([x.amax(-1), x.mean(-1), phase_features(T, offset, x.device).expand(B, 4, T)], 1)   # offset: number or [B]
        return self.head2(F.mish(self.head1(x)))[:, 0]

    def forward(self, x, offset: int = 0):
        """x [B, N] audio at 24 kHz, frame boundaries at offset + k * 3200 -> logits [B, ceil(N / 120)]."""
        m, c = self.map(x)
        return self.from_map(m, c, offset)


class Judge:
    """Final scores m_t of an utterance's audio, window by window, on a stream of its own. One request at a time
    (static buffers). Pitch map in float64, convolutions in fp32 without cuDNN's TF32 mode."""

    def __init__(self, path, device, window=192, graphs=True):
        f = safe_open(str(path), "pt", device="cpu")
        self.net = PitchDisc(**json.loads(f.metadata()["arch"])).to(device).eval()
        self.net.load_state_dict({k: f.get_tensor(k) for k in f.keys()})
        self.pm = PitchMap().to(device).double()
        self.dev, self.W = device, window
        self.x = torch.zeros(1, window * HOP, device=device)
        self.off = torch.zeros(1, dtype=torch.float64, device=device)
        self.m = torch.zeros(window, device=device)
        self.stream = torch.cuda.Stream(device) if torch.device(device).type == "cuda" else None
        self.graph = None
        if graphs:
            with torch.inference_mode(), torch.cuda.stream(self.stream):
                self._run()
                torch.cuda.synchronize()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=self.stream):
                    self._run()

    def _scores(self, x, off):
        with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
            return self._scores_fp32(x, off)

    def _scores_fp32(self, x, off):
        m, _ = self.pm(x.double())
        m = ((m.clamp(0, 1) * 255).round() / 255).float()                   # 8 bits
        c = m.amax(-1)
        p = torch.sigmoid(self.net.from_map(m, c, off))
        return F.avg_pool1d((p * (c > 0.5))[:, None], 20, 1, padding=10)[:, 0, :m.shape[1]]

    def _run(self):
        self.m.copy_(self._scores(self.x, self.off)[0])

    def scores(self, audio, lo, hi, total=None):
        """GPU tensor of m for steps lo..hi of the utterance whose audio is in `audio` (a 1-D GPU buffer, at least
        `window` steps of samples longer than the audio). total: the utterance has ended with this many samples (the
        last steps then see its end as the whole utterance does); otherwise steps up to hi must be final. Runs on the
        detector's stream: the caller orders it after the audio is written and waits for its result."""
        if total is None:
            assert hi - lo + 1 + CONTEXT + LAG <= self.W, "range too long for one window"
            w0 = max(0, hi + LAG - self.W)
            self.x[0].copy_(audio[w0 * HOP:(w0 + self.W) * HOP])
            self.off.fill_(float((-w0 * HOP) % FRAME))
            self.graph.replay() if self.graph is not None else self._run()
            return self.m[lo - w0:hi - w0 + 1]
        w0 = max(0, lo - CONTEXT)                                           # exact length: ends where the utterance ends
        assert hi - w0 < 4 * self.W
        return self._scores(audio[w0 * HOP:total][None], float((-w0 * HOP) % FRAME))[0, lo - w0:hi - w0 + 1]

    def scores_many(self, windows):
        """m [K, window] of K windows at once: (audio buffer, first step w0) each, as scores() reads them (the
        utterance goes on past the window). Runs on the current stream."""
        x = torch.empty((len(windows), self.W * HOP), device=self.dev)
        for k, (audio, w0) in enumerate(windows):
            x[k].copy_(audio[w0 * HOP:(w0 + self.W) * HOP])
        off = torch.tensor([float((-w0 * HOP) % FRAME) for _, w0 in windows], dtype=torch.float64, device=self.dev)
        return self._scores(x, off)

    # a pitch map kept per utterance (the batch engine): each step's map is computed once, when its audio is final
    def segments(self, audio, t0, t1):
        """The analysis segments [t1 - t0, L] of steps t0 .. t1 - 1 of the utterance in `audio` (zeros before it)."""
        L, c = self.pm.win + self.pm.tmax, self.pm.centre
        s0 = t0 * HOP - c
        x = audio[max(0, s0):(t1 - 1) * HOP - c + L]
        return (F.pad(x, (-s0, 0)) if s0 < 0 else x).unfold(0, L, HOP)

    def map_rows(self, seg):
        """8-bit pitch map rows [K, P] from analysis segments [K, L] (float64 inside, as _scores)."""
        m, _ = self.pm.maps(seg.double())
        return ((m.clamp(0, 1) * 255).round() / 255).float()

    def scores_maps(self, m, off):
        """m [K, W, P]: the 8-bit maps of K windows, off [K]: their frame phase -> scores [K, W] (as _scores)."""
        with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
            c = m.amax(-1)
            p = torch.sigmoid(self.net.from_map(m, c, off))
            return F.avg_pool1d((p * (c > 0.5))[:, None], 20, 1, padding=10)[:, 0, :m.shape[1]]

    @torch.inference_mode()
    def whole(self, x):
        """m of a whole utterance at once (the reference for the streaming scores)."""
        return self._scores(torch.as_tensor(x, device=self.dev).float()[None], 0.0)[0]

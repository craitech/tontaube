"""Residual VibeVoice-latent adapter with a silence gate, on the /predict decode (decode_chunked_with_context), and on
/stream with STREAM_VV_ADAPTER (correct_stream_frames).

VV_ADAPTER_CKPT=<checkpoint .pt> turns it on (ServerConfig.vv_adapter_ckpt). Unset or empty (the default) = off:
the decode is byte-identical to the decode without the adapter, on both routes. STREAM_VV_ADAPTER=0 keeps /stream on
the production frames while /predict uses the adapter.

Checkpoint identity: VV_ADAPTER_SHA256 is required with VV_ADAPTER_CKPT; the file must hash to it, and the hash must
be in APPROVED_SHA256 unless VV_ADAPTER_ALLOW_UNLISTED=1. TTSEngine.load calls load_vv_adapter before any other model
loads, so a missing file, a wrong hash or a bad checkpoint stops the server within seconds.

What it does, per request (all 4-level DualCodec codes of the request, 12.5 Hz):
    z = z_prod + gate * dz(DC latent, z_prod)
  * z_prod: the production VibeVoice latent, i.e. the frames the windowed VV encode collects anyway.
  * DC latent: the DualCodec pre-decoder latent of the codes (CB0 semantic branch + CB1-3 acoustic residuals),
    [1, 1024, K]; computed per decode window and kept for the window's content codes only.
  * gate: per VibeVoice frame (3200 samples) from the level of the DualCodec-decoded audio: 0 below -65 dBFS (the
    frame stays production), 1 above -55 dBFS, linear in between.
  * Tail fix: the adapter emits floor(3K/5) frames, so the DC latent is edge-padded to Kp = 5 * ceil(K/5) codes and
    the last window's audio is zero-padded so the VV encode yields 3 * Kp / 5 frames. The caller cuts the decoded
    audio back to the length the path has with the adapter off (K x 1920 samples with DECODE_END_FIX=1).
  * Precision: the adapter weights stay float32; on CUDA it runs under bf16 autocast, on CPU in float32.
  * Memory: the adapter runs in chunks of CHUNK_CODES codes with CONTEXT_CODES codes of left context (more than its
    receptive field, ~75 codes), so its GPU memory does not grow with the request length. A request of up to
    CHUNK_CODES codes (60 s) is one pass.

Model: Adapter is a causal convolutional backbone (CausalBlock stacks) from the DualCodec latent (12.5 Hz) to the
VibeVoice latent (7.5 Hz); ResAdapter is its residual form, which also reads the production latent and outputs
z_prod + dz; silence_gate weights dz per frame. Only the residual path is implemented.
"""
import contextlib
import hashlib
import io
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.logging_utils import get_logger

logger = get_logger("vv_adapter")

DC_HOP = 1920   # samples per DualCodec code (12.5 Hz at 24 kHz)
VV_HOP = 3200   # samples per VibeVoice frame (7.5 Hz)
DC_HZ = 12.5
N_ACOUSTIC = 3  # the adapter reads 4-level codes: CB0 + 3 acoustic codebooks
CHUNK_CODES = 750   # 60 s per adapter pass (a multiple of 5)
CONTEXT_CODES = 100  # 8 s of left context per pass (a multiple of 5; receptive field ~75 codes)
GRID_CODES = 5      # 5 DualCodec codes = 3 VibeVoice frames = 9600 samples = 0.4 s

# Adapter checkpoints approved for serving: sha256 -> what it is. Other adapter checkpoints have the same format and
# load without error, so the hash is what tells them apart.
# Found automatically in the model root when VV_ADAPTER_CKPT is unset (the approved list still applies).
VV_ADAPTER_FILE = "vibevoice_adapter.pt"

APPROVED_SHA256 = {
    "80543a36669725947891dfe1002e24531f3816edd05c946b5e76861582d98181": "TontaubeV1.1 VibeVoice adapter",
}


# ---------------------------------------------------------------- model
def silence_gate(a, T, lo=-65.0, hi=-55.0, hop=VV_HOP):
    """Per VibeVoice frame, the weight of the correction from the level of the DualCodec-decoded audio `a` [B, N]:
    0 below `lo` dBFS (output = production), 1 above `hi`, linear in between. Frame j only uses samples of frame j.
    Returns [B, T, 1]."""
    n = min(a.shape[1] // hop, T)
    fr = a[:, : n * hop].reshape(a.shape[0], n, hop).float().pow(2).mean(-1)
    db = 10 * torch.log10(fr + 1e-12)
    g = ((db - lo) / (hi - lo)).clamp(0, 1)
    if n < T:
        g = F.pad(g, (0, T - n), value=1.0)
    return g[..., None]


class CausalBlock(nn.Module):
    def __init__(self, d, k=7, dil=1, mult=4):
        super().__init__()
        self.pad = (k - 1) * dil
        self.dw = nn.Conv1d(d, d, k, dilation=dil, groups=d)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Linear(mult * d, d))
        self.gamma = nn.Parameter(torch.full((d,), 0.1))

    def forward(self, x):  # [B, d, T]
        y = self.dw(F.pad(x, (self.pad, 0)))
        y = self.ff(self.norm(y.transpose(1, 2))) * self.gamma
        return x + y.transpose(1, 2)


class Adapter(nn.Module):
    """Causal DualCodec latent (1024-d, 12.5 Hz) -> VibeVoice latent (64-d, 7.5 Hz) backbone. Both rates share a
    37.5 Hz grid: DualCodec frame k = sub-frames 3k..3k+2, VibeVoice frame j = sub-frames 5j..5j+4. `lookahead`
    (DualCodec frames, default 1 = 80 ms) shifts the input left."""

    def __init__(self, d_in=1024, d=384, hi_dils=(1, 2, 4, 8, 1, 2, 4, 8), n_lo=4, lookahead=1, z_dim=64):
        super().__init__()
        self.lookahead = lookahead
        self.inp = nn.Conv1d(d_in, d, 1)
        self.phase = nn.Parameter(torch.zeros(d, 3))
        self.hi = nn.Sequential(*[CausalBlock(d, 7, dil) for dil in hi_dils])
        self.fold = nn.Linear(5 * d, d)
        self.lo = nn.Sequential(*[CausalBlock(d, 3, 1) for _ in range(n_lo)])
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, z_dim)
        self.register_buffer("z_mean", torch.zeros(z_dim))
        self.register_buffer("z_std", torch.ones(z_dim))
        self.register_buffer("x_scale", torch.ones(1))


class ResAdapter(Adapter):
    """Residual adapter: z = z_prod + f(DC latent, z_prod); z_prod enters at the 7.5 Hz stage (frame j only)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        d = self.inp.out_channels
        self.zin = nn.Linear(self.out.out_features, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, zp, gate=None):
        """x: [B, 1024, K] DualCodec latent, zp: [B, >= floor(3K/5), 64] production latent -> [B, floor(3K/5), 64]."""
        B, _, K = x.shape
        x = x / self.x_scale
        if self.lookahead:
            x = F.pad(x[:, :, self.lookahead:], (0, self.lookahead))
        h = self.inp(x)
        h = h.repeat_interleave(3, dim=2) + self.phase.repeat(1, K)[None]
        h = self.hi(h)
        T = (3 * K) // 5
        d = h.shape[1]
        h = h[:, :, : 5 * T].reshape(B, d, T, 5).permute(0, 2, 3, 1).reshape(B, T, 5 * d)
        zp = zp[:, :T].float()
        h = self.fold(h) + self.zin((zp - self.z_mean) / self.z_std)
        h = self.lo(h.transpose(1, 2))
        dz = self.out(self.out_norm(h.transpose(1, 2)))
        if gate is not None:
            dz = dz * gate[:, :T].to(dz.dtype)
        return zp + dz * self.z_std


# ---------------------------------------------------------------- serving glue
class VVAdapterCheckpointError(ValueError):
    """VV_ADAPTER_CKPT / VV_ADAPTER_SHA256 do not name an approved adapter checkpoint. Raised by TTSEngine.load before
    any model loads: the server does not start (no fallback to the decoder without the adapter)."""


def _approved() -> str:
    return "; ".join(f"{h} = {what}" for h, what in APPROVED_SHA256.items())


def read_vv_adapter_ckpt(path: str, sha256: str, allow_unlisted: bool = False) -> tuple[bytes, str]:
    """Read the checkpoint file once and check its identity. Returns (its bytes, its sha256): the caller loads these
    bytes, so the bytes that were hashed are the bytes that are loaded.

    Raises VVAdapterCheckpointError if `sha256` (VV_ADAPTER_SHA256) is empty or not a sha256 hex digest, if `path` is
    not a file, if the file's sha256 is not `sha256`, or if it is not in APPROVED_SHA256 and allow_unlisted
    (VV_ADAPTER_ALLOW_UNLISTED=1) is off."""
    expected = (sha256 or "").strip().lower()
    if not expected:
        raise VVAdapterCheckpointError(
            f"VV_ADAPTER_CKPT={path} is set but VV_ADAPTER_SHA256 is not. Set VV_ADAPTER_SHA256 to the checkpoint's "
            f"sha256 (approved: {_approved()}). The server does not start.")
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise VVAdapterCheckpointError(
            f"VV_ADAPTER_SHA256={sha256!r} is not a sha256 hex digest (64 hex characters). The server does not start.")
    if not os.path.isfile(path):
        raise VVAdapterCheckpointError(f"VV_ADAPTER_CKPT={path}: no such file. The server does not start.")
    with open(path, "rb") as f:
        data = f.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise VVAdapterCheckpointError(
            f"VV_ADAPTER_CKPT={path} has sha256 {actual}, but VV_ADAPTER_SHA256 is {expected}: a different checkpoint "
            f"or a bad copy. The server does not start.")
    if actual not in APPROVED_SHA256:
        if not allow_unlisted:
            raise VVAdapterCheckpointError(
                f"VV_ADAPTER_CKPT={path} (sha256 {actual}) is not an approved adapter checkpoint (approved: "
                f"{_approved()}). Set VV_ADAPTER_ALLOW_UNLISTED=1 only for a deliberate experiment. "
                f"The server does not start.")
        logger.warning(
            "VibeVoice adapter checkpoint %s (sha256 %s) is NOT an approved checkpoint; loading it only because "
            "VV_ADAPTER_ALLOW_UNLISTED=1", path, actual)
    return data, actual


def load_vv_adapter(path: str, sha256: str, allow_unlisted: bool = False) -> ResAdapter:
    """Check the checkpoint's identity (read_vv_adapter_ckpt), then build the residual adapter from the checked bytes
    ({"model", "args"}) on the CPU, in eval mode, without gradients.
    activate_vv_adapter moves it to the VibeVoice device. Raises VVAdapterCheckpointError on any problem."""
    data, sha = read_vv_adapter_ckpt(path, sha256, allow_unlisted)
    ck = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    del data
    args = ck["args"]
    if not args.get("residual"):
        raise VVAdapterCheckpointError(f"VV_ADAPTER_CKPT={path} is not a residual adapter checkpoint (args {args})")
    model = ResAdapter(d=args.get("d", 384), lookahead=args.get("lookahead", 1))
    model.load_state_dict(ck["model"])
    model = model.eval().requires_grad_(False)
    model.sha256, model.path = sha, path
    logger.info("VibeVoice adapter checkpoint checked: %s (sha256 %s)", path, sha)
    return model


def activate_vv_adapter(model: ResAdapter, device, stream: bool = False) -> ResAdapter:
    """Move a checked adapter (load_vv_adapter) to the VibeVoice device and log the startup line. stream: whether /stream
    uses it too (STREAM_VV_ADAPTER)."""
    model = model.to(device)
    logger.info(
        "VibeVoice adapter on: %s (sha256 %s, %.1f M parameters, %s; %s)",
        model.path, model.sha256, sum(p.numel() for p in model.parameters()) / 1e6, device,
        "/predict and /stream" if stream else "/predict only, /stream unchanged (STREAM_VV_ADAPTER=0)",
    )
    return model


def check_grid(what: str, n_codes: int) -> None:
    """Raise RuntimeError unless n_codes is a multiple of GRID_CODES (5 codes = 3 VibeVoice frames = 9600 samples).

    What this protects: the adapter adds gate * dz, computed from the DualCodec latent and the gate on the request's
    global grid (code k = 37.5 Hz sub-frames 3k..3k+2, VibeVoice frame j = request samples 3200j..3200j+3199), to
    z_prod, the VibeVoice frames decode_chunked_with_context collects per 30 s window. A window's z_prod sits on that
    global grid only if its left context and the content of every window before it are whole multiples of 5 codes
    (today: 6 s = 75 codes and 30 s = 375 codes). Otherwise left_pad_vv = int(left_pad_tokens / hz * 7.5) and the
    per-window frame count int(n_content / hz * 7.5) are floored, z_prod slides against the latent and the gate in every
    window after the first (5.0 s of context = 62 codes: by 640 samples), and nothing else raises: the correction just
    lands on the wrong frames. Called per decode window (add_window) and once at startup for the context."""
    if n_codes % GRID_CODES:
        raise RuntimeError(
            f"VibeVoice adapter: {what} is {n_codes} DualCodec codes, not a multiple of {GRID_CODES} codes "
            f"(3 VibeVoice frames); the adapter's correction would land on the wrong frames")


@torch.no_grad()
def dc_latent(dc_inference, semantic_codes: list[int], acoustic_codes: list[list[int]]) -> torch.Tensor:
    """The quantized sum the DualCodec acoustic decoder consumes, [1, 1024, T] at 12.5 Hz, in the DualCodec model's
    dtype (float16 on CUDA in serving). The same tensors as app.utils.codes_to_audio_tensor builds, and the same
    latent as DualCodec's decode_from_codes computes before its decoder."""
    m = dc_inference.model
    device = next(m.parameters()).device
    sem = torch.tensor(semantic_codes, dtype=torch.long, device=device)[None, None]
    ac = torch.tensor(acoustic_codes, dtype=torch.long, device=device)[None]
    s = m.semantic_vq.from_codes(sem)[0]
    if m.decode_semantic_for_codec:
        s = m.convnext_decoder(s)
    return m.dac.quantizer.from_codes(ac)[0] + s


def _fit(x: torch.Tensor, n: int) -> torch.Tensor:
    """x [..., N] trimmed or zero-padded at the end to n samples."""
    return x[..., :n] if x.shape[-1] >= n else F.pad(x, (0, n - x.shape[-1]))


@torch.no_grad()
def run_adapter(model, latent, z_prod, gate, chunk_codes: int = CHUNK_CODES, context_codes: int = CONTEXT_CODES):
    """latent [1, 1024, Kp] (Kp a multiple of 5), z_prod [1, 3Kp/5, 64], gate [1, 3Kp/5, 1] -> z [1, 3Kp/5, 64]
    (float32, cpu). Runs in chunks of chunk_codes with context_codes of left context and 5 codes of right context
    (the 1-code lookahead); chunk edges sit on the 5-code / 3-frame grid, so the result equals one pass."""
    assert chunk_codes % 5 == 0 and context_codes % 5 == 0
    device = next(model.parameters()).device
    K = latent.shape[-1]
    assert K % 5 == 0 and z_prod.shape[1] == gate.shape[1] == 3 * K // 5, (latent.shape, z_prod.shape, gate.shape)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext())
    out = []
    for a in range(0, K, chunk_codes):
        b = min(a + chunk_codes, K)
        s, e = max(0, a - context_codes), min(K, b + 5)
        with amp:
            z = model(latent[..., s:e].to(device).float(), z_prod[:, 3 * s // 5:3 * e // 5].to(device),
                      gate=gate[:, 3 * s // 5:3 * e // 5].to(device))
        out.append(z[:, 3 * (a - s) // 5:3 * (b - s) // 5].float().cpu())
    return torch.cat(out, dim=1)


class VVAdapterRequest:
    """One request's adapter inputs, collected over decode_chunked_with_context's windows (content part only, cpu),
    then applied to the collected VibeVoice frames before the streaming decode."""

    def __init__(self, model, n_codes: int, hz: float):
        self.model = model
        self.n_codes = n_codes
        self.hz = hz
        self.n_codes_padded = -(-n_codes // 5) * 5   # tail fix: a whole number of 5-code / 3-frame groups
        self.n_frames = 3 * self.n_codes_padded // 5  # VibeVoice frames the adapter emits and the encode must cover
        self.n_frames_seen = 0
        self.n_frames_off = 0  # frames the decode keeps with the adapter off: the output length without DECODE_END_FIX
        self.latent: list[torch.Tensor] = []
        self.gate: list[torch.Tensor] = []

    @classmethod
    def create(cls, model, n_codes: int, hz: float, n_acoustic: int) -> "VVAdapterRequest | None":
        if n_codes <= 0:
            return None  # no codes, nothing to correct: the decode then fails exactly as with the adapter off
        if hz != DC_HZ or n_acoustic != N_ACOUSTIC:
            logger.warning("VibeVoice adapter skipped for this request: needs %d-level codes at %.1f Hz "
                           "(got %d levels at %.1f Hz, %d codes)", N_ACOUSTIC + 1, DC_HZ, n_acoustic + 1, hz, n_codes)
            return None
        return cls(model, n_codes, hz)

    def add_window(self, dc_inference, win_sem, win_ac, window_audio, left_pad_tokens: int, left_pad_samples: int,
                   n_content_tokens: int, last: bool) -> tuple[torch.Tensor, int]:
        """Collect one decode window's DC latent and gate (content codes only). Returns the audio to VV-encode (the
        window audio itself; for the last window, zero-padded so the encode covers the adapter's frame grid) and the
        number of content VibeVoice frames to keep from that encode."""
        # the window must sit on the 5-code / 3-frame grid, or z_prod would slide against the latent and the gate
        check_grid("this decode window's left context", left_pad_tokens)
        if not last:
            check_grid("a decode window's content (every window but the last)", n_content_tokens)
        lat = dc_latent(dc_inference, win_sem, win_ac)[..., left_pad_tokens:left_pad_tokens + n_content_tokens]
        self.latent.append(lat.float().cpu())
        n_vv = self.n_frames - self.n_frames_seen if last else int(n_content_tokens / self.hz * 7.5)
        self.n_frames_seen += n_vv
        n_dc = n_content_tokens * DC_HOP
        # DualCodec decodes a few samples short: zero-pad the content to n_dc, then to whole frames
        content = _fit(_fit(window_audio[..., left_pad_samples:left_pad_samples + n_dc], n_dc), n_vv * VV_HOP)
        self.gate.append(silence_gate(content.reshape(1, -1), n_vv).cpu())
        if last:
            window_audio = _fit(_fit(window_audio, left_pad_samples + n_dc), left_pad_samples + n_vv * VV_HOP)
        return window_audio, n_vv

    def apply(self, z_prod: torch.Tensor) -> torch.Tensor:
        """z_prod [1, 3Kp/5, D] (collected VibeVoice frames) -> z = z_prod + gate * dz, same shape (float32, cpu)."""
        if z_prod.shape[1] != self.n_frames:
            raise RuntimeError(f"VibeVoice adapter: {z_prod.shape[1]} frames collected, {self.n_frames} expected")
        latent = F.pad(torch.cat(self.latent, dim=-1), (0, self.n_codes_padded - self.n_codes), mode="replicate")
        z = run_adapter(self.model, latent, z_prod, torch.cat(self.gate, dim=1))
        self.latent, self.gate = [], []
        return z


# ---------------------------------------------------------------- /stream (STREAM_VV_ADAPTER)
STREAM_LATENT_CONTEXT_CODES = 75  # DualCodec latent context left of a stream pass: 6 s, as /predict's decode windows


def _grid_up(n_codes: int) -> int:
    return -(-n_codes // GRID_CODES) * GRID_CODES


def check_stream_lookahead(model, pad_tokens: int) -> None:
    """Raise RuntimeError unless the stream holds back at least the adapter's lookahead (streaming_dc_pad_tokens codes past
    the last decoded frame; 5 by default, lookahead 1): correct_stream_frames reads that many codes past the frames it
    corrects.
    Called once at startup when /stream uses the adapter."""
    if pad_tokens < int(model.lookahead):
        raise RuntimeError(
            f"VibeVoice adapter on /stream: streaming_dc_pad_tokens is {pad_tokens} codes, less than the adapter's "
            f"lookahead of {int(model.lookahead)} codes; set STREAM_VV_ADAPTER=0 or raise the pad")


def stream_tail_audio(audio: torch.Tensor, n_codes: int) -> torch.Tensor:
    """The flush's DualCodec audio for the adapter's tail fix, as /predict's last window: cut or zero-padded to the codes'
    length (K x 1920 samples), then zero-padded to the 3 * Kp / 5 VibeVoice frames the adapter emits (Kp = 5 * ceil(K / 5))."""
    return _fit(_fit(audio, n_codes * DC_HOP), 3 * _grid_up(n_codes) // 5 * VV_HOP)


@torch.no_grad()
def correct_stream_frames(model, dc_inference, semantic_codes: list[int], acoustic_codes: list[list[int]],
                          audio: torch.Tensor, z_prod: torch.Tensor, a: int, b: int, final: bool = False,
                          first_code: int = 0) -> torch.Tensor:
    """/stream: z = z_prod + gate * dz for the VibeVoice frames [a, b) of a stream that holds K codes (from the request's
    first new code, the prefix stripped as on /predict), their DualCodec audio `audio` [1, 1, ~K x 1920] and its
    VibeVoice encode `z_prod` [1, >= b, D]. Returns [1, b - a, D] (float32, cpu) for the stream's VibeVoice decode.

    Frames [a, b) equal one adapter pass over the whole request on the same inputs (/predict's recipe):
      * left: the pass starts CONTEXT_CODES codes (more than the receptive field) before frame a's 5-code group, on the
        5-code / 3-frame grid, so z_prod, the latent and the gate sit on the request's global grid;
      * right: frame j reads the codes up to ceil(5 (j + 1) / 3) - 1 + lookahead. For b < the end of the request
        (final=False) these codes must be held: K >= ceil(5 b / 3) + lookahead (the stream lags streaming_dc_pad_tokens
        codes; check_stream_lookahead). Past them the pass is filled to the grid with copies: the backbone is causal and
        z_prod and the gate enter per frame, so nothing past frame b - 1's codes reaches frames < b.
      * final=True (the flush, b = every frame left): /predict's tail fix. The latent of the K codes is edge-padded to
        Kp = 5 * ceil(K / 5), and `audio` (and z_prod, its encode) must come from stream_tail_audio, so b = 3 * Kp / 5.
    The DualCodec latent is computed for codes [start - STREAM_LATENT_CONTEXT_CODES, K) and kept for the pass.
    first_code (a multiple of 5, at most the pass start): `audio` and `z_prod` start at that code instead of the
    stream's first (the lite backend's bounded streaming decode context); a, b stay stream frame indices."""
    n_codes = len(semantic_codes)
    lookahead = int(model.lookahead)
    if final:
        e = _grid_up(n_codes)
        if b != 3 * e // 5:
            raise RuntimeError(f"VibeVoice adapter on /stream: the flush ends at frame {b}, the tail fix at {3 * e // 5}")
    else:
        need = -(-5 * b // 3) + lookahead
        if need > n_codes:
            raise RuntimeError(f"VibeVoice adapter on /stream: frame {b - 1} reads {need} codes, the stream holds "
                               f"{n_codes} (streaming_dc_pad_tokens below the adapter's lookahead)")
        e = _grid_up(need)
    f0 = 3 * first_code // 5
    if f0 + z_prod.shape[1] < b:
        raise RuntimeError(f"VibeVoice adapter on /stream: {f0 + z_prod.shape[1]} VibeVoice frames, frame {b - 1} requested")
    s = max(0, GRID_CODES * (a // 3) - CONTEXT_CODES)
    if first_code % GRID_CODES or first_code > s:
        raise RuntimeError(f"VibeVoice adapter on /stream: decode window starts at code {first_code}, the pass at {s}")
    s_lat = max(0, s - STREAM_LATENT_CONTEXT_CODES)
    n = 3 * (e - s) // 5
    lat = dc_latent(dc_inference, semantic_codes[s_lat:], [c[s_lat:] for c in acoustic_codes])
    lat = lat[..., s - s_lat:min(e, n_codes) - s_lat].float().cpu()
    lat = F.pad(lat, (0, e - s - lat.shape[-1]), mode="replicate")
    gate = silence_gate(_fit(audio.reshape(1, -1)[:, (s - first_code) * DC_HOP:(e - first_code) * DC_HOP], n * VV_HOP),
                        n).cpu()
    zp = z_prod[:, 3 * s // 5 - f0:3 * e // 5 - f0].float().cpu()
    zp = F.pad(zp.transpose(1, 2), (0, n - zp.shape[1]), mode="replicate").transpose(1, 2)
    z = run_adapter(model, lat, zp, gate)
    return z[:, a - 3 * s // 5:b - 3 * s // 5]

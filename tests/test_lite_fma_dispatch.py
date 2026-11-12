"""Decode kernel choice per GPU: tensor-core kernels from compute capability 7.0 on, FMA kernels below (no GPU needed)."""
import pytest
import torch

from app.lite.gpu import arch, kernels, kernels_fma, tuning

TENSOR_CORE = [(7, 0), (7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)]
PRE_VOLTA = [(5, 2), (6, 0), (6, 1)]


@pytest.fixture
def gpus(monkeypatch):
    """Fake CUDA devices: index -> compute capability; `asked` records each capability lookup."""
    arch.reset()
    table: dict[int, tuple[int, int]] = {}
    asked: list[int] = []

    def get_device_capability(device=None):
        index = device.index if isinstance(device, torch.device) else device
        asked.append(index)
        return table[index]

    monkeypatch.setattr(torch.cuda, "get_device_capability", get_device_capability)
    monkeypatch.delenv(arch.FMA_KERNELS_ENV, raising=False)
    yield table, asked
    arch.reset()


def cuda(index: int) -> torch.device:
    return torch.device("cuda", index)


@pytest.mark.parametrize("capability", TENSOR_CORE)
def test_auto_keeps_tensor_core_kernels_from_volta_on(gpus, capability):
    table, _ = gpus
    table[0] = capability
    assert arch.use_fma_kernels(cuda(0)) is False


@pytest.mark.parametrize("capability", PRE_VOLTA)
def test_auto_uses_fma_kernels_before_volta(gpus, capability):
    table, _ = gpus
    table[0] = capability
    assert arch.use_fma_kernels(cuda(0)) is True


@pytest.mark.parametrize("raw, expected", [
    ("auto", None), ("", None), (" Auto ", None),
    ("on", True), ("ON", True), ("1", True),
    ("off", False), (" Off", False), ("0", False),
])
@pytest.mark.parametrize("capability", TENSOR_CORE + PRE_VOLTA)
def test_override(gpus, monkeypatch, capability, raw, expected):
    table, _ = gpus
    table[0] = capability
    monkeypatch.setenv(arch.FMA_KERNELS_ENV, raw)
    want = capability < (7, 0) if expected is None else expected
    assert arch.use_fma_kernels(cuda(0)) is want


def test_unknown_setting_is_an_error(gpus, monkeypatch):
    table, _ = gpus
    table[0] = (8, 6)
    monkeypatch.setenv(arch.FMA_KERNELS_ENV, "fma")
    with pytest.raises(ValueError, match="LITE_FMA_KERNELS"):
        arch.use_fma_kernels(cuda(0))
    assert arch.fma_kernels_setting({}) == "auto"
    assert arch.fma_kernels_setting({arch.FMA_KERNELS_ENV: "off"}) == "off"


def test_decision_follows_each_device_and_is_cached(gpus, monkeypatch):
    table, asked = gpus
    table.update({0: (8, 6), 1: (6, 1), 2: (8, 9), 3: (5, 2)})
    for _ in range(3):
        assert [arch.use_fma_kernels(cuda(i)) for i in range(4)] == [False, True, False, True]
    assert asked == [0, 1, 2, 3]
    monkeypatch.setenv(arch.FMA_KERNELS_ENV, "on")
    assert arch.use_fma_kernels(cuda(0)) is False
    arch.reset()
    assert arch.use_fma_kernels(cuda(0)) is True


def test_device_without_index_is_the_current_device(gpus, monkeypatch):
    table, asked = gpus
    table.update({0: (8, 6), 1: (6, 1)})
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert arch.use_fma_kernels(torch.device("cuda")) is True
    assert asked == [1]


def test_cpu_tensors_never_use_fma_kernels(gpus, monkeypatch):
    _, asked = gpus
    monkeypatch.setenv(arch.FMA_KERNELS_ENV, "on")
    assert arch.use_fma_kernels(torch.device("cpu")) is False
    assert asked == []


class Launches:
    """Stands in for Triton kernels: kernel[grid](*args, **meta) only records the launch."""

    def __init__(self):
        self.log: list[tuple[str, tuple, dict]] = []

    def kernel(self, name: str):
        log = self.log

        class Kernel:
            def __getitem__(self, grid):
                return lambda *args, **meta: log.append((name, grid, meta))

        return Kernel()

    def names(self) -> list[str]:
        return [name for name, _, _ in self.log]


@pytest.fixture
def launches(monkeypatch):
    rec = Launches()
    for module, name in ((kernels, "_qgemv_kernel"), (kernels, "_qgemv_reduce"), (kernels, "_decode_attn_partial"),
                         (kernels, "_decode_attn_combine"), (kernels_fma, "_qgemv_fma_kernel"),
                         (kernels_fma, "_qgemv_reduce"), (kernels_fma, "_decode_attn_partial_fma"),
                         (kernels_fma, "_decode_attn_combine")):
        monkeypatch.setattr(module, name, rec.kernel(name))
    monkeypatch.setattr(tuning, "gemv_config", lambda *args, **kw: (32, 1, 4))
    return rec


@pytest.fixture
def on_gpu0(monkeypatch):
    """Kernel call sites see CPU test tensors as living on fake GPU 0; records the devices they ask about."""
    asked = []

    def use_fma_kernels(device):
        asked.append(device)
        return arch.use_fma_kernels(cuda(0))

    monkeypatch.setattr(kernels, "use_fma_kernels", use_fma_kernels)
    return asked


def _qgemv(m: int = 1):
    x = torch.zeros(m, 256, dtype=torch.bfloat16)
    codes = torch.zeros(96, 128, dtype=torch.uint8)
    scales = torch.zeros(96, 4, dtype=torch.float16)
    out = kernels.qgemv(x, codes, scales, scales, 4, 64, 96)
    assert out.shape == (m, 96) and out.dtype == torch.bfloat16
    return x


def _attention(batch: int = 2):
    att = kernels.DecodeAttention(batch, 16, 8, 128, 512, 64, "cpu")
    q = torch.zeros(batch, 16, 128, dtype=torch.bfloat16)
    pool = torch.zeros(9, 64, 8, 128, dtype=torch.bfloat16)
    table = torch.zeros(batch, 8, dtype=torch.int32)
    out = att(q, pool, pool, table, torch.ones(batch, dtype=torch.int32))
    assert out.shape == q.shape
    return q


@pytest.mark.parametrize("capability", TENSOR_CORE)
def test_tensor_core_gpus_launch_only_the_stock_kernels(gpus, launches, on_gpu0, capability):
    table, _ = gpus
    table[0] = capability
    x = _qgemv(1)
    q = _attention(2)
    assert on_gpu0 == [x.device, q.device]
    assert launches.names() == ["_qgemv_kernel", "_decode_attn_partial", "_decode_attn_combine"]
    (_, grid, meta), (_, agrid, ameta), _ = launches.log
    assert grid == (3, 1) and meta["M_PAD"] == 16 and meta["BLOCK_K"] == 64 and meta["num_warps"] == 4
    assert agrid == (2, 8, 2) and ameta["G_PAD"] == 16 and ameta["BLOCKS_PER_SPLIT"] == 4 and ameta["num_warps"] == 4


@pytest.mark.parametrize("capability", PRE_VOLTA)
def test_pre_volta_gpus_launch_the_fma_kernels(gpus, launches, on_gpu0, capability):
    table, _ = gpus
    table[0] = capability
    _qgemv(3)
    _attention(2)
    names = launches.names()
    assert names[0] == "_qgemv_fma_kernel" and names[-2:] == ["_decode_attn_partial_fma", "_decode_attn_combine"]
    assert "_qgemv_kernel" not in names and "_decode_attn_partial" not in names
    attn = launches.log[-2]
    assert attn[1] == (2, 8, 512 // attn[2]["SPLIT_TOKENS"]) and attn[2]["G_PAD"] == 2


def test_off_launches_the_stock_kernels_on_pascal(gpus, launches, on_gpu0, monkeypatch):
    table, _ = gpus
    table[0] = (6, 1)
    monkeypatch.setenv(arch.FMA_KERNELS_ENV, "off")
    _qgemv(1)
    _attention(1)
    assert launches.names() == ["_qgemv_kernel", "_decode_attn_partial", "_decode_attn_combine"]


def test_unsupported_gemv_shapes_fall_back_to_the_stock_kernel(gpus, launches, on_gpu0, monkeypatch):
    table, _ = gpus
    table[0] = (6, 1)
    monkeypatch.setattr(kernels_fma, "qgemv_supported", lambda *args: False)
    _qgemv(1)
    assert launches.names() == ["_qgemv_kernel"]

"""Pick a PyTorch build that supports the installed NVIDIA GPU, before models load.

The default PyTorch wheels (CUDA 12.8) no longer ship kernels for Pascal and older
GPUs (compute capability below 7.0). The CUDA 12.6 wheels still do. They are the
``cu126`` extra in pyproject.toml, kept in a separate environment (.venv-cu126).

``check_torch_variant`` runs at the start of ``tontaube serve``, before any model is
downloaded and before CUDA is initialised. It never fails the launch by itself: any
error while probing is swallowed. It only acts when the GPU is certainly not covered
by the installed build, either by re-running the command in the cu126 environment or
by exiting with the commands to install it.

TONTAUBE_TORCH_VARIANT: ``auto`` (default), ``default`` (never check), or ``cu126``
(always use the cu126 build).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

VARIANT_ENV = "TONTAUBE_TORCH_VARIANT"
REEXEC_GUARD_ENV = "TONTAUBE_TORCH_REEXEC"
VARIANTS = ("auto", "default", "cu126")

# Architectures in the torch 2.9.1 cu126 wheels. A GPU is covered by a build when it
# can run one of its cubins: the same architecture, or an older minor version of the
# same major version (an sm_60 cubin runs on a 6.1 GPU).
CU126_ARCHS = ("sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90")

CU126_INDEX = "https://download.pytorch.org/whl/cu126"
CU126_ENV_DIR = ".venv-cu126"

Capability = tuple[int, int]


@dataclass(frozen=True)
class Gpu:
    name: str
    capability: Capability


@dataclass(frozen=True)
class Decision:
    """What to do: ``proceed``, ``reexec`` (run the cu126 environment) or ``exit``."""

    action: str
    message: str = ""


def parse_capability(text: str) -> Capability | None:
    """Parse ``"6.1"`` (nvidia-smi) into ``(6, 1)``; None for anything else."""
    match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", text or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def parse_gpu_query(output: str) -> list[Gpu]:
    """Parse ``nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader``."""
    gpus: list[Gpu] = []
    for line in (output or "").splitlines():
        name, sep, cap_text = line.rpartition(",")
        capability = parse_capability(cap_text) if sep else None
        if capability is not None:
            gpus.append(Gpu(name.strip() or "NVIDIA GPU", capability))
    return gpus


def select_gpu(gpus: Sequence[Gpu], visible_devices: str | None) -> Gpu | None:
    """The GPU the server will use: device 0 of the ones CUDA_VISIBLE_DEVICES leaves.

    nvidia-smi ignores CUDA_VISIBLE_DEVICES, so apply it here. Entries that are not
    plain indices (UUIDs, MIG) cannot be mapped; return None (no check) then.
    """
    if visible_devices is None:
        return gpus[0] if gpus else None
    first = visible_devices.split(",")[0].strip()
    if not first.isdigit():
        return None  # empty or -1 (no GPU), or an identifier we do not map
    index = int(first)
    return gpus[index] if index < len(gpus) else None


def _arch_numbers(archs: Sequence[str], prefix: str) -> list[int]:
    return [int(a[len(prefix):]) for a in archs if re.fullmatch(prefix + r"\d+", a)]


def build_supports(capability: Capability, archs: Sequence[str]) -> bool:
    """Whether a build with these ``torch.cuda.get_arch_list()`` entries runs the GPU."""
    major, minor = capability
    for number in _arch_numbers(archs, "sm_"):
        if number // 10 == major and number % 10 <= minor:
            return True
    # PTX (compute_XX) is JIT-compiled for any newer architecture.
    return any(number <= major * 10 + minor for number in _arch_numbers(archs, "compute_"))


def pip_commands() -> str:
    return (
        f"  pip:  python -m pip install --no-deps --require-hashes "
        f"--extra-index-url {CU126_INDEX} -r requirements-pip-cu126.lock\n"
        "        python -m pip install --no-deps --no-build-isolation ."
    )


def _install_hint() -> str:
    return (
        "Use the CUDA 12.6 build of PyTorch, from a source checkout with uv:\n"
        "  uv run --frozen --extra cu126 tontaube serve\n"
        "or with pip, in a fresh virtual environment:\n"
        f"{pip_commands()}\n"
        f"Set {VARIANT_ENV}=default to skip this check, or LITE_DEVICE=cpu to run on the CPU."
    )


def decide(
    *,
    variant: str,
    gpu: Gpu | None,
    installed_archs: Sequence[str],
    installed_is_cu126: bool,
    reexec_done: bool,
    can_reexec: bool,
) -> Decision:
    """Pure decision from what was detected; see the module docstring."""
    if variant == "default" or gpu is None:
        return Decision("proceed")
    cap = f"{gpu.capability[0]}.{gpu.capability[1]}"
    label = f"{gpu.name} (compute capability {cap})"

    if variant == "cu126" and not installed_is_cu126:
        if reexec_done:
            return Decision("proceed")  # the loop guard: never re-run twice
        if can_reexec:
            return Decision("reexec", f"{VARIANT_ENV}=cu126: running with the CUDA 12.6 PyTorch build.")
        return Decision(
            "exit",
            f"{VARIANT_ENV}=cu126 is set but this environment has the default PyTorch build.\n"
            + _install_hint(),
        )

    if not installed_archs or build_supports(gpu.capability, installed_archs):
        return Decision("proceed")

    if build_supports(gpu.capability, CU126_ARCHS) and not installed_is_cu126:
        reason = (
            f"{label} is not supported by the installed PyTorch build "
            f"({', '.join(installed_archs)}), but the CUDA 12.6 build supports it."
        )
        if reexec_done:
            return Decision("exit", f"{reason}\nThe CUDA 12.6 environment did not take effect.\n" + _install_hint())
        if can_reexec:
            return Decision(
                "reexec",
                f"Detected {label}. The default PyTorch build does not support it; "
                f"starting in the CUDA 12.6 environment ({CU126_ENV_DIR}). "
                "The first start downloads several GB.",
            )
        return Decision("exit", f"{reason}\n" + _install_hint())

    if installed_is_cu126 and max(_arch_numbers(CU126_ARCHS, "sm_"), default=0) < gpu.capability[0] * 10:
        return Decision(
            "exit",
            f"{label} is newer than the installed CUDA 12.6 PyTorch build supports.\n"
            f"Use the default build: unset {VARIANT_ENV}, or run `uv run --frozen tontaube serve` "
            "without --extra cu126.",
        )
    return Decision(
        "exit",
        f"{label} is too old for the PyTorch builds Tontaube supports "
        f"(installed: {', '.join(installed_archs)}; CUDA 12.6 build: {', '.join(CU126_ARCHS)}).\n"
        "Run on the CPU instead with LITE_DEVICE=cpu "
        f"(or set {VARIANT_ENV}=default to try this GPU anyway).",
    )


def variant_setting(environ: Mapping[str, str]) -> str:
    variant = environ.get(VARIANT_ENV, "auto").strip().lower() or "auto"
    if variant not in VARIANTS:
        raise ValueError(f"unsupported {VARIANT_ENV} {variant!r}; expected one of {', '.join(VARIANTS)}")
    return variant


def checkout_root() -> Path | None:
    """The repository root when running from a uv-managed source checkout, else None."""
    root = Path(__file__).resolve().parent.parent
    if (root / "pyproject.toml").is_file() and (root / "uv.lock").is_file() and (root / "app" / "cli.py").is_file():
        return root
    return None


def reexec_command(
    root: Path,
    argv: Sequence[str],
    environ: Mapping[str, str],
    *,
    python_version: str,
    uv: str = "uv",
) -> tuple[list[str], dict[str, str]]:
    """The cu126 command and environment: a separate project environment, loop guard set."""
    command = [
        uv, "run", "--frozen", "--project", str(root), "--python", python_version,
        "--extra", "cu126", "tontaube", *argv,
    ]
    env = dict(environ)
    env.pop("VIRTUAL_ENV", None)  # an activated .venv would only make uv warn
    env["UV_PROJECT_ENVIRONMENT"] = str(root / CU126_ENV_DIR)
    env[REEXEC_GUARD_ENV] = "1"
    return command, env


def query_gpus_nvidia_smi(run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> list[Gpu]:
    """GPUs and compute capabilities from nvidia-smi; no CUDA context is created."""
    if not shutil.which("nvidia-smi"):
        return []
    result = run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    return parse_gpu_query(result.stdout) if result.returncode == 0 else []


def query_gpus_nvml() -> list[Gpu]:
    """Fallback for nvidia-smi that is missing or too old to know ``compute_cap``."""
    import pynvml  # nvidia-ml-py, a dependency of vLLM

    pynvml.nvmlInit()
    try:
        gpus = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            name = name.decode() if isinstance(name, bytes) else name
            gpus.append(Gpu(name, tuple(pynvml.nvmlDeviceGetCudaComputeCapability(handle))))
        return gpus
    finally:
        pynvml.nvmlShutdown()


def detect_gpus() -> list[Gpu]:
    try:
        gpus = query_gpus_nvidia_smi()
    except Exception:
        gpus = []
    if gpus:
        return gpus
    if not (Path("/proc/driver/nvidia").exists() or shutil.which("nvidia-smi")):
        return []  # no NVIDIA driver here: nothing to probe, and no import cost
    try:
        return query_gpus_nvml()
    except Exception:
        return []


def installed_torch_build() -> tuple[list[str], bool]:
    """(``torch.cuda.get_arch_list()``, is it the cu126 build). Does not initialise CUDA."""
    import torch

    return list(torch.cuda.get_arch_list()), str(torch.__version__).endswith("+cu126")


def evaluate(
    argv: Sequence[str],
    environ: Mapping[str, str],
    *,
    detect: Callable[[], list[Gpu]] = detect_gpus,
    torch_build: Callable[[], tuple[list[str], bool]] = installed_torch_build,
    root: Callable[[], Path | None] = checkout_root,
    which: Callable[[str], str | None] = shutil.which,
) -> tuple[Decision, tuple[list[str], dict[str, str]] | None]:
    """Detect and decide. Returns the decision and, for ``reexec``, the command to run."""
    variant = variant_setting(environ)
    if variant == "default":
        return Decision("proceed"), None
    if environ.get("LITE_DEVICE", "").strip().lower() == "cpu":
        return Decision("proceed"), None
    if environ.get("TTS_BACKEND", "lite").strip().lower() == "vllm":
        return Decision("proceed"), None  # vLLM ships kernels for its own torch build

    gpu = select_gpu(detect(), environ.get("CUDA_VISIBLE_DEVICES"))
    if gpu is None:
        return Decision("proceed"), None
    archs, is_cu126 = torch_build()
    checkout = root()
    uv = which("uv")
    can_reexec = checkout is not None and uv is not None
    decision = decide(
        variant=variant,
        gpu=gpu,
        installed_archs=archs,
        installed_is_cu126=is_cu126,
        reexec_done=environ.get(REEXEC_GUARD_ENV) == "1",
        can_reexec=can_reexec,
    )
    if decision.action == "reexec" and checkout is not None and uv is not None:
        command = reexec_command(
            checkout, argv, environ,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}", uv=uv,
        )
        return decision, command
    return decision, None


def check_torch_variant(argv: Sequence[str], environ: Mapping[str, str] | None = None) -> None:
    """Entry point for the CLI. Returns, exits with a message, or replaces the process."""
    environ = os.environ if environ is None else environ
    try:
        variant_setting(environ)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc  # a misspelt setting is not a probe failure
    try:
        decision, command = evaluate(argv, environ)
    except Exception:
        return  # a failed probe must never block the launch
    if decision.action == "exit":
        raise SystemExit(decision.message)
    if decision.action == "reexec" and command is not None:
        print(decision.message, flush=True)
        sys.stdout.flush()
        args, env = command
        try:
            os.execvpe(args[0], args, env)
        except OSError as exc:
            print(f"Could not start the CUDA 12.6 environment ({exc}); continuing.", flush=True)

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app import cli, torch_variant as tv
from app.torch_variant import CU126_ARCHS, Decision, Gpu

DEFAULT_ARCHS = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
PASCAL = Gpu("NVIDIA GeForce GTX 1070 Ti", (6, 1))
AMPERE = Gpu("NVIDIA GeForce RTX 3090", (8, 6))
KEPLER = Gpu("NVIDIA Tesla K80", (3, 7))
ROOT = Path("/srv/tontaube")


def decide(**overrides) -> Decision:
    values = dict(
        variant="auto",
        gpu=PASCAL,
        installed_archs=DEFAULT_ARCHS,
        installed_is_cu126=False,
        reexec_done=False,
        can_reexec=True,
    )
    values.update(overrides)
    return tv.decide(**values)


def test_parse_capability():
    assert tv.parse_capability("6.1\n") == (6, 1)
    assert tv.parse_capability(" 12.0 ") == (12, 0)
    for bad in ("", "[N/A]", "6", "6.1.2", "a.b", None):
        assert tv.parse_capability(bad) is None


def test_parse_gpu_query_handles_names_and_garbage():
    output = "NVIDIA GeForce GTX 1070 Ti, 6.1\nNVIDIA A100, SXM4, 8.0\nno-comma-here\nX, [N/A]\n"
    assert tv.parse_gpu_query(output) == [Gpu("NVIDIA GeForce GTX 1070 Ti", (6, 1)), Gpu("NVIDIA A100, SXM4", (8, 0))]
    assert tv.parse_gpu_query("") == []
    assert tv.parse_gpu_query('Field "compute_cap" is not a valid field to query.\n') == []


def test_select_gpu_follows_cuda_visible_devices():
    gpus = [PASCAL, AMPERE]
    assert tv.select_gpu(gpus, None) == PASCAL
    assert tv.select_gpu(gpus, "1") == AMPERE
    assert tv.select_gpu(gpus, "1,0") == AMPERE
    assert tv.select_gpu(gpus, "") is None
    assert tv.select_gpu(gpus, "-1") is None
    assert tv.select_gpu(gpus, "GPU-1234") is None
    assert tv.select_gpu(gpus, "5") is None
    assert tv.select_gpu([], None) is None


def test_build_supports_compares_with_the_installed_arch_list():
    assert not tv.build_supports((6, 1), DEFAULT_ARCHS)
    assert tv.build_supports((7, 0), DEFAULT_ARCHS)
    assert tv.build_supports((8, 9), DEFAULT_ARCHS)  # an sm_86 cubin runs on 8.9
    assert not tv.build_supports((7, 0), ["sm_75"])  # a newer minor does not run on an older GPU
    assert tv.build_supports((6, 1), CU126_ARCHS)  # via sm_60
    assert tv.build_supports((5, 2), CU126_ARCHS)
    assert not tv.build_supports((3, 7), CU126_ARCHS)
    assert tv.build_supports((9, 0), ["sm_80", "compute_80"])  # PTX can be JIT-compiled
    assert not tv.build_supports((6, 1), ["compute_70"])


def test_supported_gpu_proceeds():
    assert decide(gpu=AMPERE) == Decision("proceed")


def test_no_gpu_or_default_variant_proceeds():
    assert decide(gpu=None).action == "proceed"
    assert decide(variant="default").action == "proceed"
    assert decide(installed_archs=[]).action == "proceed"  # torch cannot tell: do nothing


def test_unsupported_gpu_reexecs_when_possible():
    decision = decide()
    assert decision.action == "reexec"
    assert "GTX 1070 Ti" in decision.message and "6.1" in decision.message and ".venv-cu126" in decision.message


def test_unsupported_gpu_exits_with_both_commands_when_it_cannot_reexec():
    decision = decide(can_reexec=False)
    assert decision.action == "exit"
    assert "GTX 1070 Ti" in decision.message
    assert "uv run --frozen --extra cu126 tontaube serve" in decision.message
    assert "requirements-pip-cu126.lock" in decision.message
    assert "https://download.pytorch.org/whl/cu126" in decision.message


def test_the_guard_prevents_a_second_reexec():
    assert decide(reexec_done=True).action == "exit"


def test_gpu_too_old_for_cu126_suggests_the_cpu():
    decision = decide(gpu=KEPLER)
    assert decision.action == "exit"
    assert "too old" in decision.message and "LITE_DEVICE=cpu" in decision.message
    assert "--extra cu126" not in decision.message


def test_gpu_too_old_inside_the_cu126_build_suggests_the_cpu():
    decision = decide(gpu=KEPLER, installed_archs=list(CU126_ARCHS), installed_is_cu126=True)
    assert decision.action == "exit" and "LITE_DEVICE=cpu" in decision.message


def test_cu126_build_supporting_the_gpu_proceeds():
    assert decide(installed_archs=list(CU126_ARCHS), installed_is_cu126=True) == Decision("proceed")


def test_gpu_too_new_for_a_forced_cu126_build_points_to_the_default_build():
    blackwell = Gpu("NVIDIA GeForce RTX 5090", (12, 0))
    decision = decide(gpu=blackwell, installed_archs=list(CU126_ARCHS), installed_is_cu126=True)
    assert decision.action == "exit" and "newer" in decision.message


def test_forced_cu126_variant():
    assert decide(variant="cu126", gpu=AMPERE).action == "reexec"
    assert decide(variant="cu126", gpu=AMPERE, can_reexec=False).action == "exit"
    assert decide(variant="cu126", gpu=AMPERE, reexec_done=True).action == "proceed"
    assert decide(variant="cu126", gpu=AMPERE, installed_archs=list(CU126_ARCHS), installed_is_cu126=True).action == "proceed"


def test_reexec_command_uses_a_separate_environment_and_a_guard():
    command, env = tv.reexec_command(
        ROOT, ["serve", "--port", "9000"], {"PATH": "/bin", "HF_HOME": "/hf", "VIRTUAL_ENV": ".venv"},
        python_version="3.12", uv="/bin/uv",
    )
    assert command == [
        "/bin/uv", "run", "--frozen", "--project", "/srv/tontaube", "--python", "3.12",
        "--extra", "cu126", "tontaube", "serve", "--port", "9000",
    ]
    assert env["UV_PROJECT_ENVIRONMENT"] == "/srv/tontaube/.venv-cu126"
    assert env[tv.REEXEC_GUARD_ENV] == "1"
    assert env["HF_HOME"] == "/hf"  # the environment is passed through
    assert "VIRTUAL_ENV" not in env


def evaluate(environ=None, *, gpus=(PASCAL,), archs=DEFAULT_ARCHS, is_cu126=False, root=ROOT, uv="/bin/uv"):
    detected = []

    def detect():
        detected.append(True)
        return list(gpus)

    result = tv.evaluate(
        ["serve"],
        environ or {},
        detect=detect,
        torch_build=lambda: (list(archs), is_cu126),
        root=lambda: root,
        which=lambda name: uv,
    )
    return result, bool(detected)


def test_evaluate_reexecs_for_pascal_in_a_checkout():
    (decision, command), _ = evaluate()
    assert decision.action == "reexec"
    assert command[0][:2] == ["/bin/uv", "run"] and command[1][tv.REEXEC_GUARD_ENV] == "1"


def test_evaluate_exits_without_uv_or_checkout():
    (decision, command), _ = evaluate(uv=None)
    assert decision.action == "exit" and command is None
    (decision, command), _ = evaluate(root=None)
    assert decision.action == "exit" and command is None


@pytest.mark.parametrize(
    "environ",
    [
        {"TONTAUBE_TORCH_VARIANT": "default"},
        {"LITE_DEVICE": "cpu"},
        {"LITE_DEVICE": " CPU "},
        {"TTS_BACKEND": "vllm"},
        {"CUDA_VISIBLE_DEVICES": ""},
    ],
)
def test_evaluate_does_nothing_when_the_gpu_is_not_used(environ):
    (decision, command), _ = evaluate(environ)
    assert decision == Decision("proceed") and command is None


def test_cpu_and_default_variant_never_probe_the_gpu():
    for environ in ({"LITE_DEVICE": "cpu"}, {"TONTAUBE_TORCH_VARIANT": "default"}):
        _, probed = evaluate(environ)
        assert not probed


def test_evaluate_without_a_gpu_proceeds():
    (decision, _), _ = evaluate(gpus=())
    assert decision == Decision("proceed")


def test_nvidia_smi_parsing_and_failures():
    def fake_run(stdout, returncode=0):
        return lambda *a, **k: SimpleNamespace(stdout=stdout, returncode=returncode)

    original = tv.shutil.which
    tv.shutil.which = lambda name: "/usr/bin/nvidia-smi"
    try:
        assert tv.query_gpus_nvidia_smi(fake_run("NVIDIA GeForce GTX 1070 Ti, 6.1\n")) == [PASCAL]
        assert tv.query_gpus_nvidia_smi(fake_run('Field "compute_cap" is not a valid field\n', 2)) == []
    finally:
        tv.shutil.which = original


def test_detect_falls_back_to_nvml_when_nvidia_smi_does_not_know_the_field(monkeypatch):
    monkeypatch.setattr(tv, "query_gpus_nvidia_smi", lambda: [])
    monkeypatch.setattr(tv.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(tv, "query_gpus_nvml", lambda: [PASCAL])
    assert tv.detect_gpus() == [PASCAL]


def test_detect_swallows_probe_errors(monkeypatch):
    def boom():
        raise OSError("no nvidia-smi")

    monkeypatch.setattr(tv, "query_gpus_nvidia_smi", boom)
    monkeypatch.setattr(tv, "query_gpus_nvml", boom)
    monkeypatch.setattr(tv.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    assert tv.detect_gpus() == []


def test_check_swallows_errors_and_returns(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("torch is broken")

    monkeypatch.setattr(tv, "evaluate", boom)
    tv.check_torch_variant(["serve"], {})  # returns normally


def test_check_rejects_an_unknown_variant():
    with pytest.raises(SystemExit, match="TONTAUBE_TORCH_VARIANT"):
        tv.check_torch_variant(["serve"], {"TONTAUBE_TORCH_VARIANT": "cu121"})


def test_check_prints_a_line_and_replaces_the_process(monkeypatch, capsys):
    calls = []
    command = (["/bin/uv", "run"], {"X": "1"})
    monkeypatch.setattr(tv, "evaluate", lambda argv, environ: (Decision("reexec", "Detected a GPU."), command))
    monkeypatch.setattr(tv.os, "execvpe", lambda *args: calls.append(args))

    tv.check_torch_variant(["serve"], {})

    assert calls == [("/bin/uv", ["/bin/uv", "run"], {"X": "1"})]
    assert capsys.readouterr().out == "Detected a GPU.\n"


def test_check_continues_when_the_exec_fails(monkeypatch, capsys):
    def fail(*args):
        raise OSError("exec format error")

    monkeypatch.setattr(tv, "evaluate", lambda argv, environ: (Decision("reexec", "msg"), (["uv"], {})))
    monkeypatch.setattr(tv.os, "execvpe", fail)
    tv.check_torch_variant(["serve"], {})
    assert "continuing" in capsys.readouterr().out


def test_check_exits_with_the_message(monkeypatch):
    monkeypatch.setattr(tv, "evaluate", lambda argv, environ: (Decision("exit", "GPU too old"), None))
    with pytest.raises(SystemExit, match="GPU too old"):
        tv.check_torch_variant(["serve"], {})


def test_serve_runs_the_check_before_loading_the_app(monkeypatch):
    order: list[str] = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "app.main",
        SimpleNamespace(main=lambda **kwargs: order.append("serve")),
    )
    monkeypatch.setattr(tv, "check_torch_variant", lambda argv: order.append(f"check {argv}"))

    cli.main(["serve", "--port", "9000"])

    assert order == ["check ['serve', '--port', '9000']", "serve"]


def test_preflight_and_ui_do_not_run_the_check(monkeypatch):
    called = []
    monkeypatch.setattr(tv, "check_torch_variant", lambda argv: called.append(argv))
    monkeypatch.setitem(__import__("sys").modules, "app.preflight", SimpleNamespace(main=lambda: None))
    cli.main(["preflight"])
    assert called == []

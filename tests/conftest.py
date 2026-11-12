"""Mock heavy dependencies so app modules can be imported without GPU/models."""
import sys
from unittest.mock import MagicMock

# Must run before any app.* import
_MOCK_MODULES = [
    'vllm', 'vllm.engine', 'vllm.engine.arg_utils', 'vllm.engine.async_llm_engine', 'vllm.inputs',
    'vllm.config', 'vllm.config.compilation',
    'vllm.model_executor', 'vllm.model_executor.models', 'vllm.model_executor.models.qwen3',
    'vllm.model_executor.models.utils',
    'vllm.model_executor.model_loader', 'vllm.model_executor.model_loader.weight_utils',
    'vllm.sequence',
    'dualcodec',
    'torchaudio',
    'soundfile', 'transformers',
    'pydub',
    'VibeVoice', 'VibeVoice.vibevoice', 'VibeVoice.vibevoice.modular',
    'VibeVoice.vibevoice.modular.modular_vibevoice_tokenizer',
]
for mod in _MOCK_MODULES:
    sys.modules.setdefault(mod, MagicMock())


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_torch_variant_switch(monkeypatch):
    """Tests that run `cli.main(["serve"])` must never probe the GPU or re-exec into uv."""
    monkeypatch.setenv("TONTAUBE_TORCH_VARIANT", "default")

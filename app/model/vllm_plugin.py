"""vLLM plugin to register custom model architectures."""


def register():
    from vllm import ModelRegistry

    if "Qwen3DualCodecForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "Qwen3DualCodecForCausalLM",
            "app.model.vllm_model:Qwen3DualCodecForCausalLM",
        )

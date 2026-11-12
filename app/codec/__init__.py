"""Memory- and speed-optimized implementations around the codec stack.

DualCodec's W2V-BERT encoder (w2v), host-resident weights (offload), VibeVoice
with int8 FFN layers on the GPU (vibevoice_quant) and VibeVoice on the CPU
(vibevoice_cpu). Each produces the same outputs as the reference modules up
to the stated quantization.
"""

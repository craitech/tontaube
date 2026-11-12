"""CPU execution of the lite backend (x86-64 with AVX2).

The same quantized weights as on the GPU (bundles or plan strings) run with
the AVX2 kernels of app/native (lm_kernels.cpp): decode steps use int4/int8 weight x
int16 activation dot products (activations quantized per 64 inputs) and a
flash-decoding attention kernel over a bf16 KV cache; prompt processing
dequantizes L3-sized blocks of weight rows for fp32 MKL matmuls.

Each codebook model keeps a dense KV cache for one sequence, and a new request
reuses the KV of the longest prefix it shares with the previous one (voice
prompt block, retries). The engine runs requests one at a time per model and
one model step at a time across models, because every step already uses all
cores.
"""

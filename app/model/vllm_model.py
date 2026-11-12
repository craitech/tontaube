"""vLLM-compatible TontaubeV1 model.

Input layout for a target_codebook=K model (no SOS):
  - [cb0 prompt][cb1 prompt]...[cbK prompt]: each block at positions 1..P
  - [text with text markers]: from P+1 via the aligned marker recurrence
    (marker_pos = cur + max(n_text + chunk_realign_offset, n_audio))
  - [PAD][cb0 audio (+markers) (+EOS)]: PAD at P, audio from P+1
  - ([PAD][cb_i audio]) x K: audio aligned to cb0 non-marker positions

Since RoPE only depends on relative position differences, all complex
position calculations are done during prefill and shifted so that the
last raw prefill position = seq_len - 1. A fixed coordinate translation is
then applied to every prefill and decode position.
"""
from typing import Optional

import torch
import torch.nn as nn
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.sequence import IntermediateTensors

from app.model.positions import (  # noqa: F401  (re-exported for callers)
    ACOUSTIC_VOCAB_SIZE,
    AUDIO_MARKER_LOCAL_IDX,
    SEMANTIC_VOCAB_SIZE,
    LayoutIds,
    front_prompt_positions,
    get_codebook_offset,
    get_codebook_vocab_size,
)


class AudioTokenHead(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, output_size)
        self.act = nn.Mish()

    def forward(self, hidden_states):
        x = self.act(self.fc1(hidden_states))
        return self.fc2(x)


class Qwen3DualCodecForCausalLM(nn.Module):
    """TontaubeV1 front-prompt scheme (see module docstring).

    Serialized layout (target_codebook=K):
        Token:    prompt_cb0..cbK  text...  PAD  cb0_audio...  [EOS]  (PAD cb_i_audio...) x K
        Position: 1..P per block   P+1..    P    P+1..                (P    cb0-aligned)

    All position calculations run during prefill and are shifted so the last
    raw prefill position = seq_len - 1. Prefill and decode positions then
    receive the same fixed coordinate translation.
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config, prefix: str = "", **kwargs):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = Qwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.dc_base_id = getattr(self.config, 'audio_token_start_id', None)
        self.n_codebooks = getattr(self.config, 'n_codebooks', None)
        self.audio_head_hidden = getattr(self.config, 'audio_head_hidden', None)
        self.target_codebook = getattr(self.config, 'target_codebook', None)
        self.pad_token_id = getattr(self.config, 'pad_token_id', None)

        self.eos_global_id = getattr(self.config, 'eos_token_id', None)
        self.audio_marker_global_id = getattr(self.config, 'audio_marker_token_id', None)
        self.text_marker_global_id = getattr(self.config, 'text_marker_token_id', None)
        self.chunk_realign_offset = getattr(self.config, 'chunk_realign_offset', None)
        required = {
            "audio_token_start_id": self.dc_base_id,
            "n_codebooks": self.n_codebooks,
            "audio_head_hidden": self.audio_head_hidden,
            "target_codebook": self.target_codebook,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_global_id,
            "audio_marker_token_id": self.audio_marker_global_id,
            "text_marker_token_id": self.text_marker_global_id,
            "chunk_realign_offset": self.chunk_realign_offset,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"config.json is missing required fields: {', '.join(missing)}")
        if self.chunk_realign_offset != 25:
            raise ValueError(
                f"TontaubeV1 requires chunk_realign_offset=25, got {self.chunk_realign_offset}"
            )

        self.audio_head = AudioTokenHead(
            self.config.hidden_size,
            intermediate_size=self.audio_head_hidden,
            output_size=get_codebook_vocab_size(self.target_codebook)
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def _compute_prefill_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute raw prefill positions, aligned to vLLM's decode indices."""
        ids = input_ids.flatten().tolist()
        n = len(ids)

        positions, gap = self._front_prompt_position_scheme(ids, n)

        target_last = n - 1 - gap
        shift = target_last - positions[-1]
        positions = [p + shift for p in positions]

        return torch.tensor(positions, device=input_ids.device, dtype=torch.long)

    def _front_prompt_position_scheme(self, ids, n):
        """Positions for the TontaubeV1 front-prompt scheme (see app.model.positions)."""
        assert n == len(ids)
        layout = LayoutIds(
            dc_base_id=self.dc_base_id,
            pad_token_id=self.pad_token_id,
            text_marker_id=self.text_marker_global_id,
            audio_marker_id=self.audio_marker_global_id,
            eos_id=self.eos_global_id,
            chunk_realign_offset=self.chunk_realign_offset,
            target_codebook=self.target_codebook,
        )
        return front_prompt_positions(ids, layout)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:

        # During CUDA graph capture: raw identity positions (tensor-only ops).
        # On CPU-only torch, the cuda call raises; treat as "not capturing".
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            custom_positions = positions
        else:
            n_tokens = positions.numel()
            pos_flat = positions.flatten()
            ids_flat = input_ids.flatten()

            zero_mask = (pos_flat == 0)
            if not zero_mask.any():
                # Pure decode batch: raw identity positions
                custom_positions = positions
            else:
                zero_indices = zero_mask.nonzero(as_tuple=True)[0]
                n_seqs = zero_indices.numel()
                group_id = zero_mask.cumsum(0) - 1           # -1 before first zero
                group_clamped = group_id.clamp_min(0)
                starts_per_tok = zero_indices[group_clamped]
                arange_n = torch.arange(n_tokens, device=pos_flat.device)
                expected = arange_n - starts_per_tok
                is_in_prefill = (pos_flat == expected) & (group_id >= 0)

                # One bulk D2H instead of per-token .item() syncs.
                starts_cpu = zero_indices.tolist()
                is_prefill_cpu = is_in_prefill.tolist()

                custom_positions = positions.clone()

                for idx, start in enumerate(starts_cpu):
                    next_start = starts_cpu[idx + 1] if idx + 1 < len(starts_cpu) else n_tokens
                    end = start + 1
                    while end < next_start and is_prefill_cpu[end]:
                        end += 1

                    seq_ids = ids_flat[start:end]

                    # Skip warmup dummy sequences (real prefills always contain
                    # the cb0-row PAD; vLLM profiling dummies do not)
                    if self.pad_token_id not in seq_ids:
                        continue

                    shifted = self._compute_prefill_positions(seq_ids.unsqueeze(0))
                    custom_positions[start:end] = shifted

                if input_ids.dim() > 1:
                    custom_positions = custom_positions.view(input_ids.shape)

        # Keep custom prefills and decode positions in the same
        # nonnegative coordinate frame without changing RoPE differences.
        custom_positions = custom_positions + self.chunk_realign_offset

        max_pos = getattr(self.config, 'max_position_embeddings', 32768)
        custom_positions = custom_positions.clamp(0, max_pos - 1)

        hidden_states = self.model(
            input_ids, custom_positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        cb = self.target_codebook
        audio_logits = self.audio_head(hidden_states)

        full_vocab_logits = torch.full(
            (audio_logits.size(0), self.config.vocab_size),
            -1e9,
            dtype=audio_logits.dtype,
            device=audio_logits.device
        )

        offset = get_codebook_offset(cb)

        if cb == 0:
            full_vocab_logits[
                :, self.dc_base_id:self.dc_base_id + SEMANTIC_VOCAB_SIZE
            ] = audio_logits[:, :SEMANTIC_VOCAB_SIZE]
            full_vocab_logits[:, self.eos_global_id] = audio_logits[:, SEMANTIC_VOCAB_SIZE]
            full_vocab_logits[:, self.audio_marker_global_id] = audio_logits[:, AUDIO_MARKER_LOCAL_IDX]
        else:
            start_id = self.dc_base_id + offset
            full_vocab_logits[:, start_id:start_id + ACOUSTIC_VOCAB_SIZE] = audio_logits[:, :ACOUSTIC_VOCAB_SIZE]

        return full_vocab_logits

    def load_weights(self, weights):
        loaded_weights = AutoWeightsLoader(self).load_weights(weights)
        params_dict = dict(self.named_parameters())
        expected_audio_weights = {
            name for name in params_dict if name.startswith("audio_head.")
        }
        missing = expected_audio_weights - loaded_weights
        if missing:
            raise ValueError(
                f"model bundle is missing audio-head weights for cb{self.target_codebook}: "
                f"{', '.join(sorted(missing))}"
            )
        return loaded_weights

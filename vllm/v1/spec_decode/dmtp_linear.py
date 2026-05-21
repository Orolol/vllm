# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dmtp linear-chain proposer (P1).

Sibling of DFlashProposer — does NOT inherit DFlash's cross-attention
infrastructure. The Dmtp drafter is a single self-attention layer over
``Q = 1 + K`` query tokens per request:

* position 0: the verified bonus token (``next_token_id``) embedded with
  the target's real ``embed_tokens``
* positions 1..K: ``mask_token_id``, substituted in-model with the
  learned ``mask_embedding``

Conditioning on the anchor target hidden state (last input position's
final hidden) is broadcast across all Q drafter positions via
``fc(concat(norm(embeds), norm(anchor)))``. No precomputed context K/V.

Sampling at positions 0..K-1 produces K speculative tokens for output
positions 1..K (causal self-attention: hidden at logical position ``i``
predicts logical position ``i+1``). This matches the standalone Dmtp
inference layout in
``nextn_to_dflash/integrated_inference.py:integrated_generate``.

The tree-verify variant (P2) is a separate class that builds a per-request
tree from the K logits using FlexAttention.
"""

from __future__ import annotations

from typing import Any

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.dmtp_timing import time_region
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

logger = init_logger(__name__)


class DmtpLinearProposer(SpecDecodeBaseProposer):
    """Linear-chain Dmtp proposer: bonus + K masks → K argmax tokens."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dmtp_linear"
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
        # Dmtp populates self.hidden_states directly with the anchor target
        # hidden replicated Q times — no learned mask_hidden, so disable the
        # buffer the base class would otherwise expect to copy from the model.
        self.parallel_drafting_hidden_state_tensor = None
        self.num_query_per_req = 1 + self.num_speculative_tokens
        self.max_query_tokens = self.max_batch_size * self.num_query_per_req

    @override
    def _warn_if_multimodal(self):
        # Dmtp targets Qwen3-family text-only models; suppress the warning.
        pass

    @override
    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with time_region("linear/_greedy_sample"):
            with time_region("linear/_greedy_sample.compute_logits"):
                logits = self.model.compute_logits(hidden_states)
            with time_region("linear/_greedy_sample.argmax"):
                tokens = logits.argmax(dim=-1)
        return tokens

    @override
    def model_returns_tuple(self) -> bool:
        # Our model.forward returns a single tensor, not a (last, all) tuple.
        return False

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        batch_size = cad.batch_size()
        K = self.num_speculative_tokens
        Q = self.num_query_per_req  # 1 bonus + K masks
        num_query_total = batch_size * Q
        device = self.device

        # End-of-input index per request (last accepted token in target view).
        query_end_loc = cad.query_start_loc[1:] - 1
        if num_rejected_tokens_gpu is not None:
            query_end_loc = query_end_loc - num_rejected_tokens_gpu

        # Anchor: target's last hidden state per request — the hidden that
        # generated next_token_ids on the target's just-finished forward.
        # Replicated across all Q drafter positions (matches standalone
        # Dmtp inference, integrated_inference.py:138).
        anchor_hidden = target_hidden_states[query_end_loc]  # [batch, H]
        H = anchor_hidden.shape[-1]
        anchor_rep = (
            anchor_hidden.unsqueeze(1).expand(-1, Q, -1).reshape(num_query_total, H)
        )

        # Positions: per request, Q consecutive positions starting at
        # bonus_pos = last_input_pos + 1.
        if target_positions.dim() == 1:
            last_input_pos = target_positions[query_end_loc]  # [batch]
        else:
            last_input_pos = target_positions[..., query_end_loc]  # [..., batch]
        offsets = torch.arange(Q, device=device, dtype=last_input_pos.dtype)
        query_positions = (last_input_pos + 1).unsqueeze(-1) + offsets
        if query_positions.dim() == 2:
            query_positions_flat = query_positions.reshape(num_query_total)
        else:
            query_positions_flat = query_positions.reshape(
                query_positions.shape[0], num_query_total
            )

        # Input ids: [bonus, mask, mask, ..., mask] per request, flattened.
        input_ids_block = torch.full(
            (batch_size, Q),
            self.parallel_drafting_token_id,
            dtype=self.input_ids.dtype,
            device=device,
        )
        input_ids_block[:, 0] = next_token_ids.to(input_ids_block.dtype)
        self.input_ids[:num_query_total] = input_ids_block.reshape(num_query_total)

        # _set_positions handles 1D vs M-RoPE / xdrope leading dims.
        self._set_positions(num_query_total, query_positions_flat.to(torch.int64))
        self.hidden_states[:num_query_total] = anchor_rep

        # Slot mapping per query position.
        req_indices = (
            torch.arange(batch_size, device=device, dtype=torch.int64)
            .repeat_interleave(Q)
        )
        bs = self.block_size
        assert bs > 0, "block_size must be initialized before set_inputs_first_pass"
        if query_positions_flat.dim() == 1:
            positions_for_slots = query_positions_flat
        else:
            # All M-RoPE dims share the same scalar position for text input.
            positions_for_slots = query_positions_flat[0]
        positions_i64 = positions_for_slots.to(torch.int64)
        block_nums = positions_i64 // bs
        block_offsets = positions_i64 % bs
        max_blocks = cad.block_table_tensor.shape[1]
        block_nums = torch.clamp(block_nums, max=max_blocks - 1)
        block_ids = cad.block_table_tensor[req_indices, block_nums].to(torch.int64)
        slot_mapping = block_ids * bs + block_offsets

        # Sample at positions 0..K-1 per request. The hidden at logical
        # position i predicts logical position i+1, so this yields K
        # speculative tokens for output positions 1..K (the K mask slots).
        # Flat index of the i-th query token of request r is r*Q + i.
        sample_block = (
            torch.arange(batch_size, device=device, dtype=torch.int32).unsqueeze(1) * Q
            + torch.arange(K, device=device, dtype=torch.int32).unsqueeze(0)
        )
        token_indices_to_sample = sample_block.reshape(batch_size * K)

        # New CAD: Q query tokens per req, seq_len = Q (no prior drafter
        # context — fresh K/Vs written to the Q slots only).
        new_query_start_loc = self.arange[: batch_size + 1] * Q  # [0,Q,2Q,...]
        new_query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * Q
        )
        new_seq_lens = torch.full(
            (batch_size,), Q, dtype=cad.seq_lens.dtype, device=device
        )
        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            query_start_loc_cpu=new_query_start_loc_cpu,
            seq_lens=new_seq_lens,
            num_reqs=batch_size,
            num_actual_tokens=num_query_total,
            max_query_len=Q,
            max_seq_len=Q,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=slot_mapping,
            causal=True,
        )
        return num_query_total, token_indices_to_sample, new_cad

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        model_kwargs = {
            "input_ids": self.input_ids[:num_input_tokens],
            "positions": self._get_positions(num_input_tokens),
            "inputs_embeds": None,
            "hidden_states": self.hidden_states[:num_input_tokens],
        }
        return model_kwargs, num_input_tokens

    @override
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Profiling / cudagraph warm-up pass. Skips the DFlash precompute
        step — Dmtp's drafter has no separate context K/V phase."""
        num_query_tokens = min(num_tokens, self.max_query_tokens)
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )
        )

        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping_dict,
        ):
            self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
                hidden_states=self.hidden_states[:num_input_tokens],
            )

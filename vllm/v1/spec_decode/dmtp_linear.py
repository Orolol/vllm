# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dmtp linear-chain proposer (P1).

Sibling of DFlashProposer — does NOT inherit DFlash's cross-attention
infrastructure. The Dmtp drafter is a single self-attention layer over
K mask query tokens, conditioned on the anchor target_hidden via fc;
there is no precomputed context K/V.

Each spec call:
  1. Set K mask_token_id query inputs per request.
  2. Replicate the anchor target_hidden (last accepted token's hidden) K
     times into self.hidden_states.
  3. Run the drafter once (parallel_drafting). Sample argmax at all K.

The tree-verify variant (P2) is a separate class that builds a per-request
tree from these K logits using FlexAttention.
"""

from __future__ import annotations

from typing import Any

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

logger = init_logger(__name__)


class DmtpLinearProposer(SpecDecodeBaseProposer):
    """Linear-chain Dmtp proposer: K masks → K argmax tokens, one forward."""

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
        # hidden replicated K times — there is no learned mask_hidden, so
        # we disable the buffer the base class would otherwise expect to
        # copy from the model.
        self.parallel_drafting_hidden_state_tensor = None
        # All K positions per request are sampled (no bonus token in
        # the drafter's view).
        self.max_query_tokens = self.max_batch_size * self.num_speculative_tokens

    @override
    def _warn_if_multimodal(self):
        # Dmtp targets Qwen3-family text-only models; suppress the warning.
        pass

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
        num_query_total = batch_size * K
        device = self.device

        # End-of-input index per request (last accepted token in target view).
        query_end_loc = cad.query_start_loc[1:] - 1
        if num_rejected_tokens_gpu is not None:
            query_end_loc = query_end_loc - num_rejected_tokens_gpu

        # Anchor: target's last hidden state per request.
        anchor_hidden = target_hidden_states[query_end_loc]  # [batch, H]
        H = anchor_hidden.shape[-1]
        anchor_rep = (
            anchor_hidden.unsqueeze(1).expand(-1, K, -1).reshape(num_query_total, H)
        )

        # Positions: bonus_pos = last_input_pos + 1; spec_pos[i] = bonus_pos+1+i.
        # target_positions may be 1D [num_tokens] or 2D [N, num_tokens] for
        # M-RoPE / xdrope. We index its LAST dim by query_end_loc to keep
        # any leading position axes intact.
        if target_positions.dim() == 1:
            last_input_pos = target_positions[query_end_loc]  # [batch]
        else:
            last_input_pos = target_positions[..., query_end_loc]  # [..., batch]
        spec_offsets = torch.arange(K, device=device, dtype=last_input_pos.dtype) + 1
        # Broadcast spec_offsets along the K dim. For 1D: [batch, 1] + [K] -> [B, K].
        # For 2D [N, batch]: [N, batch, 1] + [K] -> [N, batch, K].
        spec_positions = last_input_pos.unsqueeze(-1) + 1 + spec_offsets
        # Flatten to drafter's per-token positions: [batch*K] or [N, batch*K].
        if spec_positions.dim() == 2:
            spec_positions_flat = spec_positions.reshape(num_query_total)
        else:
            spec_positions_flat = spec_positions.reshape(
                spec_positions.shape[0], num_query_total
            )

        # Inputs.
        self.input_ids[:num_query_total] = self.parallel_drafting_token_id
        # _set_positions handles the [N, num_tokens] vs [num_tokens] cases
        # via uses_mrope / draft_uses_xdrope_dim flags on self.
        self._set_positions(num_query_total, spec_positions_flat.to(torch.int64))
        self.hidden_states[:num_query_total] = anchor_rep

        # Slot mapping: per-position lookup in the per-request block table.
        # Each spec position writes K/V to its own slot in the drafter's
        # cache; we never reuse these across calls (Dmtp is stateless).
        req_indices = (
            torch.arange(batch_size, device=device, dtype=torch.int64)
            .repeat_interleave(K)
        )
        bs = self.block_size
        assert bs > 0, "block_size must be initialized before set_inputs_first_pass"
        # Slot mapping is computed on flat [batch*K] positions (not M-RoPE'd).
        if spec_positions_flat.dim() == 1:
            positions_for_slots = spec_positions_flat
        else:
            # All M-RoPE dims share the same scalar position for text input.
            positions_for_slots = spec_positions_flat[0]
        positions_i64 = positions_for_slots.to(torch.int64)
        block_nums = positions_i64 // bs
        block_offsets = positions_i64 % bs
        max_blocks = cad.block_table_tensor.shape[1]
        block_nums = torch.clamp(block_nums, max=max_blocks - 1)
        block_ids = cad.block_table_tensor[req_indices, block_nums].to(torch.int64)
        slot_mapping = block_ids * bs + block_offsets

        # All K positions per request are sampled.
        token_indices_to_sample = self.arange[:num_query_total]

        # New CAD: K query tokens per req, seq_len = K (drafter has no prior
        # context — its K K/Vs live in the K freshly written slots only).
        new_query_start_loc = self.arange[: batch_size + 1] * K  # [0,K,2K,...]
        new_query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * K
        )
        new_seq_lens = torch.full(
            (batch_size,), K, dtype=cad.seq_lens.dtype, device=device
        )
        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            query_start_loc_cpu=new_query_start_loc_cpu,
            seq_lens=new_seq_lens,
            num_reqs=batch_size,
            num_actual_tokens=num_query_total,
            max_query_len=K,
            max_seq_len=K,
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
        # Identical to the SpecDecodeBaseProposer default but specialised
        # for our (no multimodal, hidden_states required) case.
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
        """Profiling / cudagraph warm-up pass.

        Mirrors DFlashProposer.dummy_run but skips the precompute step —
        Dmtp's drafter doesn't have a separate context K/V phase.
        """
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

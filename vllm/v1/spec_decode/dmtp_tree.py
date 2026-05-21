# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dmtp tree-verify proposer (P2', via TREE_ATTN backend).

Inherits :class:`DmtpLinearProposer` for the drafter forward (Q = 1 bonus
+ K masks query tokens, anchor target hidden replicated across Q). The
difference vs. the linear chain is the SAMPLING / VERIFY step:

* ``_greedy_sample`` builds a per-request tree from the K draft logits
  using :func:`build_dmtp_tree` (best-first beam with fixed top-K=2).
  Returns the tree nodes in flat node-order (batch * budget).
* The per-request tree visibility mask is pushed into the target's
  ``TreeAttentionMetadataBuilder`` (TREE_ATTN backend). The kernel adds
  the mask as a qq_bias to the attention scores so each tree node only
  attends to its ancestors.
* After the target's verify forward, the runner walks each tree via
  :func:`ddtree_verify` to find the longest accepted path.

Requirements:
- Target loaded with ``attention_config={"backend": "TREE_ATTN"}``
- The runner's existing DDTree position-override + ddtree_verify dispatch
  must be extended to recognise DmtpTreeProposer (handled in
  ``gpu_model_runner.py``).
"""

from __future__ import annotations

import os

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadataBuilder
from vllm.v1.spec_decode.dmtp import build_dmtp_tree
from vllm.v1.spec_decode.dmtp_linear import DmtpLinearProposer
from vllm.v1.spec_decode.dmtp_timing import time_region

logger = init_logger(__name__)

# Global tensors representing the current tree visibility mask and past KV lengths
# used to execute FlexAttention tree verification without dynamic compilation overhead.
_tree_visibility: torch.Tensor | None = None
_past_kv_lens: torch.Tensor | None = None


def tree_verify_logical_mask_mod(
    b: torch.Tensor,
    h: torch.Tensor,
    logical_q_idx: torch.Tensor,
    logical_kv_idx: torch.Tensor,
) -> torch.Tensor:
    global _tree_visibility, _past_kv_lens
    assert _tree_visibility is not None, "Tree visibility mask has not been populated"
    assert _past_kv_lens is not None, "Past KV cache length has not been populated"

    # Allow query to always attend to prompt KV cache history
    past_kv_len = _past_kv_lens[b]
    is_tree_attn = logical_kv_idx >= past_kv_len

    # Relative tree offsets: 0 to K (budget)
    q_tree_idx = logical_q_idx - past_kv_len
    kv_tree_idx = logical_kv_idx - past_kv_len

    # Clamp to avoid negative/out-of-bounds indexing inside the compiled mask function
    q_tree_idx_clamped = torch.clamp(q_tree_idx, min=0)
    kv_tree_idx_clamped = torch.clamp(kv_tree_idx, min=0)
    tree_allowed = _tree_visibility[b, q_tree_idx_clamped, kv_tree_idx_clamped]

    return (~is_tree_attn) | tree_allowed



class DmtpTreeProposer(DmtpLinearProposer):
    """Tree-verify Dmtp proposer using the TREE_ATTN backend."""

    DEFAULT_PER_POSITION_K = 2

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        assert vllm_config.speculative_config is not None
        method = vllm_config.speculative_config.method
        assert method == "dmtp_tree", (
            f"DmtpTreeProposer requires method=dmtp_tree, got {method!r}"
        )
        # DmtpLinearProposer's __init__ asserts method == "dmtp_linear" —
        # temporarily swap so super().__init__ accepts. Restore after.
        vllm_config.speculative_config.method = "dmtp_linear"
        try:
            super().__init__(vllm_config, device, runner)
        finally:
            vllm_config.speculative_config.method = method
        self.method = method

        self._runner = runner
        self._budget = self.num_speculative_tokens
        spec_cfg = vllm_config.speculative_config
        extra = getattr(spec_cfg, "extra_config", None) or {}
        self.per_position_k = int(
            extra.get("per_position_k", self.DEFAULT_PER_POSITION_K)
        )
        backend = vllm_config.attention_config.backend
        self.is_flex = (
            backend == "FLEX_ATTENTION" or getattr(backend, "name", None) == "FLEX_ATTENTION"
        )
        self._tree_visibility: torch.Tensor | None = None

        # Per-batch tree state, refreshed every _greedy_sample call.
        # Consumed by the runner's position-override (depths) and
        # ddtree_verify dispatch (child_maps).
        self._child_maps: list[list[dict[int, int]]] | None = None
        self._node_depths: list[torch.Tensor] | None = None

    @override
    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Build per-request tree from K logits, push qq_bias to TREE_ATTN.

        ``hidden_states`` arrives as ``[batch * K, hidden]`` — the parent
        propose() already indexed the K spec-prediction positions per
        request via ``token_indices_to_sample`` (set by our
        :class:`DmtpLinearProposer.set_inputs_first_pass`).
        """
        with time_region("tree/_greedy_sample"):
            K = self.num_speculative_tokens
            batch_size = hidden_states.shape[0] // K
            if batch_size == 0:
                # Defensive: propose called with empty drafter batch (e.g.
                # during warmup). Return an empty draft tensor.
                return torch.empty(0, dtype=torch.long, device=self.device)

            with time_region("tree/_greedy_sample.compute_logits"):
                logits = self.model.compute_logits(hidden_states)
            vocab_size = logits.shape[-1]
            logits_per_req = logits.float().view(batch_size, K, vocab_size)

            budget = self._budget
            target_size = budget + 1

            all_child_maps: list[list[dict[int, int]]] = []
            all_draft_tokens: list[torch.Tensor] = []
            all_visibility: list[torch.Tensor] = []
            all_node_depths: list[torch.Tensor] = []
            with time_region("tree/_greedy_sample.build_tree_loop"):
                for r in range(batch_size):
                    (
                        node_token_ids,
                        node_depths,
                        _,
                        _,
                        child_maps,
                        visibility,
                    ) = build_dmtp_tree(
                        logits_per_req[r],
                        budget=budget,
                        per_position_k=self.per_position_k,
                    )
                    all_child_maps.append(child_maps)

                    tokens = node_token_ids.to(self.device)
                    ba = tokens.shape[0]
                    if ba < budget:
                        tokens = torch.cat(
                            [tokens, tokens.new_zeros(budget - ba)]
                        )
                    all_draft_tokens.append(tokens)

                    depths = node_depths.to(self.device, dtype=torch.long)
                    if ba < budget:
                        pad = torch.arange(
                            ba + 1, ba + 1 + (budget - ba),
                            dtype=torch.long, device=self.device,
                        )
                        depths = torch.cat([depths, pad])
                    all_node_depths.append(depths)

                    vis_size = visibility.shape[0]
                    if vis_size < target_size:
                        padded = torch.zeros(
                            target_size, target_size, dtype=torch.bool
                        )
                        padded[:vis_size, :vis_size] = visibility
                        visibility = padded
                    all_visibility.append(visibility)

            self._child_maps = all_child_maps
            self._node_depths = all_node_depths

            stacked = torch.stack(all_visibility, dim=0)  # [batch, N+1, N+1]

            if self.is_flex:
                self._tree_visibility = stacked.to(self.device)
            else:
                # Push the tree mask (additive bias form: 0 / -inf) to all
                # TREE_ATTN metadata builders on the target model.
                with time_region("tree/_greedy_sample.bias_build_push"):
                    _mode = os.environ.get("DMTP_BIAS_MODE", "real")
                    if _mode == "none":
                        tree_attn_bias = torch.empty(
                            0, dtype=torch.float32, device=self.device
                        )
                    elif _mode == "zero":
                        tree_attn_bias = torch.zeros(
                            stacked.shape,
                            dtype=torch.float32,
                            device=self.device,
                        )
                    else:
                        tree_attn_bias = torch.where(
                            stacked.to(self.device),
                            torch.zeros(1, dtype=torch.float32, device=self.device),
                            torch.full(
                                (1,), float("-inf"),
                                dtype=torch.float32, device=self.device,
                            ),
                        )
                    self._update_target_tree_attn_bias(tree_attn_bias)

            draft = torch.stack(all_draft_tokens, dim=0)  # [batch, budget]
            return draft.reshape(-1).to(torch.long)

    def _update_target_tree_attn_bias(self, tree_attn_bias: torch.Tensor) -> None:
        """Push a new tree_attn_bias to all TreeAttentionMetadataBuilders.

        Mirrors :meth:`DDTreeProposer._update_target_tree_attn_bias`.
        Called once per propose step so the target's next verify forward
        uses the freshly-built tree topology.
        """
        if self._runner is None:
            return
        # In "none" ablation mode the bias is an empty 1-D tensor; keep
        # the threshold derived from the budget so the spec batch is
        # still classified as decode (just without a qq_bias).
        if tree_attn_bias.ndim >= 2:
            threshold = tree_attn_bias.shape[-2]
        else:
            threshold = self._budget + 1
        found = False
        for attn_groups in self._runner.attn_groups:
            for attn_group in attn_groups:
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, TreeAttentionMetadataBuilder):
                    builder.tree_attn_bias = tree_attn_bias
                    builder.reorder_batch_threshold = threshold - 1
                    builder._tree_decode_threshold = threshold
                    found = True
        assert found, (
            "DmtpTreeProposer requires the target to use the TREE_ATTN "
            "attention backend. Set attention_config.backend = 'TREE_ATTN'."
        )

    def enable_flex_tree_mask(self, model: torch.nn.Module) -> None:
        """Temporarily override the attention mask on FlexAttention layers."""
        for m in model.modules():
            if hasattr(m, "impl") and m.impl.__class__.__name__ == "FlexAttentionImpl":
                m.logical_mask_mod = tree_verify_logical_mask_mod

    def disable_flex_tree_mask(self, model: torch.nn.Module) -> None:
        """Clear the custom attention mask override from FlexAttention layers."""
        for m in model.modules():
            if hasattr(m, "impl") and m.impl.__class__.__name__ == "FlexAttentionImpl":
                m.logical_mask_mod = None

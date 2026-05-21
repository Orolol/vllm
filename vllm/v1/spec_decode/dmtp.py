"""Dmtp proposer: block-diffusion drafter + fixed-K beam tree-verify.

Almost identical to DDTree's proposer (vllm#42910); the only material
difference is that the per-position top-K is fixed at a small value
(default K=2) instead of `min(budget, vocab)`. This produces a tree
that explores fewer per-position alternatives but reaches deeper,
which we have empirically found to be a sweet spot for well-trained
block-diffusion drafters — see the Dmtp project for the ablations.

The drafter weight format is identical to DFlash's; the only routing
difference is that we register a separate `method: "dmtp"` so the
runtime picks our tree builder.

Reference: github.com/Orolol/dmtp
"""

from __future__ import annotations

import heapq

import numpy as np
import torch

from vllm.v1.spec_decode.ddtree import (
    DDTreeProposer,
    build_ddtree_tree,  # re-exported for compatibility
    ddtree_verify,
)


__all__ = ["DmtpProposer", "build_dmtp_tree"]


def build_dmtp_tree(
    draft_logits: torch.Tensor,
    budget: int,
    per_position_k: int = 2,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[dict[int, int]],
    torch.Tensor,
]:
    """Best-first beam-pruned tree with a *fixed* per-position top-K.

    Same output shape as :func:`build_ddtree_tree`, same accumulated-
    log-prob scoring; the only change is the upper bound on per-position
    rank exploration.

    Args:
        draft_logits: ``[depth, vocab_size]`` raw logits.
        budget: max number of non-root tree nodes to expand.
        per_position_k: max rank explored at each depth (DDTree's heap
            uses ``min(budget, vocab)`` which can be much larger).

    Returns:
        Same 6-tuple as :func:`build_ddtree_tree`. Compatible with
        :func:`ddtree_verify` and the TREE_ATTN backend.
    """
    if budget <= 0 or draft_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [{}],
            visibility,
        )

    depth_limit = int(draft_logits.shape[0])
    # The one material difference vs build_ddtree_tree:
    # cap topk at per_position_k instead of budget.
    topk = min(per_position_k, draft_logits.shape[-1])

    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_np = (
        (top_logits - log_z).to(device="cpu", dtype=torch.float32).numpy()
    )
    top_token_ids_np = top_token_ids.to(device="cpu", dtype=torch.long).numpy()

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    node_ranks_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [{}]
    node_count = 0

    first_logw = float(top_log_probs_np[0, 0])
    heap: list[tuple] = [(-first_logw, (0,), 0, 1, 0, first_logw)]

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)
        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1

        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        node_ranks_np[node_count] = rank
        parents_np[current_index] = parent_index
        child_maps.append({})
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sibling_logw = (
                logw
                - float(top_log_probs_np[depth - 1, rank])
                + float(top_log_probs_np[depth - 1, rank + 1])
            )
            heapq.heappush(
                heap,
                (
                    -sibling_logw,
                    ranks[:-1] + (rank + 1,),
                    parent_index,
                    depth,
                    rank + 1,
                    sibling_logw,
                ),
            )

        if depth < depth_limit:
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(
                heap,
                (
                    -child_logw,
                    ranks + (0,),
                    current_index,
                    depth + 1,
                    0,
                    child_logw,
                ),
            )

    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for idx in range(1, current_length):
        p = int(parents_np[idx])
        visibility_np[idx, :idx] = visibility_np[p, :idx]
        visibility_np[idx, idx] = True

    return (
        torch.from_numpy(node_token_ids_np[:node_count]),
        torch.from_numpy(node_depths_np[:node_count]),
        torch.from_numpy(node_ranks_np[:node_count]),
        parents_np[:current_length].tolist(),
        child_maps,
        torch.from_numpy(visibility_np),
    )


class DmtpProposer(DDTreeProposer):
    """Dmtp: DDTree topology with K=2 beam (drafter-quality sweet spot).

    Inherits everything from DDTreeProposer; overrides only the tree
    builder so we can configure a fixed per-position top-K instead of
    DDTree's unbounded heap.
    """

    DEFAULT_PER_POSITION_K = 2

    def __init__(self, vllm_config, device, runner):
        super().__init__(vllm_config, device, runner)
        spec_cfg = vllm_config.speculative_config
        extra = getattr(spec_cfg, "extra_config", None) or {}
        self.per_position_k = int(
            extra.get("per_position_k", self.DEFAULT_PER_POSITION_K)
        )

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Like DDTreeProposer._greedy_sample but uses build_dmtp_tree."""
        depth = self.num_speculative_tokens
        batch_size = hidden_states.shape[0] // depth

        logits = self.model.compute_logits(hidden_states)
        vocab_size = logits.shape[-1]
        logits_per_req = logits.float().view(batch_size, depth, vocab_size)

        all_child_maps: list[list[dict[int, int]]] = []
        all_draft_tokens: list[torch.Tensor] = []
        all_visibility: list[torch.Tensor] = []
        all_node_depths: list[torch.Tensor] = []
        target_size = self._budget + 1

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
                budget=self._budget,
                per_position_k=self.per_position_k,
            )
            all_child_maps.append(child_maps)

            tokens = node_token_ids.to(self.device)
            budget_actual = tokens.shape[0]
            if budget_actual < self._budget:
                tokens = torch.cat(
                    [tokens, tokens.new_zeros(self._budget - budget_actual)]
                )
            all_draft_tokens.append(tokens)

            depths = node_depths.to(self.device, dtype=torch.long)
            if budget_actual < self._budget:
                pad = torch.arange(
                    budget_actual + 1,
                    budget_actual + 1 + (self._budget - budget_actual),
                    dtype=torch.long,
                    device=self.device,
                )
                depths = torch.cat([depths, pad])
            all_node_depths.append(depths)

            vis_size = visibility.shape[0]
            if vis_size < target_size:
                padded = torch.zeros(target_size, target_size, dtype=torch.bool)
                padded[:vis_size, :vis_size] = visibility
                visibility = padded
            all_visibility.append(visibility)

        self._child_maps = all_child_maps
        self._node_depths = all_node_depths

        stacked = torch.stack(all_visibility, dim=0)
        tree_attn_bias = torch.where(
            stacked.to(self.device),
            torch.zeros(1, dtype=torch.float32, device=self.device),
            torch.full((1,), float("-inf"), dtype=torch.float32, device=self.device),
        )
        self._update_target_tree_attn_bias(tree_attn_bias)

        draft = torch.stack(all_draft_tokens, dim=0)
        return draft.reshape(-1).to(torch.long)

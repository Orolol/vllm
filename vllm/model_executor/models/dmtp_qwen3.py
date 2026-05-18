# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dmtp drafter: block-diffusion MTP-shaped draft model.

Architecturally mirrors :class:`Qwen3NextMultiTokenPredictor` but:

* Uses a learned ``mask_embedding`` substituted at positions the proposer
  marks with ``mask_token_id``.
* Conditioning on the target hidden happens via
  ``fc(concat(norm(inputs_embeds), norm(hidden_states)))``.
* Self-attention is causal over the K mask query tokens; no
  cross-attention to prior context.

The single transformer block uses Qwen3-Next-style gated-q attention
(``attn_output_gate=True``) to match our trained checkpoint shape.

Companion proposer: :class:`vllm.v1.spec_decode.dmtp_linear.DmtpLinearProposer`.
Checkpoint converter: ``Orolol/dmtp/dmtp_vllm/converter.py``.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextRMSNorm,
)

from .utils import AutoWeightsLoader, maybe_prefix

logger = init_logger(__name__)


class DmtpDecoderLayer(nn.Module):
    """Single Dmtp transformer block.

    Mirrors the ``full_attention`` path of :class:`Qwen3NextDecoderLayer`
    but takes ``config`` explicitly so it can live inside the speculative
    draft model (whose hf_config differs from the target's).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3NextAttention(
            config,
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            quant_config=None,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3NextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=None,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_out = torch.empty_like(hidden_states)
        self.self_attn(
            positions=positions,
            output=attn_out,
            hidden_states=hidden_states,
        )
        hidden_states = attn_out

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class DmtpQwen3DraftModel(nn.Module):
    """Single-layer Dmtp drafter.

    Forward signature matches what
    :class:`vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer`
    passes when ``pass_hidden_states_to_model=True``::

        model(input_ids, positions, inputs_embeds, hidden_states)

    where ``hidden_states`` is the anchor target hidden replicated across
    the K mask query positions.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        H = self.config.hidden_size

        # Mask embedding: shared [H] or per-position [block_size, H].
        block_size = getattr(self.config, "block_size", None)
        per_position = bool(
            getattr(self.config, "per_position_mask_embedding", False)
        )
        if per_position:
            assert block_size is not None, (
                "per_position_mask_embedding requires block_size in config"
            )
            self.mask_embedding = nn.Parameter(torch.zeros(block_size, H))
        else:
            self.mask_embedding = nn.Parameter(torch.zeros(H))
        self.per_position_mask_embedding = per_position
        self.block_size = block_size
        self.mask_token_id = getattr(self.config, "mask_token_id", None)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            H,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.pre_fc_norm_embedding = Qwen3NextRMSNorm(H, eps=self.config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3NextRMSNorm(H, eps=self.config.rms_norm_eps)

        self.fc = ReplicatedLinear(
            input_size=2 * H,
            output_size=H,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            prefix=maybe_prefix(prefix, "fc"),
            return_bias=False,
        )

        self.layers = nn.ModuleList(
            [
                DmtpDecoderLayer(
                    vllm_config,
                    config=self.config,
                    prefix=maybe_prefix(prefix, "layers.0"),
                )
            ]
        )
        self.norm = Qwen3NextRMSNorm(H, eps=self.config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids)
        if self.mask_token_id is None:
            return embeds
        mask = input_ids == self.mask_token_id
        if not mask.any():
            return embeds
        embeds = embeds.clone()
        if self.per_position_mask_embedding:
            # Per-position requires the proposer to expose ``mask_positions``
            # on the model so we know which of the K block slots each token
            # occupies. P1 linear-chain uses the shared single-vector path.
            mask_positions = getattr(self, "mask_positions", None)
            assert mask_positions is not None, (
                "per_position_mask_embedding requires model.mask_positions "
                "to be set by the proposer before forward"
            )
            embeds[mask] = self.mask_embedding[mask_positions[mask]]
        else:
            embeds[mask] = self.mask_embedding.to(embeds.dtype)
        return embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        assert hidden_states is not None, (
            "DmtpQwen3DraftModel requires hidden_states (target hidden) — "
            "the proposer must run with pass_hidden_states_to_model=True"
        )
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1], (
            f"hidden_states {tuple(hidden_states.shape)} mismatched with "
            f"inputs_embeds {tuple(inputs_embeds.shape)}"
        )

        normed_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        normed_hidden = self.pre_fc_norm_hidden(hidden_states)
        combined = torch.cat([normed_embeds, normed_hidden], dim=-1)
        hs = self.fc(combined)

        residual: torch.Tensor | None = None
        for layer in self.layers:
            hs, residual = layer(
                hidden_states=hs,
                residual=residual,
                positions=positions,
            )
        hs, _ = self.norm(hs, residual)
        return hs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                fused_name = name.replace(weight_name, param_name)
                if fused_name not in params_dict:
                    continue
                param = params_dict[fused_name]
                param.weight_loader(param, weight, shard_id)
                loaded.add(fused_name)
                matched = True
                break
            if matched:
                continue
            if name not in params_dict:
                logger.warning("DmtpQwen3DraftModel: skipping unknown weight %s", name)
                continue
            param = params_dict[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, weight)
            loaded.add(name)
        return loaded


class DmtpQwen3ForCausalLM(Qwen3ForCausalLM):
    """Outer module exposed to the vLLM model registry.

    Wraps :class:`DmtpQwen3DraftModel` with the lm_head + logits processor.
    The lm_head is shared with the target model via the usual vLLM
    weight-sharing mechanism (see SpecDecodeBaseProposer._load_model).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config

        self.model = DmtpQwen3DraftModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds, hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # Our checkpoint is flat. Prefix non-lm_head keys with "model." to
        # land inside DmtpQwen3DraftModel; AutoWeightsLoader then dispatches.
        model_weights: dict[str, torch.Tensor] = {}
        for name, weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "lm_head" in name:
                model_weights[name] = weight
                continue
            if not name.startswith("model."):
                name = "model." + name
            model_weights[name] = weight

        skip_substrs = []
        if not any("lm_head" in k for k in model_weights):
            skip_substrs.append("lm_head")
        loader = AutoWeightsLoader(self, skip_prefixes=None, skip_substrs=skip_substrs)
        loader.load_weights(model_weights.items())

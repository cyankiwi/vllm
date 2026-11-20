# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multimodal inference for fixed-count SenseNova Loop checkpoints with optional mHC."""

from collections.abc import Iterable
from copy import copy

import torch
from torch import nn
from torch.nn import functional as F

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.sensenova_loop_mhc import MHCRouter
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from .qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5ProcessingInfo,
    Qwen3_5RMSNorm,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from .utils import (
    AutoWeightsLoader,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)


def _expand_layer_name(
    name: str, destinations: dict[int, list[int]], prefix: str = "language_model.model"
) -> Iterable[str]:
    """Map a physical layer name to the names of all execution layers using it."""
    layers = f"{prefix}.layers."
    if not name.startswith(layers):
        yield name
        return
    index, suffix = name.removeprefix(layers).split(".", 1)
    for execution in destinations[int(index)]:
        yield f"{layers}{execution}.{suffix}"


def _loop_destinations(indices: Iterable[int]) -> dict[int, list[int]]:
    destinations: dict[int, list[int]] = {}
    for execution, physical in enumerate(indices):
        destinations.setdefault(physical, []).append(execution)
    return destinations


class SenseNovaLoopAWQEmbeddingMethod(QuantizeMethodBase):
    """Decode the row-major INT4 embedding for the requested tokens."""

    def __init__(self, group_size: int) -> None:
        self.group_size = group_size

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if input_size % (8 * self.group_size):
            raise ValueError("SenseNova Loop AWQ embedding dimensions are misaligned")
        num_rows = sum(output_partition_sizes)
        num_groups = input_size // self.group_size
        for name, width, dtype in (
            ("qweight", input_size // 8, torch.int32),
            ("qzeros", num_groups // 8, torch.int32),
            ("scales", num_groups, torch.float32),
        ):
            param = nn.Parameter(
                torch.empty(num_rows, width, dtype=dtype), requires_grad=False
            )
            set_weight_attrs(param, {"output_dim": 0, **extra_weight_attrs})
            layer.register_parameter(name, param)

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        shifts = torch.arange(0, 32, 4, device=input_.device, dtype=torch.int32)

        def unpack(packed: torch.Tensor, width: int) -> torch.Tensor:
            words = F.embedding(input_, packed)
            return ((words.unsqueeze(-1) >> shifts) & 15).reshape(*input_.shape, width)

        groups = layer.embedding_dim // self.group_size
        weights = unpack(layer.qweight, layer.embedding_dim).unflatten(
            -1, (groups, self.group_size)
        )
        zeros = unpack(layer.qzeros, groups)
        scales = F.embedding(input_, layer.scales)
        return (
            ((weights - zeros.unsqueeze(-1)) * scales.unsqueeze(-1))
            .flatten(-2)
            .to(layer.params_dtype)
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("SenseNova Loop AWQ embeddings are lookup-only")


class SenseNovaLoopDecoderLayer(Qwen3_5DecoderLayer):
    def __init__(self, vllm_config: VllmConfig, layer_type: str, prefix: str) -> None:
        super().__init__(vllm_config, layer_type, prefix)
        config = vllm_config.model_config.hf_text_config
        self.physical_index = config.loop_physical_indices[self.layer_idx]
        self.mhc = config.mhc
        self.mhc_active = self.mhc is not None and (
            self.mhc["start"] <= self.physical_index < self.mhc["end"]
        )
        if self.mhc_active:
            self.mhc_attn = MHCRouter(config.hidden_size, self.mhc)
            self.mhc_ffn = MHCRouter(config.hidden_size, self.mhc)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.mhc_active:
            return super().forward(hidden_states, residual, positions, **kwargs)
        if residual is not None:
            hidden_states = hidden_states + residual
        streams = self.mhc["streams"]
        if self.physical_index == self.mhc["start"]:
            hidden_states = hidden_states.unsqueeze(-2).expand(-1, streams, -1)
        else:
            hidden_states = hidden_states.unflatten(-1, (streams, -1))
        branch, post, mix = self.mhc_attn(hidden_states)
        branch = self.input_layernorm(branch.contiguous())
        if self.layer_type == "linear_attention":
            branch = self.linear_attn(hidden_states=branch)
        else:
            branch = self.self_attn(hidden_states=branch, positions=positions)
        hidden_states = MHCRouter.merge(hidden_states, branch, post, mix)
        branch, post, mix = self.mhc_ffn(hidden_states)
        branch = self.mlp(self.post_attention_layernorm(branch.contiguous()))
        hidden_states = MHCRouter.merge(hidden_states, branch, post, mix)
        if self.physical_index == self.mhc["end"] - 1:
            hidden_states = hidden_states.mean(-2)
        else:
            hidden_states = hidden_states.flatten(-2)
        return hidden_states, None


class SenseNovaLoopModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config
        embedding_method = None
        if isinstance(self.quant_config, AutoAWQConfig):
            embedding = self.quant_config.full_config.get("embedding")
            if embedding is not None:
                group_size = embedding.get("group_size")
                if (
                    embedding.get("layout") != "row_major"
                    or embedding.get("bits") != 4
                    or embedding.get("zero_point") is not True
                    or type(group_size) is not int
                    or group_size <= 0
                    or self.config.tie_word_embeddings
                ):
                    raise ValueError("Unsupported SenseNova Loop AWQ embedding config")
                embedding_method = SenseNovaLoopAWQEmbeddingMethod(group_size)
        self.vocab_size = self.config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.config.hidden_size,
            quant_method=embedding_method,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleList(
            SenseNovaLoopDecoderLayer(vllm_config, layer_type, f"{prefix}.layers.{i}")
            for i, layer_type in enumerate(self.config.layer_types)
        )
        self.norm = Qwen3_5RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        )
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, positions)
        if residual is None:
            return self.norm(hidden_states)
        return self.norm(hidden_states, residual)[0]

    def share_loop_parameters(self) -> None:
        first: dict[int, nn.Module] = {}
        for physical, layer in zip(self.config.loop_physical_indices, self.layers):
            if physical not in first:
                first[physical] = layer
                continue
            for name, param in first[physical].named_parameters():
                parent, _, leaf = name.rpartition(".")
                setattr(layer.get_submodule(parent), leaf, param)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=Qwen3_5Model.hf_to_vllm_mapper
        )


class SenseNovaLoopForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_text_config
        self.model = SenseNovaLoopModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class SenseNovaLoopForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    language_model_cls = SenseNovaLoopForCausalLM

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config
        spec_config = vllm_config.speculative_config
        if (
            vllm_config.parallel_config.pipeline_parallel_size != 1
            or (spec_config is not None and spec_config.method != "mtp")
            or vllm_config.lora_config is not None
            or getattr(config, "layer_scale", False)
        ):
            raise ValueError(
                "SenseNova Loop does not support pipeline parallelism, LoRA, "
                "layer_scale, or speculative methods other than MTP"
            )
        self.loop_destinations = _loop_destinations(config.loop_physical_indices)
        quant_config = vllm_config.quant_config
        if isinstance(quant_config, CompressedTensorsConfig):
            # The loader has mapped the physical checkpoint paths to VL paths.
            quant_config = copy(quant_config)
            model_prefix = maybe_prefix(prefix, "language_model.model")
            quant_config.target_scheme_map = {
                execution_name: scheme
                for name, scheme in quant_config.target_scheme_map.items()
                for execution_name in _expand_layer_name(
                    name, self.loop_destinations, model_prefix
                )
            }
            quant_config.ignore = [
                execution_name
                for name in quant_config.ignore
                for execution_name in _expand_layer_name(
                    name, self.loop_destinations, model_prefix
                )
            ]
            vllm_config = copy(vllm_config)
            vllm_config.quant_config = quant_config
        elif quant_config is not None and not isinstance(quant_config, AutoAWQConfig):
            raise ValueError(
                "SenseNova Loop supports only unquantized, AWQ and CT weights"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def expanded_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in self.hf_to_vllm_mapper.apply(weights):
                if name == "language_model.model.loop_gate.logits":
                    continue
                for execution_name in _expand_layer_name(name, self.loop_destinations):
                    yield execution_name, weight

        return AutoWeightsLoader(self).load_weights(expanded_weights())

    def process_weights_after_loading(self) -> None:
        # Parameters share storage; attention modules and their caches stay distinct.
        if isinstance(self.language_model, SenseNovaLoopForCausalLM):
            self.language_model.model.share_loop_parameters()

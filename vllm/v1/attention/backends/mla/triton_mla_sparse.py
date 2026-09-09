# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton BF16 sparse MLA fallback for SM12x GPUs."""

from typing import TYPE_CHECKING, Any, ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    FlashInferMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.ops.triton_mla_sparse import triton_mla_sparse_attention

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


class TritonMLASparseMetadataBuilder(FlashInferMLASparseMetadataBuilder):
    """FlashInfer sparse metadata with an explicit CUDA-graph claim.

    Reusing the FlashInfer builder directly would silently adopt its capture
    support. The Triton path qualifies for uniform-batch capture on its own
    terms: ``TritonMLASparseImpl.forward_mqa`` performs no host synchronization,
    and every allocation is shaped from input shapes and static config rather
    than tensor data, so a captured graph stays valid on replay.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class TritonMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type[TritonMLASparseMetadataBuilder]:
        return TritonMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return TritonMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if not use_sparse:
            return "TRITON_MLA_SPARSE requires sparse MLA"
        from vllm.config import get_current_vllm_config

        model_config = get_current_vllm_config().model_config
        if model_config is not None:
            hf_config = model_config.hf_text_config
            if hf_config.kv_lora_rank != 512:
                return "TRITON_MLA_SPARSE requires kv_lora_rank=512"
            if hf_config.qk_rope_head_dim not in (0, 64):
                return "TRITON_MLA_SPARSE requires qk_rope_head_dim in (0, 64)"
            if getattr(hf_config, "index_topk", None) != 2048:
                return "TRITON_MLA_SPARSE requires index_topk=2048"
        return None


class TritonMLASparseImpl(SparseMLACommonImpl[FlashInferMLASparseMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args: Any,
    ) -> None:
        if any((alibi_slopes, sliding_window, logits_soft_cap)):
            raise NotImplementedError(
                "TritonMLASparseImpl does not support alibi, sliding window, "
                "or logits soft cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "TritonMLASparseImpl only supports decoder attention"
            )
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        if self.topk_indices_buffer is None:
            raise ValueError("TritonMLASparseImpl requires a top-k indices buffer")
        self.supports_quant_query_input = False

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError("Triton sparse MLA requires a BF16 KV cache")
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_tokens = q.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_tokens]
        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        topk_indices = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_tokens],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        ).view(num_tokens, 1, -1)
        kv_rows = kv_rows.view(-1, 1, kv_rows.shape[-1])
        out = triton_mla_sparse_attention(q, kv_rows, topk_indices, self.scale)
        return out[:, : self.num_heads], None

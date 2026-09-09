# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.attention.backends.mla.triton_mla_sparse as backend_module
from vllm.platforms import current_platform
from vllm.platforms.cuda import _get_backend_priorities
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    TritonMLASparseBackend,
    TritonMLASparseImpl,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("rope_dim", [0, 64])
def test_forward_mqa_concatenates_nope_and_rope_without_native_concat(
    monkeypatch, rope_dim: int
) -> None:
    recorded: dict[str, object] = {}

    def fake_convert(*args, **kwargs):
        recorded["convert_width"] = kwargs["NUM_TOPK_TOKENS"]
        return torch.zeros((2, 2176), dtype=torch.int32)

    def fake_attention(q, kv, indices, scale):
        recorded.update(q=q, kv=kv, indices=indices, scale=scale)
        return torch.zeros((2, 4, 512), dtype=torch.bfloat16)

    monkeypatch.setattr(
        backend_module, "triton_convert_req_index_to_global_index", fake_convert
    )
    monkeypatch.setattr(backend_module, "triton_mla_sparse_attention", fake_attention)

    impl = object.__new__(TritonMLASparseImpl)
    impl.kv_cache_dtype = "auto"
    impl.topk_indices_buffer = torch.zeros((2, 2176), dtype=torch.int32)
    impl.scale = 0.125
    impl.num_heads = 4
    q_nope = torch.randn(2, 4, 512, dtype=torch.bfloat16)
    q_rope = torch.randn(2, 4, rope_dim, dtype=torch.bfloat16)
    kv_cache = torch.randn(1, 4, 512 + rope_dim, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        block_size=4,
        req_id_per_token=torch.zeros(2, dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
    )

    out, lse = TritonMLASparseImpl.forward_mqa(
        impl, (q_nope, q_rope), kv_cache, metadata, layer=SimpleNamespace()
    )

    assert out.shape == (2, 4, 512)
    assert lse is None
    assert recorded["convert_width"] == 2176
    assert recorded["q"].shape == (2, 4, 512 + rope_dim)
    assert recorded["indices"].shape == (2, 1, 2176)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_forward_mqa_maps_expanded_indices_into_padded_cache() -> None:
    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 12:
        pytest.skip("TRITON_MLA_SPARSE is an SM12x backend")

    device = "cuda"
    block_size = 4
    block_stride_rows = 6
    num_blocks = 5
    head_dim = 512
    topk_width = 2176
    storage = torch.full(
        (num_blocks * block_stride_rows, head_dim),
        1000,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = storage.as_strided(
        (num_blocks, block_size, head_dim),
        (block_stride_rows * head_dim, head_dim, 1),
    )
    for block in range(num_blocks):
        for offset in range(block_size):
            kv_cache[block, offset].fill_(block * 10 + offset)

    topk_indices = torch.full((2, topk_width), -1, dtype=torch.int32, device=device)
    topk_indices[0, 0] = 0
    topk_indices[0, 2112] = 5
    topk_indices[1, 0] = 2
    topk_indices[1, 2112] = 7

    impl = object.__new__(TritonMLASparseImpl)
    impl.kv_cache_dtype = "auto"
    impl.topk_indices_buffer = topk_indices
    impl.scale = head_dim**-0.5
    impl.num_heads = 1
    metadata = SimpleNamespace(
        block_size=block_size,
        req_id_per_token=torch.tensor([0, 1], dtype=torch.int32, device=device),
        block_table=torch.tensor([[3, 1], [4, 0]], dtype=torch.int32, device=device),
    )
    q_nope = torch.zeros((2, 1, head_dim), dtype=torch.bfloat16, device=device)
    q_rope = torch.empty((2, 1, 0), dtype=torch.bfloat16, device=device)

    out, lse = TritonMLASparseImpl.forward_mqa(
        impl, (q_nope, q_rope), kv_cache, metadata, layer=SimpleNamespace()
    )

    expected = torch.stack(
        (
            (kv_cache[3, 0] + kv_cache[1, 1]) / 2,
            (kv_cache[4, 2] + kv_cache[0, 3]) / 2,
        )
    ).unsqueeze(1)
    assert lse is None
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_backend_capability_and_model_contract_gates(monkeypatch) -> None:
    hf_config = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=0, index_topk=2048)
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(hf_text_config=hf_config)),
    )
    kwargs = dict(
        head_size=512,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=64,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        use_mm_prefix=False,
        device_capability=DeviceCapability(12, 1),
    )

    assert TritonMLASparseBackend.supports_compute_capability(DeviceCapability(12, 1))
    assert not TritonMLASparseBackend.supports_compute_capability(
        DeviceCapability(9, 0)
    )
    assert TritonMLASparseBackend.supports_head_size(512)
    assert not TritonMLASparseBackend.supports_head_size(128)
    assert not TritonMLASparseBackend.supports_kv_cache_dtype("fp8_ds_mla")
    assert not TritonMLASparseBackend.supports_device_cpu_query_lens_mismatch()
    assert TritonMLASparseBackend.supports_combination(**kwargs) is None
    assert "requires sparse MLA" in (
        TritonMLASparseBackend.supports_combination(**(kwargs | {"use_sparse": False}))
        or ""
    )
    invalid_reasons = TritonMLASparseBackend.validate_configuration(
        **kwargs,
        use_per_head_quant_scales=False,
        attn_type=AttentionType.DECODER,
        use_adaptive_verification=True,
    )
    assert (
        "device-cpu query lens mismatch not supported, "
        "this is needed for adaptive verification"
    ) in invalid_reasons

    hf_config.index_topk = 1024
    assert "index_topk=2048" in (
        TritonMLASparseBackend.supports_combination(**kwargs) or ""
    )


def test_sm12_priority_uses_triton_as_bf16_sparse_fallback() -> None:
    priorities = _get_backend_priorities(
        use_mla=True,
        device_capability=DeviceCapability(12, 1),
        num_heads=64,
        kv_cache_dtype="bfloat16",
        head_size=512,
    )

    assert priorities.index(AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120) < (
        priorities.index(AttentionBackendEnum.TRITON_MLA_SPARSE)
    )

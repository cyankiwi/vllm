# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for wna16 MoE support with small (group_size=32) quantization groups.

These cover the two things that break when a compressed-tensors ``pack-quantized``
checkpoint uses group_size=32 instead of the more common 128:

1. ``_ensure_block_size_k_divisible`` must honour the moe_wna16 CUDA kernel
   constraint ``BLOCK_SIZE_K // group_size in [1, 2, 4, 8]``.
2. The int32 packing used by compressed-tensors, viewed as uint8, must already
   match the ``[N, K // 2]`` nibble layout the ``use_int4_w4a16`` kernel expects
   (this is what lets K2-Horizon MoVA value experts stay packed).
"""
import pytest
import torch

from vllm.model_executor.layers.fused_moe.fused_moe import (
    _ensure_block_size_k_divisible,
)


@pytest.mark.parametrize("size_k", [2560, 4096, 8192])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("block_size_k", [128, 256, 512, 1024])
def test_block_size_k_respects_cuda_kernel_ratio(size_k, group_size, block_size_k):
    out = _ensure_block_size_k_divisible(size_k, block_size_k, group_size)
    assert size_k % out == 0, "BLOCK_SIZE_K must divide size_k"
    assert out % group_size == 0, "BLOCK_SIZE_K must be a multiple of group_size"
    # moe_wna16.cu aborts unless this ratio is one of [1, 2, 4, 8].
    assert out // group_size in (1, 2, 4, 8), (
        f"BLOCK_SIZE_K // group_size = {out // group_size} is rejected by "
        "the moe_wna16 CUDA kernel"
    )


def test_block_size_k_regression_group32():
    """group_size=32 with size_k=2560 previously produced BLOCK_SIZE_K=512."""
    assert _ensure_block_size_k_divisible(2560, 512, 32) == 256


def test_compressed_tensors_int32_view_matches_kernel_nibble_layout():
    pytest.importorskip("compressed_tensors")
    from compressed_tensors.compressors.pack_quantized.helpers import (
        pack_to_int32,
        unpack_from_int32,
    )

    torch.manual_seed(0)
    N, K = 16, 64
    q = torch.randint(-8, 8, (N, K), dtype=torch.int8)
    packed = pack_to_int32(q, 4)  # [N, K // 8] int32, packed along K
    assert packed.shape == (N, K // 8)

    # The kernel reads uint8 [N, K // 2]: low nibble = even k, high = odd k.
    as_u8 = packed.contiguous().view(torch.uint8).view(N, K // 2)
    lo = (as_u8 & 0xF).to(torch.int16)
    hi = (as_u8 >> 4).to(torch.int16)
    decoded = torch.stack([lo, hi], dim=-1).reshape(N, K) - 8  # undo +8 bias

    torch.testing.assert_close(decoded, q.to(torch.int16))
    torch.testing.assert_close(
        unpack_from_int32(packed, 4, torch.Size([N, K])).to(torch.int16),
        q.to(torch.int16),
    )


def test_zero_point_repack_to_kernel_layout():
    """Zero points are packed along N by compressed-tensors, along N/2 by the kernel."""
    pytest.importorskip("compressed_tensors")
    from compressed_tensors.compressors.pack_quantized.helpers import (
        pack_to_int32,
        unpack_from_int32,
    )

    torch.manual_seed(0)
    N, num_groups = 16, 4
    zp = torch.randint(-8, 8, (N, num_groups), dtype=torch.int8)
    packed_zp = pack_to_int32(zp, 4, packed_dim=0)  # [N // 8, num_groups]
    assert packed_zp.shape == (N // 8, num_groups)

    zp_u = unpack_from_int32(packed_zp, 4, torch.Size([N, num_groups]), packed_dim=0)
    zp_u = (zp_u.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8)
    kernel_zp = zp_u[1::2, :] * 16 + zp_u[0::2, :]  # [N // 2, num_groups]
    assert kernel_zp.shape == (N // 2, num_groups)

    # Kernel indexes row n // 2 and shifts by (n % 2) * 4.
    for n in range(N):
        got = (kernel_zp[n // 2] >> ((n % 2) * 4)) & 0xF
        torch.testing.assert_close(got.to(torch.int16), zp[n].to(torch.int16) + 8)

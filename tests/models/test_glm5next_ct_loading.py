# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regressions for GLM5Next compressed-tensors weight loading."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.model_executor.layers.linear as linear_module
import vllm.model_executor.layers.mla as mla_module
import vllm.model_executor.models.glm4_1v as glm4_module
import vllm.model_executor.parameter as parameter_module
import vllm.models.glm5next.nvidia.attention as attention_module
import vllm.models.glm5next.nvidia.model as model_module
import vllm.models.glm5next.nvidia.mtp as mtp_module
from vllm.model_executor.layers.linear import LinearBase, ReplicatedLinear
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
from vllm.models.glm5next.nvidia.attention import Glm5NextMLAAttention
from vllm.models.glm5next.nvidia.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
    _try_load_fp8_attn_proj,
)
from vllm.models.glm5next.nvidia.mtp import Glm5NextMTP

pytestmark = pytest.mark.cpu_test


class _RecordingParameter:
    def __init__(self) -> None:
        self.loads: list[tuple[torch.Tensor, tuple, dict]] = []

    def weight_loader(self, param, tensor, *args, **kwargs) -> None:
        self.loads.append((tensor.clone(), args, kwargs))


class _FakeMainModel(nn.Module):
    def __init__(self, params: dict[str, _RecordingParameter]) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            is_moe=False,
            num_nextn_predict_layers=0,
            mla_nope=False,
            qk_rope_head_dim=0,
            is_linear_attn=True,
        )
        self.params = params

    def named_parameters(self, *args, **kwargs):
        return iter(self.params.items())


@pytest.mark.parametrize("scale_first", [False, True])
@pytest.mark.parametrize(
    ("projection", "target", "expected_shard"),
    [("o_proj", "o_proj", None), ("q_a_proj", "fused_qkv_a_proj", 0)],
)
def test_raw_fp8_attention_still_dequantizes_in_both_orders(
    scale_first: bool,
    projection: str,
    target: str,
    expected_shard: int | None,
) -> None:
    source_prefix = f"layers.3.self_attn.{projection}"
    target_name = f"layers.3.self_attn.{target}.weight"
    param = nn.Parameter(torch.empty(128, 128, dtype=torch.bfloat16))
    loaded_shards: list[int | None] = []

    def load(p, weight, shard=None):
        p.data.copy_(weight)
        loaded_shards.append(shard)

    param.weight_loader = load
    params = {target_name: param}
    tensors = [
        (
            f"{source_prefix}.weight",
            torch.ones(128, 128, dtype=torch.float8_e4m3fn),
        ),
        (f"{source_prefix}.weight_scale_inv", torch.tensor([[2.0]])),
    ]
    if scale_first:
        tensors.reverse()

    pending: dict = {}
    loaded: set[str] = set()
    for name, tensor in tensors:
        assert _try_load_fp8_attn_proj(
            name, tensor, pending, params, loaded, kv_a_pad_size=0
        )

    torch.testing.assert_close(param, torch.full_like(param, 2))
    assert loaded == {target_name}
    assert loaded_shards == [expected_shard]


@pytest.mark.parametrize("projection", ["q_a_proj", "kv_a_proj_with_mqa", "o_proj"])
def test_ct_fp8_bypasses_raw_fp8_loader(projection: str) -> None:
    prefix = f"layers.3.self_attn.{projection}"
    params = {
        f"{prefix}.weight": nn.Parameter(
            torch.empty(128, 128, dtype=torch.float8_e4m3fn), requires_grad=False
        ),
        f"{prefix}.weight_scale": nn.Parameter(torch.ones(1, 1)),
    }
    pending: dict = {}
    loaded: set[str] = set()

    assert not _try_load_fp8_attn_proj(
        f"{prefix}.weight",
        params[f"{prefix}.weight"],
        pending,
        params,
        loaded,
        kv_a_pad_size=0,
    )
    assert not pending
    assert not loaded


def test_ct_unfused_mla_wrapper_pads_only_nope_activation(monkeypatch) -> None:
    for module in (linear_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        attention_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    class FakeMLAAttention(nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.layer_name = kwargs["prefix"]
            self.k_pe = None

        def forward(self, q, kv_c, k_pe, output_shape, **kwargs):
            self.k_pe = k_pe
            return q.new_zeros(output_shape)

    rmsnorm_inputs = []

    def record_fused_rmsnorm(q_c, kv_c, *args):
        rmsnorm_inputs.append((q_c, kv_c))
        return q_c, kv_c

    monkeypatch.setattr(mla_module, "MLAAttention", FakeMLAAttention)
    monkeypatch.setattr(mla_module, "fused_q_kv_rmsnorm", record_fused_rmsnorm)
    quant_config = compressed_tensors.CompressedTensorsConfig(
        target_scheme_map={},
        ignore=[],
        quant_format="mixed-precision",
    )
    config = SimpleNamespace(
        rms_norm_eps=1e-5,
        rope_parameters=None,
        index_topk=None,
    )
    attention = Glm5NextMLAAttention(
        object(),
        config,
        hidden_size=8,
        num_heads=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=3,
        v_head_dim=2,
        q_lora_rank=4,
        kv_lora_rank=2,
        quant_config=quant_config,
        prefix="layers.0.self_attn",
        skip_rope=True,
    )
    assert attention.q_a_proj is not None
    assert attention.kv_a_proj_with_mqa is not None
    for projection in (attention.q_a_proj, attention.kv_a_proj_with_mqa):
        projection.register_parameter(
            "weight_scale", nn.Parameter(torch.randn(1, 1), requires_grad=False)
        )
    for linear in attention.modules():
        if not isinstance(linear, LinearBase):
            continue
        # create_weights allocates with torch.empty; make the forward and the
        # immutability snapshot deterministic instead of reading undefined data.
        nn.init.normal_(linear.weight)
        # On CPU the unquantized GEMM needs layer.cpu_linear, which only
        # process_weights_after_loading installs; the skip flag keeps weights.
        linear._cpu_skip_gemm_dispatch = True
        linear.quant_method.process_weights_after_loading(linear)
    before = {
        name: parameter.detach().clone()
        for name, parameter in attention.named_parameters()
        if name.startswith(("q_a_proj.", "kv_a_proj_with_mqa."))
    }

    output = attention(torch.randn(5, 8), torch.arange(5))

    assert output.shape == (5, 8)
    assert attention.mla_attn.fuse_qkv_rmsnorm
    assert len(rmsnorm_inputs) == 1
    assert rmsnorm_inputs[0][1].shape == (5, 2)
    assert attention.mla_attn.mla_attn.k_pe.shape == (5, 1, 3)
    assert not torch.count_nonzero(attention.mla_attn.mla_attn.k_pe)
    parameter_names = dict(attention.named_parameters())
    assert "q_a_proj.weight" in parameter_names
    assert "kv_a_proj_with_mqa.weight" in parameter_names
    assert not any("fused_qkv_a_proj" in name for name in parameter_names)
    for name, expected in before.items():
        torch.testing.assert_close(parameter_names[name], expected, rtol=0, atol=0)


def test_stacked_loader_prefers_heterogeneous_ct_destinations() -> None:
    packed_name = "layers.4.self_attn.q_proj.weight_packed"
    bf16_name = "layers.4.self_attn.b_proj.weight"
    destinations = {
        packed_name: _RecordingParameter(),
        bf16_name: _RecordingParameter(),
    }
    model = _FakeMainModel(destinations)
    packed = torch.arange(8, dtype=torch.int32)
    bf16 = torch.arange(8, dtype=torch.bfloat16)

    loaded = Glm5NextModel.load_weights(
        model, [(packed_name, packed), (bf16_name, bf16)]
    )

    assert loaded == set(destinations)
    torch.testing.assert_close(destinations[packed_name].loads[0][0], packed)
    torch.testing.assert_close(destinations[bf16_name].loads[0][0], bf16)
    assert destinations[packed_name].loads[0][1] == ()
    assert destinations[bf16_name].loads[0][1] == ()


def test_ct_ignored_kv_a_projection_does_not_pad_weight(monkeypatch) -> None:
    # Exercise the real shape-enforcing loader without a distributed runtime.
    for module in (linear_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    prefix = "layers.4.self_attn.kv_a_proj_with_mqa"
    name = f"{prefix}.weight"
    quant_config = compressed_tensors.CompressedTensorsConfig(
        target_scheme_map={"Linear": {"weights": "would-quantize"}},
        ignore=[prefix],
        quant_format="mixed-precision",
    )
    projection = ReplicatedLinear(
        8,
        2,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
        prefix=prefix,
    )
    assert not hasattr(projection, "weight_scale")
    param = projection.weight
    model = _FakeMainModel({name: param})
    model.config.mla_nope = True
    model.config.qk_rope_head_dim = 3
    weight = torch.arange(16, dtype=torch.bfloat16).view(2, 8)

    loaded = Glm5NextModel.load_weights(model, [(name, weight)])

    assert loaded == {name}
    torch.testing.assert_close(param, weight)


def test_combined_kda_conv_weight_loads_three_destinations() -> None:
    prefix = "layers.4.self_attn"
    destinations = {
        f"{prefix}.{projection}_conv1d.weight": _RecordingParameter()
        for projection in ("q", "k", "v")
    }
    model = _FakeMainModel(destinations)
    combined = torch.arange(24).view(6, 1, 4)

    loaded = Glm5NextModel.load_weights(model, [(f"{prefix}.conv1d.weight", combined)])

    assert loaded == set(destinations)
    for destination, expected in zip(destinations.values(), combined.chunk(3, dim=0)):
        assert len(destination.loads) == 1
        torch.testing.assert_close(destination.loads[0][0], expected)


def test_wrapper_loaders_pass_their_early_ct_mapper(monkeypatch) -> None:
    calls: list[object] = []

    class Loader:
        def __init__(self, module) -> None:
            calls.append(module)

        def load_weights(self, weights, mapper):
            assert list(weights) == [("unused", torch.ones(1))]
            calls.append(mapper)
            return {"loaded"}

    monkeypatch.setattr(model_module, "AutoWeightsLoader", Loader)
    monkeypatch.setattr(glm4_module, "AutoWeightsLoader", Loader)
    causal = object.__new__(Glm5NextForCausalLM)
    conditional = object.__new__(Glm5NextForConditionalGeneration)

    assert Glm5NextForCausalLM.load_weights(causal, [("unused", torch.ones(1))]) == {
        "loaded"
    }
    assert Glm5NextForConditionalGeneration.load_weights(
        conditional, [("unused", torch.ones(1))]
    ) == {"loaded"}
    assert calls == [
        causal,
        Glm5NextForCausalLM.hf_to_vllm_mapper,
        conditional,
        Glm5NextForConditionalGeneration.hf_to_vllm_mapper,
    ]


def test_entrypoint_mappers_cover_wrapper_and_internal_names() -> None:
    mm_mapper = Glm5NextForConditionalGeneration.hf_to_vllm_mapper
    assert mm_mapper._map_name("model.visual.patch_embed.weight") == (
        "visual.patch_embed.weight"
    )
    assert mm_mapper._map_name("lm_head.weight") == "language_model.lm_head.weight"
    assert (
        mm_mapper._map_name(
            "model.language_model.layers.4.self_attn.forget_gate.f_a_proj.weight"
        )
        == "language_model.model.layers.4.self_attn.f_a_proj.weight"
    )
    assert mm_mapper._map_name("model.language_model.layers.4.attn_hc.fn") == (
        "language_model.model.layers.4.hc_attn_fn"
    )
    assert mm_mapper._map_name("model.language_model.layers.4.ffn_hc.base") == (
        "language_model.model.layers.4.hc_ffn_base"
    )
    assert Glm5NextForCausalLM.packed_modules_mapping == {
        "gate_up_proj": ["gate_proj", "up_proj"]
    }
    assert Glm5NextForConditionalGeneration.packed_modules_mapping == (
        Glm5NextForCausalLM.packed_modules_mapping
    )
    assert Glm5NextMTP.packed_modules_mapping == (
        Glm5NextForCausalLM.packed_modules_mapping
    )


def test_exact_ct_targets_win_over_linear_fallback_after_mapping() -> None:
    target = "model.language_model.layers.45.self_attn.q_a_proj"
    ignored = "model.language_model.layers.4.self_attn.forget_gate.f_a_proj"
    exact_scheme = {"weights": "exact"}
    fallback_scheme = {"weights": "fallback"}
    quant_config = compressed_tensors.CompressedTensorsConfig(
        target_scheme_map={target: exact_scheme, "Linear": fallback_scheme},
        ignore=[ignored],
        quant_format="mixed-precision",
    )
    quant_config.packed_modules_mapping = (
        Glm5NextForConditionalGeneration.packed_modules_mapping
    )

    quant_config.apply_vllm_mapper(Glm5NextForConditionalGeneration.hf_to_vllm_mapper)
    mapped_target = "language_model.model.layers.45.self_attn.q_a_proj"

    assert quant_config.get_scheme_dict(nn.Linear(1, 1), mapped_target) is exact_scheme
    assert quant_config.ignore == ["language_model.model.layers.4.self_attn.f_a_proj"]


def test_mtp_loader_prefers_source_named_ct_destination(monkeypatch) -> None:
    monkeypatch.setattr(
        mtp_module, "fused_moe_make_expert_params_mapping", lambda *args, **kwargs: []
    )
    source = "model.language_model.layers.45.self_attn.q_a_proj.weight_scale"
    target = "model.layers.45.mtp_block.self_attn.q_a_proj.weight_scale"
    destination = _RecordingParameter()

    model = object.__new__(Glm5NextMTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=45,
        num_nextn_predict_layers=1,
        n_routed_experts=1,
        mla_nope=False,
        qk_rope_head_dim=0,
    )
    model.num_redundant_experts = 0
    model.model = SimpleNamespace(mtp_start_layer_idx=45, num_mtp_layers=1)
    model.named_parameters = lambda *args, **kwargs: iter([(target, destination)])

    loaded = Glm5NextMTP.load_weights(model, [(source, torch.ones(1, 1))])

    assert loaded == {target}
    assert len(destination.loads) == 1

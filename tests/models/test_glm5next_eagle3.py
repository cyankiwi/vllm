# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.layers.mhc import hc_contract
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.models.glm5next.nvidia import model as glm5_model
from vllm.models.glm5next.nvidia.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)


def _make_model(layers, capture_layers=(), sequence_parallel=False):
    model = object.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    object.__setattr__(model, "layers", layers)
    object.__setattr__(model, "_active_layers", layers)
    object.__setattr__(model, "start_layer", 0)
    object.__setattr__(model, "end_layer", len(layers))
    object.__setattr__(model, "aux_hidden_state_layers", capture_layers)
    object.__setattr__(model, "is_sequence_parallel", sequence_parallel)
    object.__setattr__(model, "norm", nn.Identity())
    return model


class _DeferredMhcLayer:
    def __init__(self, hidden_states, residual, post, comb):
        self.hidden_states = hidden_states
        self.residual = residual
        self.post = post
        self.comb = comb
        self.n = residual.shape[1]
        self.hc_post_calls = 0

    def __call__(self, positions, hidden_states, residual, post, comb):
        return self.hidden_states, self.residual, self.post, self.comb

    def hc_post(self, hidden_states, residual, post, comb):
        self.hc_post_calls += 1
        return post * hidden_states[:, None, :] + torch.einsum(
            "tij,tih->tjh", comb, residual
        )


class _FinalLayer:
    def __init__(self, output):
        self.output = output
        self.received = None

    def __call__(self, positions, hidden_states, residual, post, comb):
        self.received = (hidden_states, residual, post, comb)
        return self.output, None, None, None


def test_aux_capture_materializes_mhc_without_mutating_forward_state(monkeypatch):
    torch.manual_seed(0)
    tokens, streams, hidden_size = 5, 4, 8
    pending = torch.randn(tokens, hidden_size)
    residual = torch.randn(tokens, streams, hidden_size)
    post = torch.randn(tokens, streams, 1)
    comb = torch.randn(tokens, streams, streams)
    final = torch.randn(tokens, hidden_size)
    deferred = _DeferredMhcLayer(pending, residual, post, comb)
    final_layer = _FinalLayer(final)
    model = _make_model([deferred, final_layer], capture_layers=(1,))
    monkeypatch.setattr(
        glm5_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.arange(tokens),
        intermediate_tensors=None,
        inputs_embeds=torch.randn(tokens, hidden_size),
    )

    expected = hc_contract(
        post * pending[:, None, :] + torch.einsum("tij,tih->tjh", comb, residual),
        streams,
    )
    torch.testing.assert_close(output, final)
    torch.testing.assert_close(aux_hidden_states[0], expected)
    assert deferred.hc_post_calls == 1
    assert final_layer.received is not None
    assert final_layer.received[0] is pending
    assert final_layer.received[1] is residual
    assert final_layer.received[2] is post
    assert final_layer.received[3] is comb


def test_aux_capture_preserves_tensor_return_without_taps(monkeypatch):
    final = torch.randn(3, 8)
    model = _make_model([_FinalLayer(final)])
    monkeypatch.setattr(
        glm5_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    output = model.forward(
        input_ids=None,
        positions=torch.arange(3),
        intermediate_tensors=None,
        inputs_embeds=torch.randn(3, 8),
    )

    assert output is final


def test_aux_capture_does_not_double_add_non_mhc_residual():
    hidden_states = torch.randn(3, 8)
    residual = torch.randn(3, 8)
    model = _make_model([None])

    output = model._aux_hidden_state(
        1, hidden_states, residual, None, None, hidden_states.shape[0]
    )

    assert output is hidden_states


def test_aux_capture_uses_mrope_token_dimension_under_sequence_parallel(monkeypatch):
    tokens = 5
    final = torch.randn(tokens, 8)
    model = _make_model(
        [_FinalLayer(final)], capture_layers=(1,), sequence_parallel=True
    )
    monkeypatch.setattr(
        glm5_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(glm5_model, "sp_shard", lambda value: value)
    monkeypatch.setattr(glm5_model, "sp_all_gather", lambda value: value)

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.arange(tokens).repeat(3, 1),
        intermediate_tensors=None,
        inputs_embeds=torch.randn(tokens, 8),
    )

    assert output.shape[0] == tokens
    assert aux_hidden_states[0].shape[0] == tokens


def test_glm_wrappers_delegate_eagle3_configuration_and_output(monkeypatch):
    assert supports_eagle3(Glm5NextForCausalLM)
    assert supports_eagle3(Glm5NextForConditionalGeneration)

    inner = _make_model([None] * 45)
    causal = object.__new__(Glm5NextForCausalLM)
    nn.Module.__init__(causal)
    object.__setattr__(causal, "model", inner)
    causal.set_aux_hidden_state_layers((6, 15, 25, 34, 43))
    assert inner.aux_hidden_state_layers == (6, 15, 25, 34, 43)

    tokens = 3
    final = torch.randn(tokens, 8)
    inner = _make_model([_FinalLayer(final)])
    multimodal = object.__new__(Glm5NextForConditionalGeneration)
    nn.Module.__init__(multimodal)
    language_model = SimpleNamespace(
        model=inner,
        embed_input_ids=lambda _: None,
        forward=lambda input_ids, positions: None,
    )
    object.__setattr__(multimodal, "language_model", language_model)
    object.__setattr__(multimodal, "_language_model_names", ["language_model"])
    multimodal.set_aux_hidden_state_layers((1,))
    assert inner.aux_hidden_state_layers == (1,)
    monkeypatch.setattr(
        glm5_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    output, aux_hidden_states = multimodal.forward(
        input_ids=None,
        positions=torch.arange(tokens).repeat(3, 1),
        inputs_embeds=torch.randn(tokens, 8),
    )

    torch.testing.assert_close(output, final)
    torch.testing.assert_close(aux_hidden_states[0], final)

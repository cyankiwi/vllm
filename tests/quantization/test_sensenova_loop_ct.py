# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.models.sensenova_loop import (
    _expand_layer_name,
    _loop_destinations,
)


@pytest.mark.parametrize("suffix", ["mlp.down_proj", "linear_attn.in_proj_a"])
def test_expand_physical_ct_target_or_ignore(suffix):
    destinations = _loop_destinations([0, 1, 2, 1, 2, 3])
    name = f"language_model.model.layers.2.{suffix}"
    assert list(_expand_layer_name(name, destinations)) == [
        f"language_model.model.layers.{index}.{suffix}" for index in (2, 4)
    ]
    for name in ("Linear", "lm_head", "re:.*mtp.*", "visual.blocks.0.attn.qkv"):
        assert list(_expand_layer_name(name, destinations)) == [name]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.model_executor.models.glm4_1v import Glm4vProcessingInfo
from vllm.models.glm5next.nvidia.multimodal import Glm5NextProcessingInfo

pytestmark = pytest.mark.cpu_test

_PIXELS_PER_TOKEN = 2 * (14 * 2) ** 2


def _processor():
    return SimpleNamespace(
        min_image_tokens=16,
        max_image_tokens=8000,
        patch_size=14,
        merge_size=2,
        temporal_patch_size=2,
    )


def test_pixel_budget_honors_runtime_token_limits() -> None:
    info = Mock(spec=Glm5NextProcessingInfo)
    info.ctx = Mock()
    info.ctx.get_merged_mm_kwargs.return_value = {
        "min_image_tokens": "8",
        "max_image_tokens": "167",
    }

    budget = Glm5NextProcessingInfo._processor_pixel_budget(info, _processor(), "image")

    assert budget == (8 * _PIXELS_PER_TOKEN, 167 * _PIXELS_PER_TOKEN)
    info.ctx.get_merged_mm_kwargs.assert_called_once_with({}, modality="image")


@pytest.mark.parametrize("modality", [None, "image"])
def test_pixel_budget_none_override_uses_processor_defaults(modality) -> None:
    info = Mock(spec=Glm5NextProcessingInfo)
    info.ctx = Mock()
    info.ctx.get_merged_mm_kwargs.return_value = {
        "min_image_tokens": None,
        "max_image_tokens": None,
    }

    budget = Glm5NextProcessingInfo._processor_pixel_budget(
        info, _processor(), modality
    )

    assert budget == (16 * _PIXELS_PER_TOKEN, 8000 * _PIXELS_PER_TOKEN)
    info.ctx.get_merged_mm_kwargs.assert_called_once_with({}, modality=modality)


def test_pixel_budget_keeps_missing_limit_error() -> None:
    info = Mock(spec=Glm5NextProcessingInfo)
    info.ctx = Mock()
    info.ctx.get_merged_mm_kwargs.return_value = {}
    processor = _processor()
    processor.max_image_tokens = None

    with pytest.raises(ValueError, match="max_image_tokens must be provided"):
        Glm5NextProcessingInfo._processor_pixel_budget(info, processor, "image")


def test_dummy_frames_skip_video_search_when_video_is_disabled() -> None:
    info = Mock(spec=Glm5NextProcessingInfo)

    num_frames = Glm5NextProcessingInfo.get_num_frames_with_most_features(
        info, seq_len=2048, mm_counts={"image": 1, "video": 0}
    )

    assert num_frames == 1
    info._get_video_max_pixels.assert_not_called()
    info._get_vision_info.assert_not_called()


def test_enabled_video_delegates_to_inherited_search(monkeypatch) -> None:
    inherited = Mock(return_value=37)
    monkeypatch.setattr(
        Glm4vProcessingInfo, "get_num_frames_with_most_features", inherited
    )
    info = Mock(spec=Glm5NextProcessingInfo)

    num_frames = Glm5NextProcessingInfo.get_num_frames_with_most_features(
        info, seq_len=2048, mm_counts={"image": 0, "video": 1}
    )

    assert num_frames == 37
    inherited.assert_called_once_with(2048, {"image": 0, "video": 1})

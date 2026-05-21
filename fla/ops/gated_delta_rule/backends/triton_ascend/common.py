# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for gated-delta-rule triton-ascend backend (Huawei Ascend NPU)."""

from fla.utils import IS_NPU, is_npu_available


def triton_ascend_is_active() -> bool:
    """True when running on Ascend NPU with ``torch.npu`` (triton-ascend stack)."""
    return IS_NPU and is_npu_available()

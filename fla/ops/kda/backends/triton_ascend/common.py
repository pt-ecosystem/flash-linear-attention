# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the triton-ascend KDA backend."""

from fla.utils import IS_NPU, is_npu_available


def triton_ascend_is_active() -> bool:
    """True when running on Ascend NPU with triton-ascend as the Triton backend."""
    return IS_NPU and is_npu_available()

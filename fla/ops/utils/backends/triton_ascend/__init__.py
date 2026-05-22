# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triton-Ascend backend for utils ops (Huawei Ascend NPU)."""

from __future__ import annotations

from functools import cache

import torch

from fla.ops.backends import BaseBackend

from . import cumsum as cumsum_ascend
from .common import triton_ascend_is_active


class UtilsTritonAscendBackend(BaseBackend):
    """Utils backend for Ascend NPUs using triton-ascend.

    Active when ``IS_NPU`` and ``torch.npu`` are available. Disable with ``FLA_TRITON_ASCEND=0``.

    Dispatched entry points mirror ``@dispatch("utils")`` on ``fla.ops.utils.cumsum``.
    """

    backend_type = "triton-ascend"
    package_name = None
    env_var = "FLA_TRITON_ASCEND"
    default_enable = True
    priority = 4

    @classmethod
    def is_available(cls) -> bool:
        return triton_ascend_is_active()

    @classmethod
    @cache
    def can_use(cls) -> bool:
        return cls.is_available() and cls.is_enabled()

    def _ascend_verifier(self, g: torch.Tensor, *args, **kwargs) -> tuple[bool, str | None]:
        if not triton_ascend_is_active():
            return False, "triton-ascend backend requires Ascend NPU (IS_NPU, torch.npu available)"
        return True, None

    def chunk_local_cumsum_verifier(self, g: torch.Tensor, *args, **kwargs) -> tuple[bool, str | None]:
        return self._ascend_verifier(g, *args, **kwargs)

    def chunk_local_cumsum(self, g: torch.Tensor, chunk_size: int, **kwargs):
        return cumsum_ascend.chunk_local_cumsum(g=g, chunk_size=chunk_size, **kwargs)


__all__ = ["UtilsTritonAscendBackend", "triton_ascend_is_active"]

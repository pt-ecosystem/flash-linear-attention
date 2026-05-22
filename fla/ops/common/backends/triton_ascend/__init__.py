# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triton-Ascend backend for shared common chunk ops (Huawei Ascend NPU)."""

from __future__ import annotations

from functools import cache

import torch

from fla.ops.backends import BaseBackend

from . import chunk_delta_h as chunk_delta_h_ascend
from . import chunk_o as chunk_o_ascend
from .common import triton_ascend_is_active


class CommonTritonAscendBackend(BaseBackend):
    """Common chunk backend for Ascend NPUs using triton-ascend.

    Active when ``IS_NPU`` and ``torch.npu`` are available. Disable with ``FLA_TRITON_ASCEND=0``.

    Dispatched entry points mirror ``@dispatch("common")`` on
    ``fla.ops.common.chunk_delta_h`` and ``fla.ops.common.chunk_o``.
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

    def _ascend_verifier(self, tensor: torch.Tensor, *args, **kwargs) -> tuple[bool, str | None]:
        if not triton_ascend_is_active():
            return False, "triton-ascend backend requires Ascend NPU (IS_NPU, torch.npu available)"
        return True, None

    def chunk_gated_delta_rule_fwd_h_verifier(self, k: torch.Tensor, *args, **kwargs) -> tuple[bool, str | None]:
        return self._ascend_verifier(k, *args, **kwargs)

    def chunk_gated_delta_rule_fwd_h(self, k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, **kwargs):
        return chunk_delta_h_ascend.chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, **kwargs)

    def chunk_gated_delta_rule_bwd_dhu_verifier(self, q: torch.Tensor, *args, **kwargs) -> tuple[bool, str | None]:
        return self._ascend_verifier(q, *args, **kwargs)

    def chunk_gated_delta_rule_bwd_dhu(self, q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, **kwargs):
        return chunk_delta_h_ascend.chunk_gated_delta_rule_bwd_dhu(q=q, k=k, w=w, **kwargs)

    def chunk_fwd_o_verifier(self, q: torch.Tensor, *args, **kwargs) -> tuple[bool, str | None]:
        return self._ascend_verifier(q, *args, **kwargs)

    def chunk_fwd_o(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h: torch.Tensor, **kwargs):
        return chunk_o_ascend.chunk_fwd_o(q=q, k=k, v=v, h=h, **kwargs)

    def chunk_bwd_dv_local_verifier(self, q: torch.Tensor, *args, **kwargs) -> tuple[bool, str | None]:
        return self._ascend_verifier(q, *args, **kwargs)

    def chunk_bwd_dv_local(self, q: torch.Tensor, k: torch.Tensor, do: torch.Tensor, **kwargs):
        return chunk_o_ascend.chunk_bwd_dv_local(q=q, k=k, do=do, **kwargs)

    def chunk_bwd_dqkwg_verifier(self, q: torch.Tensor, *args, **kwargs) -> tuple[bool, str | None]:
        return self._ascend_verifier(q, *args, **kwargs)

    def chunk_bwd_dqkwg(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do: torch.Tensor, **kwargs):
        return chunk_o_ascend.chunk_bwd_dqkwg(q=q, k=k, v=v, do=do, **kwargs)


__all__ = ["CommonTritonAscendBackend", "triton_ascend_is_active"]

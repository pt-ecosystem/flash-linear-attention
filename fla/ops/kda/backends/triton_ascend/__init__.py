# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triton-Ascend backend for KDA (Huawei Ascend NPU)."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import torch

from fla.ops.backends import BaseBackend
from fla.ops.kda.backends.triton_ascend.chunk import ChunkKDAFunctionAscend
from fla.ops.kda.backends.triton_ascend.chunk_bwd import chunk_kda_bwd_wy_dqkg_fused_ascend
from fla.ops.kda.backends.triton_ascend.common import triton_ascend_is_active

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext


class KDATritonAscendBackend(BaseBackend):
    """KDA backend for Ascend NPUs using triton-ascend.

    Enabled on NPU when Triton reports backend ``npu`` and ``torch.npu`` is available.
    Disable with ``FLA_TRITON_ASCEND=0``.

    Implements:
    - ``chunk_kda``: full forward/backward via ``ChunkKDAFunctionAscend``
    - ``chunk_kda_bwd_wy_dqkg_fused``: NPU-tuned fused backward kernel
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

    def _ascend_verifier(self, *args, **kwargs) -> tuple[bool, str | None]:
        if not triton_ascend_is_active():
            return False, "triton-ascend backend requires Ascend NPU (Triton backend=npu, torch.npu available)"
        if kwargs.get("cp_context") is not None:
            return False, "triton-ascend KDA does not support context parallel yet"
        return True, None

    def chunk_kda_verifier(self, *args, **kwargs) -> tuple[bool, str | None]:
        return self._ascend_verifier(*args, **kwargs)

    def chunk_kda_bwd_wy_dqkg_fused_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> tuple[bool, str | None]:
        ok, reason = self._ascend_verifier(q, k, v, **kwargs)
        if not ok:
            return ok, reason
        if v.shape[2] != k.shape[2]:
            return False, (
                "triton-ascend KDA does not support GQA in the fused bwd kernel; "
                "falling back to default Triton"
            )
        return True, None

    def chunk_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        cp_context: FLACPContext | None = None,
        chunk_size: int = 64,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        **kwargs,
    ):
        if scale is None:
            scale = q.shape[-1] ** -0.5

        return ChunkKDAFunctionAscend.apply(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            scale,
            initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel,
            use_gate_in_kernel,
            use_beta_sigmoid_in_kernel,
            state_v_first,
            cu_seqlens,
            cu_seqlens_cpu,
            safe_gate,
            lower_bound,
            chunk_size,
            disable_recompute,
            return_intermediate_states,
            cp_context,
        )

    def chunk_kda_bwd_wy_dqkg_fused(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        v_new: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        h: torch.Tensor,
        do: torch.Tensor,
        dh: torch.Tensor,
        dv: torch.Tensor,
        scale: float | None = None,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.LongTensor | None = None,
    ):
        return chunk_kda_bwd_wy_dqkg_fused_ascend(
            q=q,
            k=k,
            v=v,
            v_new=v_new,
            g=g,
            beta=beta,
            A=A,
            h=h,
            do=do,
            dh=dh,
            dv=dv,
            scale=scale,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )


__all__ = ['KDATritonAscendBackend']

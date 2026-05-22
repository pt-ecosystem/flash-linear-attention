# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triton-Ascend backend for gated delta rule (Huawei Ascend NPU)."""

from __future__ import annotations

from functools import cache

import torch

from fla.ops.backends import BaseBackend

from . import chunk_fwd as chunk_fwd_ascend
from . import fused_recurrent as fused_recurrent_ascend
from . import gate as gate_ascend
from . import wy_fast as wy_fast_ascend
from .common import triton_ascend_is_active


class GatedDeltaRuleTritonAscendBackend(BaseBackend):
    """Gated delta rule backend for Ascend NPUs using triton-ascend.

    Active when ``IS_NPU`` and ``torch.npu`` are available. Disable with ``FLA_TRITON_ASCEND=0``.

    Dispatched entry points mirror ``@dispatch("gated_delta_rule")`` on default implementations
    under ``fla.ops.gated_delta_rule`` (chunk_fwd / gate / wy_fast / fused_recurrent).
    Shared common/utils ops use ``common`` and ``utils`` registries respectively.
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
            return False, "triton-ascend backend requires Ascend NPU (IS_NPU, torch.npu available)"
        if kwargs.get("cp_context") is not None:
            return False, "triton-ascend gated_delta_rule does not support context parallel yet"
        return True, None

    def chunk_gated_delta_rule_fwd_intra_verifier(
        self, k: torch.Tensor, *args, **kwargs
    ) -> tuple[bool, str | None]:
        return self._ascend_verifier(k, *args, **kwargs)

    def chunk_gated_delta_rule_fwd_intra(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.LongTensor | None = None,
    ):
        return chunk_fwd_ascend.chunk_gated_delta_rule_fwd_intra(
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )

    def gdn_gate_chunk_cumsum_verifier(self, g: torch.Tensor, *args, **kwargs) -> tuple[bool, str | None]:
        return self._ascend_verifier(g, *args, **kwargs)

    def gdn_gate_chunk_cumsum(
        self,
        g: torch.Tensor,
        A_log: torch.Tensor,
        chunk_size: int,
        scale: float = None,
        dt_bias: torch.Tensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        output_dtype: torch.dtype | None = torch.float,
    ):
        return gate_ascend.gdn_gate_chunk_cumsum(
            g=g,
            A_log=A_log,
            chunk_size=chunk_size,
            scale=scale,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            output_dtype=output_dtype,
        )

    def gdn_gate_bwd_verifier(self, g: torch.Tensor, *args, **kwargs) -> tuple[bool, str | None]:
        return self._ascend_verifier(g, *args, **kwargs)

    def gdn_gate_bwd(
        self,
        g: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor | None,
        dyg: torch.Tensor,
    ):
        return gate_ascend.gdn_gate_bwd(g=g, A_log=A_log, dt_bias=dt_bias, dyg=dyg)

    def recompute_w_u_fwd_verifier(self, k: torch.Tensor, *args, **kwargs) -> tuple[bool, str | None]:
        return self._ascend_verifier(k, *args, **kwargs)

    def recompute_w_u_fwd(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        g: torch.Tensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
    ):
        return wy_fast_ascend.recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    def prepare_wy_repr_bwd_verifier(self, k: torch.Tensor, *args, **kwargs) -> tuple[bool, str | None]:
        return self._ascend_verifier(k, *args, **kwargs)

    def prepare_wy_repr_bwd(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        dw: torch.Tensor,
        du: torch.Tensor,
        g: torch.Tensor = None,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
    ):
        return wy_fast_ascend.prepare_wy_repr_bwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            dw=dw,
            du=du,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    def fused_recurrent_gated_delta_rule_fwd_verifier(
        self, q: torch.Tensor, *args, **kwargs
    ) -> tuple[bool, str | None]:
        return self._ascend_verifier(q, *args, **kwargs)

    def fused_recurrent_gated_delta_rule_fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        gv: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        scale: float = None,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        return fused_recurrent_ascend.fused_recurrent_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            gk=gk,
            gv=gv,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
        )


__all__ = ["GatedDeltaRuleTritonAscendBackend", "triton_ascend_is_active"]

# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triton-Ascend backend for KDA (Huawei Ascend NPU)."""

from __future__ import annotations

from functools import cache

import torch

from fla.ops.backends import BaseBackend
from fla.ops.kda.backends.triton_ascend.common import triton_ascend_is_active
from fla.ops.kda.backends.triton_ascend.fused_recurrent import fused_recurrent_kda_fwd as fused_recurrent_kda_fwd_ascend


class KDATritonAscendBackend(BaseBackend):
    """KDA backend for Ascend NPUs using triton-ascend.

    Enabled on NPU when Triton reports backend ``npu`` and ``torch.npu`` is available.
    Disable with ``FLA_TRITON_ASCEND=0``.

    Implements:
    - ``fused_recurrent_kda_fwd``: vLLM-style decode with continuous batching on NPU

    ``chunk_kda`` and chunk backward helpers use the default Triton implementation.
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

    def fused_recurrent_kda_fwd_verifier(self, q: torch.Tensor, **kwargs) -> tuple[bool, str | None]:
        return self._ascend_verifier(q, **kwargs)

    def fused_recurrent_kda_fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        scale: float | None = None,
        output_final_state: bool = False,
        inplace_final_state: bool = True,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        ssm_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        lower_bound: float | None = None,
        out: torch.Tensor | None = None,
        **kwargs,
    ):
        return fused_recurrent_kda_fwd_ascend(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            scale=scale,
            output_final_state=output_final_state,
            inplace_final_state=inplace_final_state,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            lower_bound=lower_bound,
            out=out,
            **kwargs,
        )


__all__ = ['KDATritonAscendBackend']

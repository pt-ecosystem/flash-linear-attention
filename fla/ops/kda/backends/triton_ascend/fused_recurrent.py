# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Ascend NPU fused_recurrent_kda_fwd (vLLM-style decode) via triton-ascend."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus

# Tuned for Atlas: avoid autotune bench; match other triton_ascend KDA kernels.
NPU_NUM_WARPS = 2
NPU_NUM_STAGES = 1
NPU_BV = 32


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_kda_fwd_kernel_npu(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    lower_bound,
    scale: tl.constexpr,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    NK = tl.cdiv(K, BK)
    i_k = pid % NK
    pid_rest = pid // NK

    i_v = pid_rest % NV
    i_nh = pid_rest // NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv

    p_g = g + (bos * HV + i_hv) * K + o_k
    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    if STATE_V_FIRST:
        mask_h = mask_v[:, None] & mask_k[None, :]
    else:
        mask_h = mask_k[:, None] & mask_v[None, :]

    if STATE_V_FIRST:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            p_h0 = (
                h0
                + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                    tl.int64
                )
                * stride_init_state_token
            )
            if STATE_V_FIRST:
                p_h0 = p_h0 + i_hv * K * V + o_v[:, None] * K + o_k[None, :]
            else:
                p_h0 = p_h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        else:
            if STATE_V_FIRST:
                p_h0 = h0 + (i_n * HV + i_hv) * K * V + o_v[:, None] * K + o_k[None, :]
            else:
                p_h0 = h0 + (i_n * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in tl.range(0, T, num_stages=num_stages):
        b_q = tl.load(p_q, mask=mask_k, other=0, eviction_policy='evict_last').to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0, eviction_policy='evict_last').to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0, eviction_policy='evict_first').to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_g = tl.load(p_g, eviction_policy='evict_last').to(tl.float32)

        if USE_GATE_IN_KERNEL:
            b_A = tl.load(A_log + i_hv).to(tl.float32)

            if HAS_DT_BIAS:
                b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0).to(tl.float32)
                b_g = b_g + b_bias

            if USE_LOWER_BOUND:
                b_gk = lower_bound * tl.sigmoid(exp(b_A) * b_g)
            else:
                b_gk = -exp(b_A) * softplus(b_g)
        else:
            b_gk = b_g

        if STATE_V_FIRST:
            b_h *= exp(b_gk[None, :])
        else:
            b_h *= exp(b_gk[:, None])

        if STATE_V_FIRST:
            b_v -= tl.sum(b_h * b_k[None, :], 1)
        else:
            b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0, eviction_policy='evict_first').to(tl.float32)
        else:
            b_beta = tl.load(p_beta, eviction_policy='evict_last').to(tl.float32)
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
        b_v *= b_beta
        if STATE_V_FIRST:
            b_h += b_v[:, None] * b_k[None, :]
            b_o = tl.sum(b_h * b_q[None, :], 1)
        else:
            b_h += b_k[:, None] * b_v[None, :]
            b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v, eviction_policy='evict_first')

        if IS_CONTINUOUS_BATCHING:
            if INPLACE_FINAL_STATE:
                p_ht = (
                    ht
                    + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                        tl.int64
                    )
                    * stride_final_state_token
                )
            else:
                p_ht = ht + (bos + i_t) * stride_final_state_token
            if STATE_V_FIRST:
                p_ht = p_ht + i_hv * K * V + o_v[:, None] * K + o_k[None, :]
            else:
                p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_g += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)

    if not IS_CONTINUOUS_BATCHING:
        if STORE_FINAL_STATE:
            if STATE_V_FIRST:
                p_ht = ht + (i_n * HV + i_hv) * K * V + o_v[:, None] * K + o_k[None, :]
            else:
                p_ht = ht + (i_n * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@torch.compiler.disable
def fused_recurrent_kda_fwd(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = k.shape[-1] ** -0.5

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = NPU_BV

    if out is None:
        out = torch.zeros_like(v)
    else:
        assert out.shape == v.shape
    if inplace_final_state:
        assert initial_state is not None
        final_state = initial_state
    elif output_final_state:
        if state_v_first:
            final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
    else:
        final_state = None

    stride_init_state_token = initial_state.stride(0) if initial_state is not None else 1
    stride_final_state_token = final_state.stride(0) if final_state is not None else 1

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (triton.cdiv(V, BV) * N * HV,)
    fused_recurrent_kda_fwd_kernel_npu[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        lower_bound=lower_bound,
        scale=scale,
        N=N,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_GATE_IN_KERNEL=use_gate_in_kernel,
        APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
        STATE_V_FIRST=state_v_first,
        num_warps=NPU_NUM_WARPS,
        num_stages=NPU_NUM_STAGES,
    )

    return out, final_state


__all__ = ['fused_recurrent_kda_fwd']

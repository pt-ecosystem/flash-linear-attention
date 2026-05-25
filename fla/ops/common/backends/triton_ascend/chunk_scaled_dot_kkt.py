# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2

# Fixed launch for Ascend: no autotune; BC/BK tiling avoids UB overflow on 64x64 tiles.
NPU_NUM_WARPS = 2
NPU_NUM_STAGES = 1
NPU_BC = 16
NPU_BK = 32


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    k += (bos * H + i_h // (HV // H)) * K
    A += (bos * HV + i_h) * BT

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    p_b0 = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (i_tc0,), (BC,), (0,))
    p_b1 = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (i_tc1,), (BC,), (0,))
    p_b2 = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (i_tc2,), (BC,), (0,))
    p_b3 = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (i_tc3,), (BC,), (0,))
    b_b0 = tl.load(p_b0, boundary_check=(0,)).to(tl.float32)
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
    b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)

    if USE_G:
        p_g0 = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_tc0,), (BC,), (0,))
        p_g1 = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_tc1,), (BC,), (0,))
        p_g2 = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_tc2,), (BC,), (0,))
        p_g3 = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_tc3,), (BC,), (0,))
        b_g0 = tl.load(p_g0, boundary_check=(0,)).to(tl.float32)
        b_g1 = tl.load(p_g1, boundary_check=(0,)).to(tl.float32)
        b_g2 = tl.load(p_g2, boundary_check=(0,)).to(tl.float32)
        b_g3 = tl.load(p_g3, boundary_check=(0,)).to(tl.float32)

    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k0 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
        p_k2 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
        p_k3 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_k3 = tl.load(p_k3, boundary_check=(0, 1))
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))
        b_A11 += tl.dot(b_k1, tl.trans(b_k1))
        b_A22 += tl.dot(b_k2, tl.trans(b_k2))
        b_A33 += tl.dot(b_k3, tl.trans(b_k3))
        b_A10 += tl.dot(b_k1, tl.trans(b_k0))
        b_A20 += tl.dot(b_k2, tl.trans(b_k0))
        b_A21 += tl.dot(b_k2, tl.trans(b_k1))
        b_A30 += tl.dot(b_k3, tl.trans(b_k0))
        b_A31 += tl.dot(b_k3, tl.trans(b_k1))
        b_A32 += tl.dot(b_k3, tl.trans(b_k2))

    m_d = o_i[:, None] > o_i[None, :]

    if USE_G:
        b_A00 = b_A00 * tl.where(m_d & m_tc0[:, None] & m_tc0[None, :], exp2(b_g0[:, None] - b_g0[None, :]), 0.)
        b_A11 = b_A11 * tl.where(m_d & m_tc1[:, None] & m_tc1[None, :], exp2(b_g1[:, None] - b_g1[None, :]), 0.)
        b_A22 = b_A22 * tl.where(m_d & m_tc2[:, None] & m_tc2[None, :], exp2(b_g2[:, None] - b_g2[None, :]), 0.)
        b_A33 = b_A33 * tl.where(m_d & m_tc3[:, None] & m_tc3[None, :], exp2(b_g3[:, None] - b_g3[None, :]), 0.)
        b_A10 = b_A10 * tl.where(m_tc1[:, None] & m_tc0[None, :], exp2(b_g1[:, None] - b_g0[None, :]), 0.)
        b_A20 = b_A20 * tl.where(m_tc2[:, None] & m_tc0[None, :], exp2(b_g2[:, None] - b_g0[None, :]), 0.)
        b_A21 = b_A21 * tl.where(m_tc2[:, None] & m_tc1[None, :], exp2(b_g2[:, None] - b_g1[None, :]), 0.)
        b_A30 = b_A30 * tl.where(m_tc3[:, None] & m_tc0[None, :], exp2(b_g3[:, None] - b_g0[None, :]), 0.)
        b_A31 = b_A31 * tl.where(m_tc3[:, None] & m_tc1[None, :], exp2(b_g3[:, None] - b_g1[None, :]), 0.)
        b_A32 = b_A32 * tl.where(m_tc3[:, None] & m_tc2[None, :], exp2(b_g3[:, None] - b_g2[None, :]), 0.)
    else:
        b_A00 = tl.where(m_d, b_A00, 0.)
        b_A11 = tl.where(m_d, b_A11, 0.)
        b_A22 = tl.where(m_d, b_A22, 0.)
        b_A33 = tl.where(m_d, b_A33, 0.)

    b_A00 = b_A00 * b_b0[:, None]
    b_A11 = b_A11 * b_b1[:, None]
    b_A22 = b_A22 * b_b2[:, None]
    b_A33 = b_A33 * b_b3[:, None]
    b_A10 = b_A10 * b_b1[:, None]
    b_A20 = b_A20 * b_b2[:, None]
    b_A21 = b_A21 * b_b2[:, None]
    b_A30 = b_A30 * b_b3[:, None]
    b_A31 = b_A31 * b_b3[:, None]
    b_A32 = b_A32 * b_b3[:, None]

    out_ty = A.dtype.element_ty
    p_A00 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_A10 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_A11 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_A20 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_A21 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_A22 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0))
    p_A30 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_A31 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_A32 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
    p_A33 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0))
    tl.store(p_A00, b_A00.to(out_ty), boundary_check=(0, 1))
    tl.store(p_A10, b_A10.to(out_ty), boundary_check=(0, 1))
    tl.store(p_A11, b_A11.to(out_ty), boundary_check=(0, 1))
    tl.store(p_A20, b_A20.to(out_ty), boundary_check=(0, 1))
    tl.store(p_A21, b_A21.to(out_ty), boundary_check=(0, 1))
    tl.store(p_A22, b_A22.to(out_ty), boundary_check=(0, 1))
    tl.store(p_A30, b_A30.to(out_ty), boundary_check=(0, 1))
    tl.store(p_A31, b_A31.to(out_ty), boundary_check=(0, 1))
    tl.store(p_A32, b_A32.to(out_ty), boundary_check=(0, 1))
    tl.store(p_A33, b_A33.to(out_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, HV = *k.shape, beta.shape[2]
    BT = chunk_size
    BC = NPU_BC
    if output_dtype is None:
        output_dtype = k.dtype
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(NPU_BK, triton.next_power_of_2(K))
    # Use empty (not zeros): torch.zeros triggers NPU internal-format hangs in follow-up kernels.
    A = torch.empty(B, T, HV, BT, device=k.device, dtype=output_dtype)
    chunk_scaled_dot_kkt_fwd_kernel[(NT, B * HV)](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        num_warps=NPU_NUM_WARPS,
        num_stages=NPU_NUM_STAGES,
    )
    return A

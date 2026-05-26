# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2

# Tuned for Atlas: avoid autotune bench; match triton_ascend KDA kernels.
NPU_NUM_WARPS = 2
NPU_NUM_STAGES = 1
NPU_BC = 16
NPU_BK = 32
NPU_BV = 16


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A_base = A + (bos * HV + i_h) * BT
    for i_v in range(tl.cdiv(V, BV)):
        for i_br in range(BT // BC):
            t_off = i_t * BT + i_br * BC
            b_u = tl.zeros([BC, BV], dtype=tl.float32)
            for i_bc in range(BT // BC):
                c_off = i_t * BT + i_bc * BC
                p_A = tl.make_block_ptr(A_base, (T, BT), (HV * BT, 1), (t_off, i_bc * BC), (BC, BC), (1, 0))
                b_Ab = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
                p_v = tl.make_block_ptr(v + (bos * HV + i_h) * V, (T, V), (HV * V, 1), (c_off, i_v * BV), (BC, BV), (1, 0))
                p_bc = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (c_off,), (BC,), (0,))
                b_v = tl.load(p_v, boundary_check=(0, 1))
                b_b_c = tl.load(p_bc, boundary_check=(0,))
                b_vb = (b_v * b_b_c[:, None]).to(tl.float32)
                b_u += tl.dot(b_Ab, b_vb, allow_tf32=False)
            p_u = tl.make_block_ptr(u + (bos * HV + i_h) * V, (T, V), (HV * V, 1), (t_off, i_v * BV), (BC, BV), (1, 0))
            tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        for i_br in range(BT // BC):
            t_off = i_t * BT + i_br * BC
            p_b_row = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (t_off,), (BC,), (0,))
            b_b_row = tl.load(p_b_row, boundary_check=(0,))
            if USE_G:
                p_g_row = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (t_off,), (BC,), (0,))
                b_g_row = exp2(tl.load(p_g_row, boundary_check=(0,)))
            b_w = tl.zeros([BC, BK], dtype=tl.float32)
            for i_bc in range(BT // BC):
                c_off = i_t * BT + i_bc * BC
                p_A = tl.make_block_ptr(A_base, (T, BT), (HV * BT, 1), (t_off, i_bc * BC), (BC, BC), (1, 0))
                b_Ab = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
                p_k = tl.make_block_ptr(
                    k + (bos * H + i_h // (HV // H)) * K, (T, K), (H * K, 1), (c_off, i_k * BK), (BC, BK), (1, 0),
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_kb = (b_k * b_b_row[:, None]).to(tl.float32)
                if USE_G:
                    b_kb = b_kb * b_g_row[:, None]
                b_w += tl.dot(b_Ab, b_kb, allow_tf32=False)
            p_w = tl.make_block_ptr(
                w + (bos * HV + i_h) * K, (T, K), (HV * K, 1), (t_off, i_k * BK), (BC, BK), (1, 0),
            )
            tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_build_kernel(
    k,
    v,
    beta,
    g,
    A,
    dA,
    dw,
    du,
    dk,
    dv,
    db,
    dg,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # Ascend UB ~192KB: never materialize [BT, BT] in UB; dA lives in GM (dA tensor).
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # A/dA chunk matrices use (T, BT) layout, same as chunk_scaled_dot_kkt / recompute_w_u_fwd.
    A_base = A + (bos * HV + i_h) * BT
    dA_base = dA + (bos * HV + i_h) * BT
    t_chunk = i_t * BT

    p_b = tl.make_block_ptr(beta + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)

    if USE_G:
        p_g = tl.make_block_ptr(g + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
        b_g_exp = exp2(b_g)
        b_dg = tl.zeros([BT], dtype=tl.float32)

    # phase 1: build raw dA row strips in GM (max tile [BC, BT] in UB)
    for i_ra in range(BT // BC):
        t_ra = i_ra * BC
        p_dA_row = tl.make_block_ptr(
            dA_base, (T, BT), (HV * BT, 1), (t_chunk + t_ra, 0), (BC, BT), (1, 0),
        )
        b_dA_row = tl.zeros([BC, BT], dtype=tl.float32)
        p_A_row = tl.make_block_ptr(
            A_base, (T, BT), (HV * BT, 1), (t_chunk + t_ra, 0), (BC, BT), (1, 0),
        )
        o_col = i_t * BT + tl.arange(0, BT)
        o_row = i_t * BT + t_ra + tl.arange(0, BC)
        m_row = o_row < T
        m_col = o_col < T
        b_A_row = tl.load(p_A_row, boundary_check=(0, 1)).to(tl.float32)
        # A is torch.empty from kkt; only strict-lower entries are written — mask before use.
        b_A_row = tl.where(
            (o_row[:, None] > o_col[None, :]) & (m_row[:, None] & m_col[None, :]),
            b_A_row,
            0.0,
        )
        p_b_row = tl.make_block_ptr(beta + (bos * HV + i_h), (T,), (HV,), (i_t * BT + t_ra,), (BC,), (0,))
        b_b_row = tl.load(p_b_row, boundary_check=(0,)).to(tl.float32)
        p_db_row = tl.make_block_ptr(db + (bos * HV + i_h), (T,), (HV,), (i_t * BT + t_ra,), (BC,), (0,))
        b_db_row = tl.zeros([BC], dtype=tl.float32)
        if USE_G:
            p_g_row = tl.make_block_ptr(g + (bos * HV + i_h), (T,), (HV,), (i_t * BT + t_ra,), (BC,), (0,))
            b_g_row = tl.load(p_g_row, boundary_check=(0,)).to(tl.float32)
            b_g_exp_row = exp2(b_g_row)
            p_dg_row = tl.make_block_ptr(dg + (bos * HV + i_h), (T,), (HV,), (i_t * BT + t_ra,), (BC,), (0,))
            b_dg_row = tl.zeros([BC], dtype=tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(
                k + (bos * H + i_h // (HV // H)) * K, (T, K), (H * K, 1),
                (i_t * BT, i_k * BK), (BT, BK), (1, 0),
            )
            p_k_row = tl.make_block_ptr(
                k + (bos * H + i_h // (HV // H)) * K, (T, K), (H * K, 1),
                (i_t * BT + t_ra, i_k * BK), (BC, BK), (1, 0),
            )
            p_dw = tl.make_block_ptr(
                dw + (bos * HV + i_h) * K, (T, K), (HV * K, 1),
                (i_t * BT, i_k * BK), (BT, BK), (1, 0),
            )
            p_dk = tl.make_block_ptr(
                dk + (bos * HV + i_h) * K, (T, K), (HV * K, 1),
                (i_t * BT + t_ra, i_k * BK), (BC, BK), (1, 0),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            b_k_row = tl.load(p_k_row, boundary_check=(0, 1)).to(tl.float32)
            b_dw = tl.load(p_dw, boundary_check=(0, 1)).to(tl.float32)
            p_dw_row = tl.make_block_ptr(
                dw + (bos * HV + i_h) * K, (T, K), (HV * K, 1),
                (i_t * BT + t_ra, i_k * BK), (BC, BK), (1, 0),
            )
            b_dw_row = tl.load(p_dw_row, boundary_check=(0, 1)).to(tl.float32)
            if USE_G:
                b_kbg = b_k * (b_b * b_g_exp)[:, None]
            else:
                b_kbg = b_k * b_b[:, None]
            b_dA_row += tl.dot(b_dw_row, tl.trans(b_kbg), allow_tf32=False)
            b_dkbg = tl.dot(b_A_row, b_dw, allow_tf32=False)
            if USE_G:
                b_kbg_row = b_k_row * b_b_row[:, None] * b_g_exp_row[:, None]
                b_dk = b_dkbg * (b_g_exp_row * b_b_row)[:, None]
                b_db_row += tl.sum(b_dkbg * b_k_row * b_g_exp_row[:, None], 1)
                b_dg_row += tl.sum(b_dkbg * b_kbg_row, 1)
            else:
                b_dk = b_dkbg * b_b_row[:, None]
                b_db_row += tl.sum(b_dkbg * b_k_row, 1)
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        for i_v in range(tl.cdiv(V, BV)):
            p_v = tl.make_block_ptr(
                v + (bos * HV + i_h) * V, (T, V), (HV * V, 1),
                (i_t * BT, i_v * BV), (BT, BV), (1, 0),
            )
            p_v_row = tl.make_block_ptr(
                v + (bos * HV + i_h) * V, (T, V), (HV * V, 1),
                (i_t * BT + t_ra, i_v * BV), (BC, BV), (1, 0),
            )
            p_du = tl.make_block_ptr(
                du + (bos * HV + i_h) * V, (T, V), (HV * V, 1),
                (i_t * BT + t_ra, i_v * BV), (BC, BV), (1, 0),
            )
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_v_row = tl.load(p_v_row, boundary_check=(0, 1))
            b_du = tl.load(p_du, boundary_check=(0, 1)).to(tl.float32)
            b_vb = (b_v * b_b[:, None]).to(tl.float32)
            b_dA_row += tl.dot(b_du, tl.trans(b_vb), allow_tf32=False)
            p_du_full = tl.make_block_ptr(
                du + (bos * HV + i_h) * V, (T, V), (HV * V, 1),
                (i_t * BT, i_v * BV), (BT, BV), (1, 0),
            )
            b_du_full = tl.load(p_du_full, boundary_check=(0, 1)).to(tl.float32)
            b_dvb = tl.dot(b_A_row, b_du_full, allow_tf32=False)
            b_db_row += tl.sum(b_dvb * b_v_row.to(tl.float32), 1)
            # dv via _compute_wy_dv_host; NPU tl.store to dv is unreliable.

        m_causal = (o_row[:, None] > o_col[None, :]) & (m_row[:, None] & m_col[None, :])
        b_dA_row = tl.where(m_causal, b_dA_row, 0.0)
        tl.store(p_dA_row, b_dA_row, boundary_check=(0, 1))
        tl.store(p_db_row, b_db_row.to(p_db_row.dtype.element_ty), boundary_check=(0,))
        if USE_G:
            tl.store(p_dg_row, b_dg_row.to(p_dg_row.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_sandwich_kernel(
    g,
    A,
    dA,
    cu_seqlens,
    chunk_indices,
    T,
    HV: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A_base = A + (bos * HV + i_h) * BT
    dA_base = dA + (bos * HV + i_h) * BT
    t_chunk = i_t * BT

    for i_r in range(BT // BC):
        for i_c in range(BT // BC):
            t_r = i_r * BC
            t_c = i_c * BC
            o_r = i_t * BT + t_r + tl.arange(0, BC)
            o_c = i_t * BT + t_c + tl.arange(0, BC)
            m_row = o_r < T
            m_col = o_c < T
            m_blk = (o_r[:, None] > o_c[None, :]) & (m_row[:, None] & m_col[None, :])
            acc = tl.zeros([BC, BC], dtype=tl.float32)
            for i_a in range(BT // BC):
                t_a = i_a * BC
                p_A_ir = tl.make_block_ptr(
                    A_base, (T, BT), (HV * BT, 1), (t_chunk + t_r, t_a), (BC, BC), (1, 0),
                )
                b_A_ir = tl.load(p_A_ir, boundary_check=(0, 1)).to(tl.float32)
                temp = tl.zeros([BC, BC], dtype=tl.float32)
                for i_b in range(BT // BC):
                    t_b = i_b * BC
                    p_dM = tl.make_block_ptr(
                        dA_base, (T, BT), (HV * BT, 1), (t_chunk + t_a, t_b), (BC, BC), (1, 0),
                    )
                    p_A_bc = tl.make_block_ptr(
                        A_base, (T, BT), (HV * BT, 1), (t_chunk + t_b, t_c), (BC, BC), (1, 0),
                    )
                    b_dM = tl.load(p_dM, boundary_check=(0, 1)).to(tl.float32)
                    b_A_bc = tl.load(p_A_bc, boundary_check=(0, 1)).to(tl.float32)
                    temp += tl.dot(b_dM, b_A_bc, allow_tf32=False)
                acc += tl.dot(b_A_ir, temp, allow_tf32=False)
            if USE_G:
                p_g_r = tl.make_block_ptr(g + (bos * HV + i_h), (T,), (HV,), (i_t * BT + t_r,), (BC,), (0,))
                p_g_c = tl.make_block_ptr(g + (bos * HV + i_h), (T,), (HV,), (i_t * BT + t_c,), (BC,), (0,))
                b_g_r = tl.load(p_g_r, boundary_check=(0,)).to(tl.float32)
                b_g_c = tl.load(p_g_c, boundary_check=(0,)).to(tl.float32)
                acc = acc * exp2(b_g_r[:, None] - b_g_c[None, :])
            acc = tl.where(m_blk, -acc, 0.0)
            p_dA_out = tl.make_block_ptr(
                dA_base, (T, BT), (HV * BT, 1), (t_chunk + t_r, t_c), (BC, BC), (1, 0),
            )
            tl.store(p_dA_out, acc, boundary_check=(0, 1))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_finish_kernel(
    k,
    beta,
    g,
    A,
    dA,
    dk,
    db,
    dg,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A_base = A + (bos * HV + i_h) * BT
    dA_base = dA + (bos * HV + i_h) * BT
    t_chunk = i_t * BT

    p_b = tl.make_block_ptr(beta + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)

    for i_r in range(BT // BC):
        t_r = i_r * BC
        p_dA_r = tl.make_block_ptr(
            dA_base, (T, BT), (HV * BT, 1), (t_chunk + t_r, 0), (BC, BT), (1, 0),
        )
        b_dA_r = tl.load(p_dA_r, boundary_check=(0, 1)).to(tl.float32)
        p_dA_cols = tl.make_block_ptr(
            dA_base, (T, BT), (HV * BT, 1), (t_chunk, t_r), (BT, BC), (1, 0),
        )
        b_dA_cols = tl.load(p_dA_cols, boundary_check=(0, 1)).to(tl.float32)
        p_b_row = tl.make_block_ptr(beta + (bos * HV + i_h), (T,), (HV,), (i_t * BT + t_r,), (BC,), (0,))
        b_b_row = tl.load(p_b_row, boundary_check=(0,)).to(tl.float32)
        p_db_row = tl.make_block_ptr(db + (bos * HV + i_h), (T,), (HV,), (i_t * BT + t_r,), (BC,), (0,))
        b_db_row = tl.load(p_db_row, boundary_check=(0,)).to(tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(
                k + (bos * H + i_h // (HV // H)) * K, (T, K), (H * K, 1),
                (i_t * BT, i_k * BK), (BT, BK), (1, 0),
            )
            p_k_row = tl.make_block_ptr(
                k + (bos * H + i_h // (HV // H)) * K, (T, K), (H * K, 1),
                (i_t * BT + t_r, i_k * BK), (BC, BK), (1, 0),
            )
            p_dk = tl.make_block_ptr(
                dk + (bos * HV + i_h) * K, (T, K), (HV * K, 1),
                (i_t * BT + t_r, i_k * BK), (BC, BK), (1, 0),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            b_k_row = tl.load(p_k_row, boundary_check=(0, 1)).to(tl.float32)
            b_dkb = tl.dot(b_dA_r, b_k, allow_tf32=False)
            b_db_row += tl.sum(b_dkb * b_k_row, 1)
            b_kb = b_k * b_b[:, None]
            # Match GPU: dk += trans(dot(trans(kb), dA)); dk2[t,b] = sum_s kb[s,b]*dA[s,t]
            b_dk = b_dkb * b_b_row[:, None] + tl.dot(
                tl.trans(b_dA_cols), b_kb.to(tl.float32), allow_tf32=False,
            )
            b_dk += tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_db_row, b_db_row.to(p_db_row.dtype.element_ty), boundary_check=(0,))

    if USE_G:
        for i_r in range(BT // BC):
            for i_c in range(BT // BC):
                t_r = i_r * BC
                t_c = i_c * BC
                p_dA = tl.make_block_ptr(
                    dA_base, (T, BT), (HV * BT, 1), (t_chunk + t_r, t_c), (BC, BC), (1, 0),
                )
                p_A = tl.make_block_ptr(
                    A_base, (T, BT), (HV * BT, 1), (t_chunk + t_r, t_c), (BC, BC), (1, 0),
                )
                p_b_r = tl.make_block_ptr(beta + (bos * HV + i_h), (T,), (HV,), (i_t * BT + t_r,), (BC,), (0,))
                b_dA = tl.load(p_dA, boundary_check=(0, 1)).to(tl.float32)
                b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
                b_b_r = tl.load(p_b_r, boundary_check=(0,)).to(tl.float32)
                b_AdA = b_dA * (b_A * b_b_r[:, None])
                p_dg_r = tl.make_block_ptr(dg + (bos * HV + i_h), (T,), (HV,), (i_t * BT + t_r,), (BC,), (0,))
                p_dg_c = tl.make_block_ptr(dg + (bos * HV + i_h), (T,), (HV,), (i_t * BT + t_c,), (BC,), (0,))
                b_dg_r = tl.load(p_dg_r, boundary_check=(0,)).to(tl.float32)
                b_dg_c = tl.load(p_dg_c, boundary_check=(0,)).to(tl.float32)
                b_dg_r += tl.sum(b_AdA, axis=1)
                b_dg_c -= tl.sum(b_AdA, axis=0)
                tl.store(p_dg_r, b_dg_r.to(p_dg_r.dtype.element_ty), boundary_check=(0,))
                tl.store(p_dg_c, b_dg_c.to(p_dg_c.dtype.element_ty), boundary_check=(0,))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]
    BK = min(NPU_BK, triton.next_power_of_2(K))
    BV = min(NPU_BV, triton.next_power_of_2(V))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = k.new_empty(B, T, HV, K)
    u = torch.empty_like(v)
    recompute_w_u_fwd_kernel[(NT, B*HV)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BC=NPU_BC,
        BK=BK,
        BV=BV,
        num_warps=NPU_NUM_WARPS,
        num_stages=NPU_NUM_STAGES,
    )
    return w, u


def _to_host_dense(x: torch.Tensor) -> torch.Tensor:
    """NPU internal/sparse layouts -> dense float CPU (permute/matmul safe)."""
    if x.device.type == 'npu':
        x = x.clone()
    y = x.detach().float()
    if y.device.type != 'cpu':
        y = y.cpu()
    if y.is_sparse or y.layout != torch.strided:
        y = y.to_dense() if y.is_sparse else torch.tensor(y.numpy(), dtype=torch.float32)
    return y.contiguous()


def _compute_wy_dv_host(
    du: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    T: int,
    BT: int,
) -> torch.Tensor:
    """dv = (A @ du) * beta per chunk; matches GPU prepare_wy_repr_bwd (T, BT) A layout."""
    out_dtype = du.dtype
    device = du.device
    on_npu = device.type == 'npu'
    B, _, HV, V = du.shape
    dv = torch.zeros(B, T, HV, V, dtype=out_dtype, device=device)
    du_w = du.float().contiguous() if not on_npu else None
    beta_w = beta.float().contiguous() if not on_npu else None
    A_w = A.float().contiguous() if not on_npu else None
    for i_t in range(triton.cdiv(T, BT)):
        t0 = i_t * BT
        t1 = min(t0 + BT, T)
        n = t1 - t0
        if n == 0:
            continue
        o = torch.arange(n)
        if on_npu:
            du_c = _to_host_dense(du[:, t0:t1]).permute(0, 2, 1, 3)
            # beta is [B, T, HV] (3D); do not use 4D permute like du.
            beta_c = _to_host_dense(beta[:, t0:t1]).permute(0, 2, 1).unsqueeze(-1)
            a = _to_host_dense(A[:, t0:t1, :, :n]).permute(0, 2, 1, 3)
        else:
            du_c = du_w[:, t0:t1, :, :].permute(0, 2, 1, 3)
            beta_c = beta_w[:, t0:t1, :].permute(0, 2, 1).unsqueeze(-1)
            a = A_w[:, t0:t1, :, :n].permute(0, 2, 1, 3)
        mask = (o[:, None] > o[None, :]).to(du_c.dtype)
        a = a * mask
        dvb = torch.matmul(a, du_c)
        b = beta_c
        chunk = (dvb * b).permute(0, 2, 1, 3).to(out_dtype)
        if on_npu:
            dv[:, t0:t1].copy_(chunk.to(device=device, non_blocking=True))
        else:
            dv[:, t0:t1] = chunk
    return dv


def _apply_wy_da_sandwich_host(
    dA: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None,
    T: int,
    BT: int,
) -> None:
    """Sandwich dA on CPU per chunk; avoids NPU prepare_wy_repr_bwd_sandwich_kernel timeout."""
    B, _, HV, _ = dA.shape
    on_npu = dA.device.type == 'npu'
    o = torch.arange(BT)
    strict_lower = o[:, None] > o[None, :]
    for i_t in range(triton.cdiv(T, BT)):
        t0 = i_t * BT
        t1 = min(t0 + BT, T)
        n = t1 - t0
        if n == 0:
            continue
        m_bool = strict_lower[:n, :n]
        if on_npu:
            da = _to_host_dense(dA[:, t0:t1, :, :n]).permute(0, 2, 1, 3)
            a = _to_host_dense(A[:, t0:t1, :, :n]).permute(0, 2, 1, 3)
            g_chunk = _to_host_dense(g[:, t0:t1, :]) if g is not None else None
        else:
            da = dA[:, t0:t1, :, :n].float().permute(0, 2, 1, 3)
            a = A[:, t0:t1, :, :n].float().permute(0, 2, 1, 3)
            g_chunk = g[:, t0:t1, :].float() if g is not None else None
        da = da * m_bool.view(1, 1, n, n).to(da.dtype)
        da = da.reshape(B * HV, n, n)
        a = a.reshape(B * HV, n, n)
        da = torch.bmm(da, a)
        da = torch.bmm(a, da)
        if g_chunk is not None:
            gb = g_chunk.permute(0, 2, 1).reshape(B * HV, n)
            if n < BT:
                gb = torch.nn.functional.pad(gb, (0, BT - n))
            gdiff = torch.exp2(gb[:, :, None] - gb[:, None, :])
            da = da * gdiff[:, :n, :n]
        da = torch.where(m_bool.unsqueeze(0), -da[:, :n, :n], torch.zeros_like(da[:, :n, :n]))
        da_phys = da.reshape(B, HV, n, n).permute(0, 2, 1, 3).to(dA.dtype)
        if on_npu:
            dA[:, t0:t1, :, :n].copy_(da_phys.to(device=dA.device, non_blocking=True))
        else:
            dA[:, t0:t1, :, :n] = da_phys


def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]
    assert BT % NPU_BC == 0, f'chunk size {BT} must be divisible by NPU_BC {NPU_BC}'
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(NPU_BK, triton.next_power_of_2(K))
    BV = min(NPU_BV, triton.next_power_of_2(V))

    dk = torch.zeros(B, T, HV, K, dtype=k.dtype, device=k.device)
    dv = torch.zeros_like(v)
    dg = torch.zeros_like(g) if g is not None else None
    db = torch.zeros_like(beta)
    dA = torch.zeros_like(A, dtype=torch.float32)
    grid = (NT, B * HV)
    launch = dict(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        HV=HV,
        BT=BT,
        BC=NPU_BC,
        num_warps=NPU_NUM_WARPS,
        num_stages=NPU_NUM_STAGES,
    )
    prepare_wy_repr_bwd_build_kernel[grid](
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dA=dA,
        dw=dw,
        du=du,
        dk=dk,
        dv=dv,
        db=db,
        dg=dg,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        **launch,
    )
    dv = _compute_wy_dv_host(du=du, beta=beta, A=A, T=T, BT=BT)
    _apply_wy_da_sandwich_host(dA=dA, A=A, g=g, T=T, BT=BT)
    prepare_wy_repr_bwd_finish_kernel[grid](
        k=k,
        beta=beta,
        g=g,
        A=A,
        dA=dA,
        dk=dk,
        db=db,
        dg=dg,
        H=H,
        K=K,
        BK=BK,
        **launch,
    )
    if H != HV:
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    return dk, dv, db, dg


fwd_recompute_w_u = recompute_w_u_fwd
bwd_prepare_wy_repr = prepare_wy_repr_bwd

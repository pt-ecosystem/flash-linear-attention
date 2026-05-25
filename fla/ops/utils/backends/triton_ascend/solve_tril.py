# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import IS_TF32_SUPPORTED, input_guard

NPU_NUM_WARPS = 2
NPU_NUM_STAGES = 1
NPU_BC = 16

if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32')
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def solve_tril_diag16_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Invert one 16x16 diagonal block of (I + A) using static unrolled forward substitution."""
    i_t, i_bh, i_blk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_row = i_t * BT + i_blk * BC
    if i_row >= T:
        return

    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    o_i = tl.arange(0, BC)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    col_off = i_blk * BC

    p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_row, col_off), (BC, BC), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    b_Ai = -tl.where(m_A, b_A, 0.)

    row_end = T - i_row
    for i in tl.static_range(2, BC):
        b_a = tl.sum(tl.where((o_i == i)[:, None], -b_A, 0.), 0)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a = b_a + tl.sum(b_a[:, None] * b_Ai, 0)
        b_row = tl.where((o_i == i)[:, None], b_a, b_Ai)
        b_Ai = tl.where(i < row_end, b_row, b_Ai)
    b_Ai += m_I

    p_Ai = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_row, col_off), (BC, BC), (1, 0))
    tl.store(p_Ai, b_Ai.to(Ai.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def solve_tril_merge64_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Merge four solved 16x16 diagonal blocks into full 64x64 (I+A)^{-1} off-diagonal blocks."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    chunk_rem = T - i_t * BT
    # Partial tail chunk: only diag16 runs for the first ceil(chunk_rem/BC) blocks.
    # Off-diagonal merge must be skipped or it reads uninitialized Ai from skipped diags.
    if chunk_rem <= BC:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    p_Ai00 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Ai11 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_Ai22 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0))
    p_Ai33 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Ai00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Ai11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai22 = tl.load(p_Ai22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai33 = tl.load(p_Ai33, boundary_check=(0, 1)).to(tl.float32)

    p_A10 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_A20 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_A21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_A30 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_A31 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_A32 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
    b_A10 = tl.load(p_A10, boundary_check=(0, 1)).to(tl.float32)
    b_A20 = tl.load(p_A20, boundary_check=(0, 1)).to(tl.float32)
    b_A21 = tl.load(p_A21, boundary_check=(0, 1)).to(tl.float32)
    b_A30 = tl.load(p_A30, boundary_check=(0, 1)).to(tl.float32)
    b_A31 = tl.load(p_A31, boundary_check=(0, 1)).to(tl.float32)
    b_A32 = tl.load(p_A32, boundary_check=(0, 1)).to(tl.float32)

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_A10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_A21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_A32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_A20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_A21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_A32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_A30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_A31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_A32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    out_ty = Ai.dtype.element_ty
    p_Ai10 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Ai20 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Ai21 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_Ai30 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_Ai31 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_Ai32 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
    tl.store(p_Ai10, b_Ai10.to(out_ty), boundary_check=(0, 1))
    tl.store(p_Ai20, b_Ai20.to(out_ty), boundary_check=(0, 1))
    tl.store(p_Ai21, b_Ai21.to(out_ty), boundary_check=(0, 1))
    tl.store(p_Ai30, b_Ai30.to(out_ty), boundary_check=(0, 1))
    tl.store(p_Ai31, b_Ai31.to(out_ty), boundary_check=(0, 1))
    tl.store(p_Ai32, b_Ai32.to(out_ty), boundary_check=(0, 1))


@input_guard
def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert A.shape[-1] == 64, 'triton_ascend solve_tril currently supports BT=64 only'
    if output_dtype is None:
        output_dtype = A.dtype

    B, T, H, BT = A.shape
    BC = NPU_BC
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

    # Must be zero-initialized: skipped diag tiles leave Ai undefined with torch.empty.
    Ai = torch.zeros_like(A, dtype=output_dtype)
    kw = dict(num_warps=NPU_NUM_WARPS, num_stages=NPU_NUM_STAGES)
    solve_tril_diag16_kernel[(NT, B * H, 4)](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        BC=BC,
        **kw,
    )
    solve_tril_merge64_kernel[(NT, B * H)](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        BC=BC,
        **kw,
    )
    return Ai

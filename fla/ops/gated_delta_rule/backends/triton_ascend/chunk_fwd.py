# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.common.backends.triton_ascend.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.utils.backends.triton_ascend.solve_tril import solve_tril
from fla.ops.gated_delta_rule.backends.triton_ascend.wy_fast import recompute_w_u_fwd
from fla.ops.utils import prepare_chunk_indices


def chunk_gated_delta_rule_fwd_intra(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    GDN intra-chunk forward on Ascend: kkt + solve_tril + recompute_w_u (split kernels).

    The fused kkt+solve kernel used on GPU triggers Ascend MLIR errors (mmadL1 / vmul root
    alloc). Two separate kernels match the default GPU semantics with NPU-fixed launches.
    """
    B, T, H, K, HV = *k.shape, beta.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=k.dtype,
        chunk_indices=chunk_indices,
    )
    # Triton-written tensors may use NPU internal format; copy before solve_tril.
    A_work = torch.empty_like(A)
    A_work.copy_(A)
    A = solve_tril(
        A=A_work,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=k.dtype,
    )
    A_solve = torch.empty_like(A)
    A_solve.copy_(A)

    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A_solve,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, A_solve

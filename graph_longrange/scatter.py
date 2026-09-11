"""Scatter reductions over graph indices.

Local implementation of the ``scatter_sum`` / ``scatter_mean`` operations
(the same semantics as ``mace.tools.scatter``, which in turn follows
``torch_scatter``), so that graph_longrange does not depend on mace.
"""


import torch


def _broadcast_index(
    src: torch.Tensor, index: torch.Tensor, dim: int
) -> torch.Tensor:
    if dim < 0:
        dim = src.dim() + dim
    if index.dim() == 1:
        for _ in range(dim):
            index = index.unsqueeze(0)
    while index.dim() < src.dim():
        index = index.unsqueeze(-1)
    return index.expand_as(src)


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    expanded_index = _broadcast_index(src, index, dim)
    if out is None:
        output_size = list(src.size())
        if dim_size is not None:
            output_size[dim] = dim_size
        elif index.numel() == 0:
            output_size[dim] = 0
        else:
            output_size[dim] = int(index.max()) + 1
        out = torch.zeros(output_size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, expanded_index, src)


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    summed = scatter_sum(src, index, dim=dim, out=out, dim_size=dim_size)
    resolved_dim_size = summed.size(dim if dim >= 0 else summed.dim() + dim)

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    counts = scatter_sum(ones, index, dim=0, dim_size=resolved_dim_size)
    counts = counts.clamp_min(1)

    count_shape = [1] * summed.dim()
    count_shape[dim if dim >= 0 else summed.dim() + dim] = resolved_dim_size
    return summed / counts.view(count_shape)

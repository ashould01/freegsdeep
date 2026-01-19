import torch
from freegsdeep.typing import *
    
def romberg(y: Tensor, dx=1.0, axis=-1) -> Tensor:
    def _tupleset(t: Tuple, d: int, v: Any) -> Tuple:
        l = list(t)
        l[d] = v
        return tuple(l)
    
    nd = len(y.shape)
    axis = axis % nd
    Nsamps = y.shape[axis]
    Ninterv = Nsamps - 1
    n = 1
    k = 0
    while n < Ninterv:
        n <<= 1
        k += 1
    if n != Ninterv:
        raise ValueError(f"Number of samples must be 2^k + 1; got at least {n} > {Ninterv}")
    # R = torch.zeros((k + 1, k + 1), dtype=torch.float64, device=y.device)
    R = torch.zeros(
        y.shape[:axis] + (k + 1, k + 1) + y.shape[axis + 1:],
        dtype=torch.float64, device=y.device
        )
    h = Ninterv * dx
    slice_all = (slice(None), ) * nd
    slice0 = _tupleset(slice_all, axis, 0)
    slicem1 = _tupleset(slice_all, axis, -1)
    slice_large = (slice(None), ) * (nd + 1)
    slice_large = _tupleset(_tupleset(slice_large, axis, 0), axis + 1, 0)
    R[slice_large] = (y[slice0] + y[slicem1]) * h / 2.0
    slice_R = slice_all
    start = stop = step = Ninterv
    for i in range(1, k + 1):
        start >>= 1
        slice_R = _tupleset(slice_R, axis, slice(start, stop, step))
        step >>= 1
        slice_large_1 = _tupleset(slice_large, axis, i)
        slice_large_2 = _tupleset(slice_large, axis, i - 1)
        R[slice_large_1] = 0.5 * (R[slice_large_2] + h * y[slice_R].sum(axis))
        for j in range(1, i + 1):
            slice_large_j1 = _tupleset(slice_large_1, axis + 1, j)
            slice_large_j2 = _tupleset(slice_large_1, axis + 1, j - 1)
            slice_large_j3 = _tupleset(slice_large_2, axis + 1, j - 1)
            prev = R[slice_large_j2]
            R[slice_large_j1] = prev + (prev - R[slice_large_j3]) / \
                ((1 << (2 * j)) - 1)
        h /= 2.0
        
    return R[slice_large_j1]
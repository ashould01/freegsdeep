import torch
import torch.nn.functional as F
from freegsdeep.typing import *

def _central_diff_1d_nonuniform(x: Tensor, f: Tensor, dim: int) -> Tensor:
    f_perm = f.movedim(dim, -1)
    x = x.to(f_perm.device).to(f_perm.dtype)

    df = torch.empty_like(f_perm)

    denom = (x[2:] - x[:-2])
    df[..., 1:-1] = (f_perm[..., 2:] - f_perm[..., :-2]) / denom

    df[..., 0] = (f_perm[..., 1] - f_perm[..., 0]) / (x[1] - x[0])
    df[..., -1] = (f_perm[..., -1] - f_perm[..., -2]) / (x[-1] - x[-2])

    return df.movedim(-1, dim)

class TorchBicubicSpline2D:
    """
    Bicubic spline-like interpolator on a rect grid with analytic derivatives.
    No autograd, no grid_sample.

    z shape: (Nx, Ny), where x_grid aligns with axis 0 and y_grid aligns with axis 1.
    """

    def __init__(self, x_grid: Tensor, y_grid: Tensor, z: Tensor):
        self.x = torch.as_tensor(x_grid)
        self.y = torch.as_tensor(y_grid)
        self.z = torch.as_tensor(z)

        self.device = self.z.device
        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device)

        self.M = torch.tensor(
            [[ 1.,  0.,  0.,  0.],
            [ 0.,  0.,  1.,  0.],
            [-3.,  3., -2., -1.],
            [ 2., -2.,  1.,  1.]],
            dtype=torch.float64, device=self.device
        )

        self.fx = _central_diff_1d_nonuniform(self.x, self.z, dim=0)
        self.fy = _central_diff_1d_nonuniform(self.y, self.z, dim=1)
        self.fxy = _central_diff_1d_nonuniform(self.y, self.fx, dim=1)

        self._build_cell_coeffs()

    def _build_cell_coeffs(self):
        dx = (self.x[1:] - self.x[:-1])  # (Nx-1,)
        dy = (self.y[1:] - self.y[:-1])  # (Ny-1,)

        # Corner values for each cell (Nx-1, Ny-1)
        f00 = self.z[:-1, :-1]
        f10 = self.z[1:,  :-1]
        f01 = self.z[:-1, 1:]
        f11 = self.z[1:,  1:]

        fx00 = self.fx[:-1, :-1]
        fx10 = self.fx[1:,  :-1]
        fx01 = self.fx[:-1, 1:]
        fx11 = self.fx[1:,  1:]

        fy00 = self.fy[:-1, :-1]
        fy10 = self.fy[1:,  :-1]
        fy01 = self.fy[:-1, 1:]
        fy11 = self.fy[1:,  1:]

        fxy00 = self.fxy[:-1, :-1]
        fxy10 = self.fxy[1:,  :-1]
        fxy01 = self.fxy[:-1, 1:]
        fxy11 = self.fxy[1:,  1:]

        # Scale derivatives to normalized (u,v) coordinates:
        # u = (x-xi)/dx_i, v = (y-yj)/dy_j
        # so: df/du = fx * dx, df/dv = fy * dy, d2f/dudv = fxy * dx * dy
        DX = dx[:, None]       # (Nx-1,1)
        DY = dy[None, :]       # (1,Ny-1)

        # Build P matrices for each cell:
        # P = [[f00, f01, fy00*dy, fy01*dy],
        #      [f10, f11, fy10*dy, fy11*dy],
        #      [fx00*dx, fx01*dx, fxy00*dx*dy, fxy01*dx*dy],
        #      [fx10*dx, fx11*dx, fxy10*dx*dy, fxy11*dx*dy]]
        P = torch.empty((
            self.z.shape[0]-1, self.z.shape[1]-1, 4, 4
            ), dtype=torch.float64, device=self.device)

        P[..., 0, 0] = f00
        P[..., 0, 1] = f01
        P[..., 1, 0] = f10
        P[..., 1, 1] = f11

        P[..., 0, 2] = fy00 * DY
        P[..., 0, 3] = fy01 * DY
        P[..., 1, 2] = fy10 * DY
        P[..., 1, 3] = fy11 * DY

        P[..., 2, 0] = fx00 * DX
        P[..., 2, 1] = fx01 * DX
        P[..., 3, 0] = fx10 * DX
        P[..., 3, 1] = fx11 * DX

        P[..., 2, 2] = fxy00 * (DX * DY)
        P[..., 2, 3] = fxy01 * (DX * DY)
        P[..., 3, 2] = fxy10 * (DX * DY)
        P[..., 3, 3] = fxy11 * (DX * DY)

        # A = M * P * M^T for each cell
        MT = self.M.t()
        self.A = self.M @ P @ MT  # (Nx-1,Ny-1,4,4)
        self.dx = dx
        self.dy = dy

    def _find_cell_indices(self, xq, yq):
        # clamp to domain (like "border" extrap)
        xq = torch.clamp(xq, self.x[0], self.x[-1])
        yq = torch.clamp(yq, self.y[0], self.y[-1])

        # indices i such that x[i] <= xq < x[i+1]
        i = torch.searchsorted(self.x, xq, right=True) - 1
        j = torch.searchsorted(self.y, yq, right=True) - 1
        i = torch.clamp(i, 0, self.x.numel() - 2)
        j = torch.clamp(j, 0, self.y.numel() - 2)
        return i, j, xq, yq

    @staticmethod
    def _polyvec(u, order=0):
        # returns [1,u,u^2,u^3] or its derivatives w.r.t u
        if order == 0:
            return torch.stack([torch.ones_like(u), u, u*u, u*u*u], dim=-1)
        if order == 1:
            return torch.stack([torch.zeros_like(u), torch.ones_like(u), 2*u, 3*u*u], dim=-1)
        if order == 2:
            return torch.stack([torch.zeros_like(u), torch.zeros_like(u), 2*torch.ones_like(u), 6*u], dim=-1)
        raise ValueError("order must be 0,1,2")

    def __call__(self, xq, yq, dx=0, dy=0, grid=False):
        xq = torch.as_tensor(xq, dtype=torch.float64, device=self.device)
        yq = torch.as_tensor(yq, dtype=torch.float64, device=self.device)

        if grid:
            if xq.ndim != 1 or yq.ndim != 1:
                raise ValueError("grid=True expects 1D xq and 1D yq.")
            X = xq[:, None].expand(xq.numel(), yq.numel())
            Y = yq[None, :].expand(xq.numel(), yq.numel())
        else:
            X, Y = torch.broadcast_tensors(xq, yq)

        i, j, Xc, Yc = self._find_cell_indices(X, Y)

        # normalized coordinates u,v in [0,1]
        x0 = self.x[i]
        y0 = self.y[j]
        du = self.dx[i]
        dv = self.dy[j]
        u = (Xc - x0) / du
        v = (Yc - y0) / dv

        # pick coefficients A for each query
        Aij = self.A[i, j]  # (...,4,4)

        pu = self._polyvec(u, order=dx)   # (...,4)
        pv = self._polyvec(v, order=dy)   # (...,4)

        # f_uv = pu^T * Aij * pv
        tmp = torch.matmul(Aij, pv.unsqueeze(-1)).squeeze(-1)     # (...,4)
        out = (pu * tmp).sum(dim=-1)                              # (...)

        # chain rule from (u,v) to (x,y)
        if dx == 1: out = out / du
        if dx == 2: out = out / (du * du)
        if dy == 1: out = out / dv
        if dy == 2: out = out / (dv * dv)

        return out
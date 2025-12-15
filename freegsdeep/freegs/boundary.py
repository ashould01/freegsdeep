from multiprocessing.dummy import Array
from freegsdeep.freegs.gradshafranov import Greens
import torch
import torch.nn as nn
import numpy as np
from freegsdeep.freegs.utils import romberg as romb
from freegsdeep.utilstyping import Tensor, Tuple

class freeBoundaryHagenow():
    def __init__(
        self, R: Tensor, Z: Tensor, nx: int, ny: int,
        dR: float, dZ: float, device: str, solver: nn.Module
        ):
        self.R, self.Z = R, Z
        self.nx, self.ny = nx, ny
        self.dR, self.dZ = dR, dZ
        self.dd = np.sqrt(dR ** 2 + dZ ** 2)
        self.coeffs = torch.tensor([[
            25.0 / 12, -4.0, 3.0, -4.0 / 3, 1.0 / 4
        ]], dtype=torch.float64, device=device)
        self.device = device
        self.solver = solver
        eps = 1e-2
        self.bndry_indices = np.concatenate([
            [(0, y, -eps, 0.0) for y in range(self.ny)], 
            [(x, 0, 0.0, -eps) for x in range(self.nx)], 
            [(self.nx - 1, y, eps, 0.0) for y in range(self.ny)],
            [(x, self.ny - 1, 0.0, eps) for x in range(self.nx)], 
        ])

    def __call__(self, R_times_Jtor: Tensor, zero_bdry: Tensor) -> Tensor:
        
        scaling_rhs = R_times_Jtor.max() - R_times_Jtor.min()
        psi_fixed = self.solver(
            self.R.reshape(1, -1, 1), self.Z.reshape(1, -1, 1),
            R_times_Jtor / scaling_rhs
            ) * scaling_rhs
        psi_fixed *= zero_bdry
        psi_fixed = psi_fixed.reshape(self.nx, self.ny)
        dUdn_L = torch.sum(self.coeffs * psi_fixed[:5, :].T, dim=1)
        dUdn_D = torch.sum(self.coeffs * psi_fixed[:, :5], dim=1)
        dUdn_R = torch.sum(self.coeffs * psi_fixed[-5:, :].T, dim=1)
        dUdn_U = torch.sum(self.coeffs * psi_fixed[:, :5], dim=1)
        
        
        dUdn_L[0] = dUdn_D[0] = torch.sum(self.coeffs * torch.diag(
            psi_fixed[:5, :5]
            )) / self.dd
        dUdn_L[-1] = dUdn_U[0] = torch.sum(self.coeffs * torch.diag(
            psi_fixed[:5, np.arange(-1, -6, -1)]
            )) / self.dd
        dUdn_R[0] = dUdn_D[-1] = torch.sum(self.coeffs * torch.diag(
            psi_fixed[np.arange(-1, -6, -1), :5]
            )) / self.dd
        dUdn_R[-1] = dUdn_U[-1] = torch.sum(self.coeffs * torch.diag(
            psi_fixed[np.arange(-1, -6, -1), np.arange(-1, -6, -1)]
            )) / self.dd
        psi_bndry = torch.zeros(
            (1, 2 * self.nx + 2 * self.ny, 1),
            dtype=torch.float64, device=self.device
            )
        for idx, (x, y, Reps, Zeps) in enumerate(self.bndry_indices):
            x = int(round(x))
            y = int(round(y))
            Rpos = self.R[x, y] + Reps
            Zpos = self.Z[x, y] + Zeps
            greenfunc = Greens(self.R[0, :], self.Z[0, :], Rpos, Zpos)
            result = romb(greenfunc * dUdn_L / self.R[0, :]) * self.dZ
            greenfunc = Greens(self.R[-1, :], self.Z[-1, :], Rpos, Zpos)
            result += romb(greenfunc * dUdn_R / self.R[-1, :]) * self.dZ
            greenfunc = Greens(self.R[:, 0], self.Z[:, 0], Rpos, Zpos)
            result += romb(greenfunc * dUdn_D / self.R[:, 0]) * self.dR
            greenfunc = Greens(self.R[:, -1], self.Z[:, -1], Rpos, Zpos)
            result += romb(greenfunc * dUdn_U / self.R[:, -1]) * self.dR
            psi_bndry[0, idx, 0] = result
        
        return psi_bndry
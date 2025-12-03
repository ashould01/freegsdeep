from multiprocessing.dummy import Array
from freegs.freegs.gradshafranov import Greens
import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import romb
from freegsdeep.utilstyping import Tensor, Tuple

class freeBoundaryHagenow():
    def __init__(
        self, R: Tensor, Z: Tensor, R_cpu: Array, Z_cpu: Array,
        nx: int, ny: int, dR: float, dZ: float, solver: nn.Module
        ):
        self.R, self.Z, self.R_cpu, self.Z_cpu = R, Z, R_cpu, Z_cpu
        self.nx, self.ny = nx, ny
        self.dR, self.dZ = dR, dZ
        self.dd = np.sqrt(dR ** 2 + dZ ** 2)
        self.coeffs = [
            (0, 25.0 / 12), (1, -4.0), (2, 3.0), (3, -4.0 / 3), (4, 1.0 / 4)
        ]
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
        psi_fixed = self.solver(self.R, self.Z, R_times_Jtor / scaling_rhs) * scaling_rhs
        psi_fixed *= zero_bdry
        psi_fixed = psi_fixed.reshape(self.nx, self.ny).detach().cpu().numpy()
        dUdn_L = sum(
            [weight * psi_fixed[index, :] for index, weight in self.coeffs]
            ) / self.dR
        dUdn_D = sum(
            [weight * psi_fixed[:, index] for index, weight in self.coeffs]
            ) / self.dZ
        dUdn_R = sum(
            [weight * psi_fixed[-(1 + index), :] for index, weight in self.coeffs]
            ) / self.dR
        dUdn_U = sum(
            [weight * psi_fixed[:, -(1 + index)] for index, weight in self.coeffs]
            ) / self.dZ
        
        dUdn_L[0] = dUdn_D[0] = sum(
            [weight * psi_fixed[index, index] for index, weight in self.coeffs]
            ) / self.dd
        dUdn_L[-1] = dUdn_U[0] = sum(
            [weight * psi_fixed[index, -(1 + index)] for index, weight in self.coeffs]
            ) / self. dd
        dUdn_R[0] = dUdn_D[-1] = sum(
            [weight * psi_fixed[-(1 + index), index] for index, weight in self.coeffs]
            ) / self.dd
        dUdn_R[-1] = dUdn_U[-1] = sum(
            [weight * psi_fixed[-(1 + index), -(1 + index)] for index, weight in self.coeffs]
            ) / self.dd

        psi_bndry = np.zeros(
            (1, 2 * self.nx + 2 * self.ny),
            dtype=np.float64
            )

        for idx, (x, y, Reps, Zeps) in enumerate(self.bndry_indices):
            x = int(round(x))
            y = int(round(y))
            Rpos = self.R_cpu[x, y] + Reps
            Zpos = self.Z_cpu[x, y] + Zeps
            greenfunc = Greens(self.R_cpu[0, :], self.Z_cpu[0, :], Rpos, Zpos)
            result = romb(greenfunc * dUdn_L / self.R_cpu[0, :]) * self.dZ
            greenfunc = Greens(self.R_cpu[-1, :], self.Z_cpu[-1, :], Rpos, Zpos)
            result += romb(greenfunc * dUdn_R / self.R_cpu[-1, :]) * self.dZ
            greenfunc = Greens(self.R_cpu[:, 0], self.Z_cpu[:, 0], Rpos, Zpos)
            result += romb(greenfunc * dUdn_D / self.R_cpu[:, 0]) * self.dR
            greenfunc = Greens(self.R_cpu[:, -1], self.Z_cpu[:, -1], Rpos, Zpos)
            result += romb(greenfunc * dUdn_U / self.R_cpu[:, -1]) * self.dR
            
            psi_bndry[0, idx] = result
        
        return psi_bndry
            
            
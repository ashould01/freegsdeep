from multiprocessing.dummy import Array
from freegsdeep.freegs.gradshafranov import Greens
import torch
import torch.nn as nn
import numpy as np
from freegsdeep.freegs.utils import romberg as romb
# from scipy.integrate import romb
from freegsdeep.typing import Tensor, Tuple

class freeBoundaryHagenow():
    def __init__(
        self, R: Tensor, Z: Tensor, bdry_pt: Tensor, nx: int, ny: int,
        dR: float, dZ: float, device: str, solver: nn.Module
        ):
        self.R, self.Z = R, Z
        self.nx, self.ny = nx, ny
        self.dR, self.dZ = dR, dZ
        self.bdry_pt = bdry_pt
        self.dd = torch.sqrt(torch.tensor(dR ** 2 + dZ ** 2, dtype=torch.float64, device=device))
        self.coeffs = torch.tensor([[
            25.0 / 12, -4.0, 3.0, -4.0 / 3, 1.0 / 4
        ]], dtype=torch.float64, device=device).reshape(1, 5, 1)
        self.device = device
        self.solver = solver
        eps = 1e-2
        self.bndry_indices = torch.tensor([
            [[-eps, 0.0] for _ in range(self.ny)], 
            [[0.0, -eps] for _ in range(self.nx)], 
            [[eps, 0.0] for _ in range(self.ny)],
            [[0.0, eps] for _ in range(self.nx)], 
        ], dtype=torch.float64, device=device).reshape(-1, 2)
        self.multiplier = torch.tensor([
            [(dR + dZ) / 2] + [dZ for _ in range(self.ny - 1)] \
            + [(dR + dZ) / 2] + [dR for _ in range(self.nx - 1)] \
            + [(dR + dZ) / 2] + [dZ for _ in range(self.ny - 1)] \
            + [(dR + dZ) / 2] + [dR for _ in range(self.nx - 1)]
        ], dtype=torch.float64, device=device).reshape(-1, 1)

    def __call__(self, R_times_Jtor: Tensor, zero_bdry: Tensor) -> Tensor:
        
        scaling_rhs = R_times_Jtor.max() - R_times_Jtor.min()
        with torch.no_grad():
            psi_fixed = self.solver.forward(
                self.R.reshape(1, -1, 1), self.Z.reshape(1, -1, 1),
                R_times_Jtor / scaling_rhs
                ) * scaling_rhs
        psi_fixed *= zero_bdry
        psi_fixed = psi_fixed.reshape(self.nx, self.ny)

        psi_fixed_batch_1 = torch.stack([
            psi_fixed[:5, :].T / self.dR, torch.flip(psi_fixed[-5:, :], dims=[0]).T / self.dR,
            psi_fixed[:, :5] / self.dZ, torch.flip(psi_fixed[:, -5:], dims=[1]) / self.dZ
        ], axis=2)
        dUdn = torch.sum(self.coeffs * psi_fixed_batch_1, axis=1)
        psi_fixed_batch_2 = torch.stack([
            torch.diag(psi_fixed[:5, :5]), 
            torch.diag(torch.flip(psi_fixed[:5, -5:], dims=[1])),
            torch.diag(torch.flip(psi_fixed[-5:, :5], dims=[0])),
            torch.diag(torch.flip(torch.flip(psi_fixed[-5:, -5:], dims=[0]), dims=[1]))
        ])
        corner_sum = torch.sum(
            self.coeffs.squeeze(2) * psi_fixed_batch_2 / self.dd, axis=1
            )
        dUdn[0, 0] = dUdn[0, 2] = corner_sum[0]
        dUdn[-1, 0] = dUdn[0, 3] = corner_sum[1]
        dUdn[0, 1] = dUdn[-1, 2] = corner_sum[2]
        dUdn[-1, 1] = dUdn[-1, 3] = corner_sum[3]
        
        Xpos = self.bdry_pt + self.bndry_indices
        Rpos = Xpos[..., 0]
        Zpos = Xpos[..., 1]
        greenfunc = Greens(
            self.bdry_pt[None, :, 0], self.bdry_pt[None, :, 1],
            Rpos[:, None], Zpos[:, None]
            )
        result = romb(
            greenfunc[:self.ny] * dUdn[:, 0:1] / \
                self.R[0, :][:, None], axis=0
            ) * self.dZ
        result += romb(
            greenfunc[self.ny:2*self.ny] * dUdn[:, 1:2] / \
                self.R[-1, :][:, None],
            axis=0
            ) * self.dZ
        result += romb(
            greenfunc[2*self.ny:2*self.ny+self.nx] * dUdn[:, 2:3] / \
                self.R[:, 0][:, None],
            axis=0
            ) * self.dR
        result += romb(
            greenfunc[2*self.ny+self.nx:] * dUdn[:, 3:4] / self.R[:, -1][:, None],
            axis=0
            ) * self.dR
        # result = torch.sum(
        #     greenfunc * dUdn.reshape(-1, 1) / self.bdry_pt[:, 0:1] * self.multiplier,
        #     dim=0)
        # result = torch.trapz(
        #     greenfunc[:self.ny] * dUdn[:, 0:1] / \
        #         self.R[:, 0][:, None], dim=0
        #     ) * self.dZ
        # result += torch.trapz(
        #     greenfunc[self.ny:2*self.ny] * dUdn[:, 1:2] / \
        #         self.R[-1, :][:, None], dim=0
        #     ) * self.dZ
        # result += torch.trapz(
        #     greenfunc[2*self.ny:2*self.ny+self.nx] * dUdn[:, 2:3] / \
        #         self.R[:, 0][:, None], dim=0
        #     ) * self.dR 
        # result += torch.trapz(
        #     greenfunc[2*self.ny+self.nx:] * dUdn[:, 3:4] / \
        #         self.R[:, -1][:, None], dim=0
        #     ) * self.dR
        return psi_fixed, result.reshape(1, -1, 1)
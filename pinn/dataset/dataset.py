import torch
import torch.nn as nn
import numpy as np
import freegs.freegs as freegs
from freegsdeep.typing import *
from torch.utils.data import Dataset
from scipy.integrate import romb

class SolovevDataset(Dataset):
    
    def __init__(
        self, Rmin: float, Rmax: float, 
        Zmin: float, Zmax: float, num_resi: int, num_bdry: int,
        kappa0: float = 1.5, q0: float = 1.5,
        ) -> None:

        self.num = num_resi
        _R_resi = torch.from_numpy(np.linspace(Rmin, Rmax, num_resi+1))
        _Z_resi = torch.from_numpy(np.linspace(Zmin, Zmax, num_resi+1))
        _R_resi = torch.from_numpy(np.random.uniform(_R_resi[:-1], _R_resi[1:]))
        _Z_resi = torch.from_numpy(np.random.uniform(_Z_resi[:-1], _Z_resi[1:]))
        _R_bdry = torch.from_numpy(np.linspace(Rmin, Rmax, num_bdry+1))
        _Z_bdry = torch.from_numpy(np.linspace(Zmin, Zmax, num_bdry+1))
        _R_bdry = torch.from_numpy(np.random.uniform(_R_bdry[:-1], _R_bdry[1:]))
        _Z_bdry = torch.from_numpy(np.random.uniform(_Z_bdry[:-1], _Z_bdry[1:]))

        perm = np.random.permutation(num_resi)
        self.residual = torch.tensor(
            [[_R_resi[i], _Z_resi[perm[i]]] for i in range(num_resi)]
            )
        self.boundary_L = torch.tensor(
            [[Rmin, _Z_bdry[i]] for i in range(num_bdry)]
        )
        self.boundary_D = torch.tensor(
            [[_R_bdry[i], Zmin] for i in range(num_bdry)]
        )
        self.boundary_R = torch.tensor(
            [[Rmax, _Z_bdry[i]] for i in range(num_bdry)]
        )
        self.boundary_U = torch.tensor(
            [[_R_bdry[i], Zmax] for i in range(num_bdry)]
        )
        self.boundary = torch.concatenate([
            self.boundary_L, self.boundary_D, self.boundary_R, self.boundary_U
        ])
        self.source = _R_resi ** 2 * (kappa0 ** 2 + 1) / (kappa0 * q0)
        self.true_resi = (_R_resi ** 2 * _Z_resi ** 2 + kappa0 ** 2 / 4 * (
            _R_resi ** 2 - 1
            ) ** 2) / (2 * kappa0 * q0)
        self.true_L = (Rmin ** 2 * _Z_bdry ** 2 + kappa0 ** 2 / 4 * (
            Rmin ** 2 - 1
            ) ** 2) / (2 * kappa0 * q0)
        self.true_D = (_R_bdry ** 2 * Zmin ** 2 + kappa0 ** 2 / 4 * (
            _R_bdry ** 2 - 1
            ) ** 2) / (2 * kappa0 * q0)
        self.true_R = (Rmax ** 2 * _Z_bdry ** 2 + kappa0 ** 2 / 4 * (
            Rmax ** 2 - 1
            ) ** 2) / (2 * kappa0 * q0)
        self.true_U = (_R_bdry ** 2 * Zmax ** 2 + kappa0 ** 2 / 4 * (
            _R_bdry ** 2 - 1
            ) ** 2) / (2 * kappa0 * q0)
        self.true_bdry = torch.concatenate([
            self.true_L, self.true_D, self.true_R, self.true_U
        ])

    def __len__(self):
        return self.num
    
    def __getitem__(self, index: int) -> Tuple[Tensor]:
        return self.residual[index], self.boundary[index], \
            self.source[index], self.true_resi[index], self.true_bdry[index]

class SolovevTriangularity(Dataset):
    
    def __init__(
        self, num_resi: int, num_bdry: int,
        a: float = 0.5, b: float = 0.7, ep: float = 0.3, lamda: float = 0.0,
        R0: float = 5/3
        ) -> None:

        self.a = a
        self.b = b
        self.ep = ep
        self.lamda = lamda
        self.num = num_resi

        _x_resi = np.linspace(-1.0, 1.0, num_resi+1)
        _x_resi = torch.from_numpy(np.random.uniform(_x_resi[:-1], _x_resi[1:]))
        _y_resi_coef = np.linspace(-1.0, 1.0, num_resi+1)
        _y_resi_coef = torch.from_numpy(np.random.uniform(
            _y_resi_coef[:-1], _y_resi_coef[1:]
            ))
        perm = np.random.permutation(num_resi)
        _y_resi = self.boundary_func(_x_resi)
        self.residual = torch.tensor(
            [[_x_resi[i], _y_resi[i] * _y_resi_coef[perm[i]]] for i in range(num_resi)]
            )
        _x_bdry = np.linspace(-1.0, 1.0, num_bdry+1)
        _x_bdry = torch.from_numpy(np.random.uniform(_x_bdry[:-1], _x_bdry[1:]))
        _y_bdry_coef = torch.from_numpy(np.random.choice([-1.0, 1.0], size=(num_bdry, )))
        _y_bdry = self.boundary_func(_x_bdry)
        self.boundary = torch.tensor(
            [[_x_bdry[i], _y_bdry[i] * _y_bdry_coef[i]] for i in range(num_bdry)]
            )
        self.true_bdry = torch.zeros_like(_x_bdry)

        alpha = (4 * (a ** 2 + b ** 2) * ep + a ** 2 * (2 * lamda - ep ** 3)) / \
            (2 * R0 ** 2 * ep * a ** 2 * b ** 2)
        beta = - lamda / (b ** 2 * ep)
        self.alpha, self.beta = alpha, beta
        self.source = alpha * (R0 * (1 + ep * _x_resi)) ** 2 + beta

    def boundary_func(self, x: Tensor):
        return self.b / self.a * torch.sqrt(
            (1 - (x - 0.5 * self.ep * (1 - x ** 2)) ** 2) / \
            ((1 - 0.25 * self.ep ** 2) * (1 + self.ep * x) ** 2 + \
                self.lamda * x * (1 + 0.5 * self.ep * x))
        )

    def __len__(self):
        return self.num
    
    def __getitem__(self, index: int) -> Tuple[Tensor]:
        return self.residual[index], self.boundary[index], \
            self.source[index]

class IterationDataset(Dataset):

    def __init__(
        self, Rmin: float, Rmax: float, 
        Zmin: float, Zmax: float, nR: int = 65, nZ: int = 65,
        ) -> None:

        mu0 = 4e-7 * torch.pi
        self.num = nR * nZ
        _R_resi = torch.from_numpy(np.linspace(Rmin, Rmax, nR))
        _Z_resi = torch.from_numpy(np.linspace(Zmin, Zmax, nZ))
        _R_resi, _Z_resi = torch.meshgrid(_R_resi, _Z_resi, indexing='ij')
        _R_resi = _R_resi.reshape(-1)
        _Z_resi = _Z_resi.reshape(-1)
        _R_bdry = torch.from_numpy(np.linspace(Rmin, Rmax, nR))
        _Z_bdry = torch.from_numpy(np.linspace(Zmin, Zmax, nZ))

        self.residual = torch.tensor(
            [[_R_resi[i], _Z_resi[i]] for i in range(nR * nZ)]
            )
        self.boundary_L = torch.tensor(
            [[Rmin, _Z_bdry[i]] for i in range(nR)]
        )
        self.boundary_D = torch.tensor(
            [[_R_bdry[i], Zmin] for i in range(nZ)]
        )
        self.boundary_R = torch.tensor(
            [[Rmax, _Z_bdry[i]] for i in range(nR)]
        )
        self.boundary_U = torch.tensor(
            [[_R_bdry[i], Zmax] for i in range(nZ)]
        )
        self.boundary = torch.concatenate([
            self.boundary_L, self.boundary_D, self.boundary_R, self.boundary_U
        ])

        atol, rtol = 1e-10, 1e-3
        blend = 0.0
        
        # paxis, Ip, fvac = 3.36e4, 4.16e5, 1.66 
        paxis, Ip, fvac = 3.36e4, 4.16e5, 0.4

        alpha_m, alpha_n = 1.0, 2.0, 
        xpoints, isoflux = [(1.1, -0.6), (1.1, 0.8)], [(1.1, -0.6, 1.1, 0.6)]

        eq = freegs.Equilibrium(
            tokamak=freegs.machine.TestTokamak(),
            Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax, nx=nR, ny=nZ,
            boundary=freegs.boundary.freeBoundaryHagenow
        )
        constrain = freegs.control.constrain(
            xpoints=xpoints, isoflux=isoflux
        )
        constrain = None
        if constrain is not None:
            constrain(eq)
        profiles = freegs.jtor.ConstrainPaxisIp(
            eq, paxis, Ip, fvac, alpha_m, alpha_n
        )
        psi = eq.psi()
        iteration, limit_iter = 0, 0
        psi_relchange = 10.0
        bdry_val = 0.0
        max_iter = 100
        bdry_val_change = np.inf
        has_been_limited = False
        check_limited = False
        psi_maxchange_iterations, psi_relchange_iterations = [], []
        self.psi_list = []
        self.bdry_val_list = []
        self.rhs = []
        self.bdry = []

        self.psi_coil_grid_list = []
        self.psi_coil_x_list = []
        self.psi_coil_isoflux_list = []
        self.psi_coil_Br_list = []
        self.psi_coil_Bz_list = []

        while True:
            psi_last = psi.copy()
            bdry_val_last = bdry_val
            if (iteration >= limit_iter or has_been_limited) and check_limited:
                eq.check_limited = True
            else:
                eq.check_limited = False
            eq._profiles = profiles
            eq._updateBoundaryPsi()
            Jtor = profiles.Jtor(eq.R, eq.Z, psi, psi_bndry=eq.psi_bndry)
            eq._applyBoundary(eq, Jtor, eq.plasma_psi)
            rhs = -mu0 * eq.R * Jtor
            rhs_tensor = rhs.copy()
            self.rhs.append(torch.from_numpy(rhs_tensor))
            self.bdry.append(torch.from_numpy(np.concatenate([
                eq.plasma_psi[0, :], eq.plasma_psi[:, 0],
                eq.plasma_psi[-1, :], eq.plasma_psi[:, -1]
                ])))
            rhs[0, :] = eq.plasma_psi[0, :]
            rhs[:, 0] = eq.plasma_psi[:, 0]
            rhs[-1, :] = eq.plasma_psi[-1, :]
            rhs[:, -1] = eq.plasma_psi[:, -1]
            plasma_psi = eq._solver(eq.plasma_psi, rhs)
            eq._updatePlasmaPsi(plasma_psi)
            psi_torch = torch.from_numpy(plasma_psi)
            self.psi_list.append(psi_torch)
            dR = eq.R[1, 0] - eq.R[0, 0]
            dZ = eq.Z[0, 1] - eq.Z[0, 0]
            eq._current = romb(romb(Jtor)) * dR * dZ
            eq.Jtor = Jtor
            if eq.is_limited:
                has_been_limited = True
            if eq.psi_bndry is not None:
                bdry_val = eq.psi_bndry
                bdry_val_change = bdry_val_last - bdry_val
                bdry_val_relchange = abs(bdry_val_change / bdry_val)
            else:
                bdry_val_relchange = 2.0 * rtol
            
            psi = eq.psi()
            psi_change = psi_last - psi
            psi_maxchange = np.amax(abs(psi_change))
            psi_relchange = psi_maxchange / (np.amax(psi) - np.amin(psi))
            psi_maxchange_iterations.append(psi_maxchange)
            psi_relchange_iterations.append(psi_relchange)
            self.bdry_val_list.append(bdry_val)

            self.psi_coil_grid_list.append(eq.tokamak.calcPsiFromGreens(eq._pgreen))
            psi_coil_isoflux0 = eq.tokamak.psi(1.1, -0.6)
            psi_coil_isoflux1 = eq.tokamak.psi(1.1, 0.6)
            psi_coil_x0 = eq.tokamak.psi(1.1, -0.6)
            psi_coil_x1 = eq.tokamak.psi(1.1, 0.8)
            self.psi_coil_isoflux_list.append(
                torch.tensor([psi_coil_isoflux0, psi_coil_isoflux1])
                )
            self.psi_coil_x = np.array([psi_coil_x0, psi_coil_x1])
            psi_Br0 = eq.tokamak.Br(*xpoints[0])
            psi_Bz0 = eq.tokamak.Bz(*xpoints[0])
            psi_Br1 = eq.tokamak.Br(*xpoints[1])
            psi_Bz1 = eq.tokamak.Bz(*xpoints[1])
            self.psi_coil_Br_list.append(torch.tensor([psi_Br0, psi_Br1]))
            self.psi_coil_Bz_list.append(torch.tensor([psi_Bz0, psi_Bz1]))

            print(f"iteration: {iteration:02d} | " \
                f"psi_relchange: {psi_relchange:.4f} | " \
                f"bndry_relchange: {bdry_val_relchange:.4f} | " \
                f"bndry_change: {bdry_val_change:.4f}"
                )
            if (
                ((psi_maxchange < atol) or (psi_relchange < rtol))
                and (abs(bdry_val_change < atol) or (bdry_val_relchange < rtol))
                or iteration >= max_iter
            ):
                break
            iteration += 1
            if constrain is not None:
                constrain(eq)
            psi = (1.0 - blend) * eq.psi() + blend * psi_last

        self.psi_coil_grid = eq.tokamak.calcPsiFromGreens(eq._pgreen)
        psi_coil_isoflux0 = eq.tokamak.psi(1.1, -0.6)
        psi_coil_isoflux1 = eq.tokamak.psi(1.1, 0.6)
        psi_coil_x0 = eq.tokamak.psi(1.1, -0.6)
        psi_coil_x1 = eq.tokamak.psi(1.1, 0.8)
        self.psi_coil_isoflux_list.append(torch.tensor([psi_coil_isoflux0, psi_coil_isoflux1]))
        self.psi_coil_x_list.append(np.array([psi_coil_x0, psi_coil_x1]))
        psi_Br0 = eq.tokamak.Br(*xpoints[0])
        psi_Bz0 = eq.tokamak.Bz(*xpoints[0])
        psi_Br1 = eq.tokamak.Br(*xpoints[1])
        psi_Bz1 = eq.tokamak.Bz(*xpoints[1])
        self.psi_coil_Br = torch.tensor([psi_Br0, psi_Br1])
        self.psi_coil_Bz = torch.tensor([psi_Bz0, psi_Bz1])

        self.source = self.rhs[-1].reshape(-1)
        self.true = self.psi_list[-1]
        self.true_bdry = self.bdry[-1]
        self.psi_coil_grid = self.psi_coil_grid_list[-1]
        self.psi_coil_x = self.psi_coil_x_list[-1]
        self.psi_coil_isoflux = self.psi_coil_isoflux_list[-1]
        self.psi_coil_Br = self.psi_coil_Br_list[-1]
        self.psi_coil_Bz = self.psi_coil_Bz_list[-1]

    def __len__(self):
        return self.num
    
    def __getitem__(self, index: int) -> Tuple[Tensor]:
        return self.residual[index], self.boundary[index], \
            self.source[index], self.true[index], self.true_bdry[index]
import os
import pickle
import torch
import freegs.freegs as freegs
from freegs4e.multigrid import createVcycle
from freegs4e.gradshafranov import GSsparse4thOrder
from freegsnke import (
    build_machine,
    equilibrium_update,
    GSstaticsolver,
    jtor_update,
)
import numpy as np
from torch.utils.data import Dataset
from freegs.freegs.gradshafranov import Greens
from freegsdeep.typing import *
from scipy.integrate import romb


class GSrhsdataset(Dataset):
    
    def __init__(
        self, Rmin: float, Rmax: float, Zmin: float, Zmax: float, 
        nR: int, nZ: int, num: int, max_iter: int = 100,
        load_path: Optional[str] = None, save_path: Optional[str] = None
        ) -> None:

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
        ], dim=0)

        if load_path is not None:
            load_path = os.path.join('data', load_path)
            self.idx_f = torch.load(os.path.join(load_path, 'index_f.pt'), weights_only=True)
            self.rhs_f = torch.load(os.path.join(load_path, 'rhs_f.pt'), weights_only=True)
            self.bdry_f = torch.load(os.path.join(load_path, 'bdry_f.pt'), weights_only=True)
            self.psi_list_f = torch.load(os.path.join(load_path, 'psi_f.pt'), weights_only=True)
            self.idx_g = torch.load(os.path.join(load_path, 'index_g.pt'), weights_only=True)
            self.psi_g = torch.load(os.path.join(load_path, 'psi_g.pt'), weights_only=True)
            self.R0_g = torch.load(os.path.join(load_path, 'R0_g.pt'), weights_only=True)
            self.tokamak_psi_g = torch.load(os.path.join(load_path, 'tokamak_psi_g.pt'), weights_only=True)
            self.constraint_g = torch.load(os.path.join(load_path, 'constraint_g.pt'), weights_only=True)
            self.U = torch.load(os.path.join(load_path, 'U.pt'), weights_only=True)
        else:
            mu0 = 4e-7 * torch.pi
            self.num = nR * nZ
            self.idx = []
            self.bdry_val_list = []
            self.axis_val_list = []
            self.psi_list = []
            self.rhs = []
            self.bdry = []
            self.U = []
            self.psi_coil_grid_list = []
            self.psi_coil_x_list = []
            self.psi_coil_isoflux_list = []
            self.psi_coil_Br_list = []
            self.psi_coil_Bz_list = []

            paxis = np.random.uniform(1e3, 1e5, (num))
            Ip = np.random.uniform(5e4, 1e6, (num))
            fvac = np.random.uniform(0.5, 2.0, (num))
            phy_list = []
            for p, i, f in zip(paxis, Ip, fvac):
                phy_list.append(
                    (p, i, f, 1.0, 2.0, [(1.1, -0.6), (1.1, 0.8)], [(1.1, -0.6, 1.1, 0.6)])
                )
            atol, rtol = 1e-6, 1e-4
            blend = 0.0
            # constrain = None
            for idx, (paxis, Ip, fvac, alpha_m, alpha_n, xpoints, isoflux) in enumerate(phy_list):
                eq = freegs.Equilibrium(
                    tokamak=freegs.machine.TestTokamak(),
                    Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax, nx=nR, ny=nZ,
                    boundary=freegs.boundary.freeBoundaryHagenow
                )
                constrain = freegs.control.constrain(
                    xpoints=xpoints, isoflux=isoflux
                )
                if constrain is not None:
                    constrain(eq)
                profiles = freegs.jtor.ConstrainPaxisIp(
                    eq, paxis, Ip, fvac, alpha_m, alpha_n
                )
                psi = eq.psi()
                iteration, limit_iter = 0, 0
                psi_relchange = 10.0
                bdry_val = 0.0
                bdry_val_change = np.inf
                has_been_limited = False
                ok_to_break = False
                check_limited = False
                psi_maxchange_iterations, psi_relchange_iterations = [], []
                print(f"Physical state {idx:04d} | " \
                    f"paxis {paxis:.2e} Ip {Ip:.2e} fvac {fvac:.2e}"
                    )
                while True:
                    self.idx.append(idx)
                    psi_last = psi.copy()
                    bdry_val_last = bdry_val
                    if (iteration >= limit_iter or has_been_limited) and check_limited:
                        eq.check_limited = True
                    else:
                        eq.check_limited = False
                    eq._profiles = profiles
                    eq._updateBoundaryPsi()
                    Jtor = profiles.Jtor(eq.R, eq.Z, psi, psi_bndry=eq.psi_bndry)
                    if Jtor is None:
                        break
                    dR = eq.R[1, 0] - eq.R[0, 0]
                    dZ = eq.Z[0, 1] - eq.Z[0, 0]
                    rhs = eq.R * Jtor
                    rhs[0, :] = 0.0
                    rhs[:, 0] = 0.0
                    rhs[-1, :] = 0.0
                    rhs[:, -1] = 0.0
                    psi_fixed = eq.callSolver(eq.plasma_psi, rhs)
                    psi_fixed_tensor = torch.from_numpy(psi_fixed)
                    self.U.append(psi_fixed_tensor)
                    coeffs = [(0, 25.0 / 12), (1, -4.0), (2, 3.0), (3, -16.0 / 12), (4, 1.0 / 4)]
                    dUdn_L = (sum([
                        weight * psi_fixed[index, :] for index, weight in coeffs
                        ]) / dR)
                    dUdn_D = (sum([
                        weight * psi_fixed[:, index] for index, weight in coeffs
                        ]) / dZ)
                    dUdn_R = (sum([
                        weight * psi_fixed[-(1 + index), :] for index, weight in coeffs
                        ]) / dR)
                    dUdn_U = (sum([
                        weight * psi_fixed[:, -(1 + index)] for index, weight in coeffs
                        ]) / dZ)
                    dd = np.sqrt(dR ** 2 + dZ ** 2)
                    dUdn_L[0] = dUdn_D[0] = (sum([
                        weight * psi_fixed[index, index] for index, weight in coeffs
                        ]) / dd)
                    dUdn_L[-1] = dUdn_U[0] = (sum([
                        weight * psi_fixed[index, -(1 + index)] for index, weight in coeffs
                        ]) / dd)
                    dUdn_R[0] = dUdn_D[-1] = (sum([
                        weight * psi_fixed[-(1 + index), index] for index, weight in coeffs
                        ]) / dd)
                    dUdn_R[-1] = dUdn_U[-1] = (sum([
                        weight * psi_fixed[-(1 + index), -(1 + index)] for index, weight in coeffs
                        ]) / dd)
                    
                    eps = 1e-2
                    bndry_indices = np.concatenate([
                            [(x, 0, 0.0, -eps) for x in range(nR)],
                            [(x, nZ - 1, 0.0, eps) for x in range(nR)],
                            [(0, y, -eps, 0.0) for y in range(nZ)],
                            [(nR - 1, y, eps, 0.0) for y in range(nZ)],
                        ]) 

                    for x, y, Reps, Zeps in bndry_indices:
                        x = int(round(x))
                        y = int(round(y))
                        Rpos = eq.R[x, y] + Reps
                        Zpos = eq.Z[x, y] + Zeps
                        greenfunc = Greens(eq.R[0, :], eq.Z[0, :], Rpos, Zpos)
                        result = romb(greenfunc * dUdn_L / eq.R[0, :]) * dZ
                        greenfunc = Greens(eq.R[-1, :], eq.Z[-1, :], Rpos, Zpos)
                        result += romb(greenfunc * dUdn_R / eq.R[-1, :]) * dZ
                        greenfunc = Greens(eq.R[:, 0], eq.Z[:, 0], Rpos, Zpos)
                        result += romb(greenfunc * dUdn_D / eq.R[:, 0]) * dR
                        greenfunc = Greens(eq.R[:, -1], eq.Z[:, -1], Rpos, Zpos)
                        result += romb(greenfunc * dUdn_U / eq.R[:, -1]) * dR
                        eq.plasma_psi[x, y] = result
                    rhs = -mu0 * eq.R * Jtor
                    rhs_tensor = rhs.copy()
                    self.rhs.append(torch.from_numpy(rhs_tensor))
                    self.bdry.append(torch.from_numpy(np.concatenate([
                        eq.plasma_psi[0, :], eq.plasma_psi[:, 0],
                        eq.plasma_psi[-1, :], eq.plasma_psi[:, -1]
                        ], axis=0)))
                    rhs[0, :] = eq.plasma_psi[0, :]
                    rhs[:, 0] = eq.plasma_psi[:, 0]
                    rhs[-1, :] = eq.plasma_psi[-1, :]
                    rhs[:, -1] = eq.plasma_psi[:, -1]
                    plasma_psi = eq._solver(eq.plasma_psi, rhs)
                    psi_torch = torch.from_numpy(plasma_psi)
                    self.psi_list.append(psi_torch)
                    eq._updatePlasmaPsi(plasma_psi)
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
                    if eq.psi_axis is not None:
                        self.axis_val_list.append(eq.psi_axis)
                    else:
                        self.axis_val_list.append(-1000.0)
                    
                    psi = eq.psi()
                    psi_change = psi_last - psi
                    psi_maxchange = np.amax(abs(psi_change))
                    psi_relchange = psi_maxchange / (np.amax(psi) - np.amin(psi))
                    psi_maxchange_iterations.append(psi_maxchange)
                    psi_relchange_iterations.append(psi_relchange)
                    self.bdry_val_list.append(bdry_val)
                    psi_coil_grid_torch = torch.from_numpy(
                        eq.tokamak.calcPsiFromGreens(eq._pgreen)
                        )
                    self.psi_coil_grid_list.append(psi_coil_grid_torch)
                    psi_coil_isoflux0 = eq.tokamak.psi(1.1, -0.6)
                    psi_coil_isoflux1 = eq.tokamak.psi(1.1, 0.6)
                    psi_coil_x0 = eq.tokamak.psi(1.1, -0.6)
                    psi_coil_x1 = eq.tokamak.psi(1.1, 0.8)
                    self.psi_coil_isoflux_list.append(
                        torch.tensor([psi_coil_isoflux0, psi_coil_isoflux1])
                        )
                    self.psi_coil_x = np.array([psi_coil_x0, psi_coil_x1])
                    self.psi_coil_x_list.append(np.array([psi_coil_x0, psi_coil_x1]))
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
                        ((psi_maxchange < atol) and (psi_relchange < rtol))
                        or (abs(bdry_val_change) < atol and (bdry_val_relchange < rtol))
                        or iteration >= max_iter
                    ):
                        break
                    iteration += 1
                    if constrain is not None:
                        constrain(eq)
                    psi = (1.0 - blend) * eq.psi() + blend * psi_last
                
            if save_path is not None:
                save_path = os.path.join('data', save_path)
                os.makedirs(save_path, exist_ok=True)
                torch.save(self.idx, os.path.join(save_path, 'index.pt'))
                torch.save(self.rhs, os.path.join(save_path, 'rhs.pt'))
                torch.save(self.bdry, os.path.join(save_path, 'bdry.pt'))
                torch.save(self.psi_list, os.path.join(save_path, 'psi.pt'))
                torch.save(self.psi_coil_isoflux_list, os.path.join(save_path, 'isoflux.pt'))
                torch.save(self.psi_coil_Br_list, os.path.join(save_path, 'Br.pt'))
                torch.save(self.psi_coil_Bz_list, os.path.join(save_path, 'Bz.pt'))
                torch.save(self.psi_coil_grid_list, os.path.join(save_path, 'coil_grid.pt'))
                torch.save(self.U, os.path.join(save_path, 'U.pt'))

    def __len__(self):
        return len(self.rhs)
    
    def __getitem__(self, index: int) -> Tuple[Tensor]:
        return self.rhs[index], self.bdry[index], self.psi_list[index], \
            self.psi_coil_isoflux_list[index], self.psi_coil_Br_list[index], \
            self.psi_coil_Bz_list[index], self.psi_coil_grid_list[index], \
            self.U[index]

class GSrhsdatasetMASTU_f(Dataset):

    def __init__(
        self, Rmin: float, Rmax: float, Zmin: float, Zmax: float, 
        nR: int, nZ: int, num: int, max_iter: int,
        load_path: Optional[str] = None, save_path: Optional[str] = None
        ) -> None:
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.nx = nR
        self.ny = nZ
        
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
        self.boundary_D = torch.tensor(
            [[_R_bdry[i], Zmin] for i in range(nR)]
        )
        self.boundary_U = torch.tensor(
            [[_R_bdry[i], Zmax] for i in range(nR)]
        )
        self.boundary_L = torch.tensor(
            [[Rmin, _Z_bdry[i]] for i in range(1, nZ-1)]
        )
        self.boundary_R = torch.tensor(
            [[Rmax, _Z_bdry[i]] for i in range(1, nZ-1)]
        )
        self.boundary = torch.concatenate([
            self.boundary_D, self.boundary_U, self.boundary_L, self.boundary_R
        ], dim=0)

        if load_path is not None:
            assert save_path is None, "Cannot load and save at the same time."
            self.load_path(load_path)
        else:
            self.generate_data(num, max_iter)
            if save_path is not None:
                self.save_data(save_path)

    def load_path(self, load_path: str) -> None:
        load_path = os.path.join('data', load_path)
        self.idx_f = torch.load(os.path.join(load_path, 'index_f.pt'), weights_only=True)
        self.rhs_f = torch.load(os.path.join(load_path, 'rhs_f.pt'), weights_only=True)
        self.bdry_f = torch.load(os.path.join(load_path, 'bdry_f.pt'), weights_only=True)
        self.psi_list_f = torch.load(os.path.join(load_path, 'psi_f.pt'), weights_only=True)
        self.idx_g = torch.load(os.path.join(load_path, 'index_g.pt'), weights_only=True)
        self.psi_g = torch.load(os.path.join(load_path, 'psi_g.pt'), weights_only=True)
        self.update_g = torch.load(os.path.join(load_path, 'update_g.pt'), weights_only=True)
        self.tokamak_psi_g = torch.load(os.path.join(load_path, 'tokamak_psi_g.pt'), weights_only=True)
        self.constraint_g = torch.load(os.path.join(load_path, 'constraint_g.pt'), weights_only=True)
        self.Q_list_g = torch.load(os.path.join(load_path, 'Q_list_g.pt'), weights_only=True)
        self.G_list_g = torch.load(os.path.join(load_path, 'G_list_g.pt'), weights_only=True)
        self.psi_list_h = torch.load(os.path.join(load_path, 'psi_h.pt'), weights_only=True)
        self.tokamak_psi_h = torch.load(os.path.join(load_path, 'tokamak_psi_h.pt'), weights_only=True)
        self.psi_axis_h = torch.load(os.path.join(load_path, 'psi_axis_h.pt'), weights_only=True)
        self.psi_bndry_h = torch.load(os.path.join(load_path, 'psi_bndry_h.pt'), weights_only=True)
        self.flag_limiter_h = torch.load(os.path.join(load_path, 'flag_limiter_h.pt'), weights_only=True)

        return None
    
    def generate_data(
        self, num: int, max_iter: int, 
        alpha_m: float = 1.8, alpha_n: float = 1.2
        ) -> None:

        _R_cpu, _Z_cpu = np.meshgrid(
            np.linspace(self.Rmin, self.Rmax, self.nx),
            np.linspace(self.Zmin, self.Zmax, self.ny),
            indexing='ij'
        )
        
        self.num = self.nx * self.ny
        self.idx_f = []
        self.psi_list_f = []
        self.rhs_f = []
        self.bdry_f = []
        self.idx_g = []
        self.update_g = []
        self.psi_list_g = []
        self.tokamak_psi_g = []
        self.R0_g = []
        self.constraint_g = []
        self.Q_list_g = []
        self.G_list_g = []
        self.psi_list_h = []
        self.tokamak_psi_h = []
        self.psi_axis_h = []
        self.psi_bndry_h = []
        self.flag_limiter_h = []

        tokamak_path = 'freegsnke/machine_configs/MAST-U'
        tokamak = build_machine.tokamak(
            active_coils_path=os.path.join(
                tokamak_path, 'MAST-U_like_active_coils.pickle'
                ),
            passive_coils_path=os.path.join(
                tokamak_path, 'MAST-U_like_passive_coils.pickle'
                ),
            limiter_path=os.path.join(
                tokamak_path, 'MAST-U_like_limiter.pickle'
                ),
            wall_path=os.path.join(
                tokamak_path, 'MAST-U_like_wall.pickle'
                ),
        )

        with open('freegsnke/examples/data/simple_diverted_currents_PaxisIp.pk', 'rb') as f:
            currents_dict_diverted = pickle.load(f)
        with open('freegsnke/examples/data/simple_limited_currents_PaxisIp.pk', 'rb') as f:
            currents_dict_limited = pickle.load(f)

        # paxis = [8e3]
        # Ip = [6e5]
        # fvac = [0.5]
        paxis = np.random.uniform(1e3, 5e4, (num))
        Ip = np.random.uniform(5e4, 1e6, (num))
        fvac = np.random.uniform(0.5, 1.5, (num))
        limited_or_diverted = np.random.choice([0, 1], size=(num))
        phy_list = []
        for p, i, f in zip(paxis, Ip, fvac):
            phy_list.append(
                (p, i, f, alpha_m, alpha_n)
            )

        target_relative_tolerance = 1e-6
        max_solving_iterations = max_iter
        Picard_handover = 0.2
        max_rel_update_size = 0.15

        linear_GS_solver = createVcycle(
            self.nx, self.ny, GSsparse4thOrder(
                self.Rmin, self.Rmax, self.Zmin, self.Zmax,
            ),
            nlevels=1, ncycle=1, niter=100, direct=False
        )

        for idx, (paxis, Ip, fvac, alpm, alpn) in enumerate(phy_list):
            eq = equilibrium_update.Equilibrium(
                tokamak=tokamak,
                Rmin=self.Rmin, Rmax=self.Rmax, Zmin=self.Zmin,
                Zmax=self.Zmax, nx=self.nx, ny=self.ny,
            )
            currents_dict = currents_dict_diverted if limited_or_diverted[idx] == 1 else currents_dict_limited 
            currents_dict_perturb = np.random.uniform(
                low=0.5, high=1.75, size=len(currents_dict)
            )
            constraint = [paxis, Ip, fvac]
            current_iter = []
            for idx2, key in enumerate(currents_dict.keys()):
                eq.tokamak.set_coil_current(
                    coil_label=key,
                    current_value=currents_dict[key] * currents_dict_perturb[idx2]
                    )
                if currents_dict[key] != 0.0:
                    current_iter.append(
                        currents_dict[key] * currents_dict_perturb[idx2]
                        )
            profiles = jtor_update.ConstrainPaxisIp(
                eq, paxis, Ip, fvac, alpm, alpn
            )
            solver = GSstaticsolver.NKGSsolver(eq)
            picard_flag = 0
            trial_plasma_psi = np.copy(eq.plasma_psi).reshape(-1)
            solver.tokamak_psi = eq.tokamak.getPsitokamak(
                vgreen=eq._vgreen
                ).reshape(-1)
            control_trial_psi = False
            n_up = 0.0 + 4 * eq.solved
            while (control_trial_psi is False) and (n_up < 10):
                try:
                    res0 = solver.F_function(
                        trial_plasma_psi, solver.tokamak_psi, profiles
                        )
                    print(f'{idx} | Residual found')
                    control_trial_psi = True
                except:
                    trial_plasma_psi /= 0.8
                    n_up += 1
                    print(f'{idx} | Residual not found with trial {n_up}')
            if control_trial_psi is False:
                eq.plasma_psi = trial_plasma_psi = eq.create_psi_plasma_default(
                    adaptive_centre=True
                )
                eq.adjust_psi_plasma()
                trial_plasma_psi = np.copy(eq.plasma_psi).reshape(-1)
                res0 = solver.F_function(
                    trial_plasma_psi, solver.tokamak_psi, profiles
                    )
                
                control_trial_psi = True
            
            solver.jtor_at_start = profiles.jtor.copy()
            norm_rel_change = solver.relative_norm_residual(res0, trial_plasma_psi)
            rel_change, del_psi = solver.relative_del_residual(res0, trial_plasma_psi)
            solver.relative_change = 1.0 * rel_change
            solver.norm_rel_change = [1.0 * norm_rel_change]

            solver.best_relative_change = rel_change
            solver.best_psi = trial_plasma_psi
            args = [solver.tokamak_psi, profiles]
            starting_direction = np.copy(res0)
            print(f'{idx} | Initial relative error {rel_change:.4e}')

            solver.initial_rel_residual = 1.0 * rel_change
            iterations = 0
            reduced_failure = False
            while (rel_change > target_relative_tolerance) * (
                iterations < max_solving_iterations
            ) and reduced_failure == False:
                if rel_change > Picard_handover:
                    print(f"{idx} | Picard iteration " + str(iterations))

                    if picard_flag < min(max_solving_iterations - 1, 3):
                        res0_2d = res0.reshape(self.nx, self.ny)
                        res0 = 0.5 * (res0_2d + res0_2d[:, ::-1]).reshape(-1)
                        picard_flag += 1
                    else:
                        picard_flag = 1
                    update = -1.0 * res0
                else:
                    print(f'{idx} | NK iteration ' + str(iterations))
                    picard_flag = False
                    solver.nksolver.Arnoldi_iteration(
                        x0=trial_plasma_psi.copy(),
                        dx=starting_direction.copy(),
                        R0=res0.copy(),
                        F_function=solver.F_function,
                        args=args,
                        step_size=2.5,
                        scaling_with_n=-1.0,
                        target_relative_unexplained_residual= \
                            0.3,
                        max_n_directions=8,
                        clip=10,
                    )
                    update = 1.0 * solver.nksolver.dx
                del_update = np.amax(update) - np.amin(update)
                if del_update / del_psi > max_rel_update_size:
                    update *= np.abs(max_rel_update_size * del_psi / del_update)
                new_residual_flag = True
                num_update_reduce = 0
                while new_residual_flag:
                    try:
                        n_trial_plasma_psi = trial_plasma_psi + update
                        solver.jtor = profiles.Jtor(
                            _R_cpu, _Z_cpu, (
                                solver.tokamak_psi + n_trial_plasma_psi
                                ).reshape(self.nx, self.ny)
                        )
                        solver.rhs = solver.rhs_before_jtor * solver.jtor 
                        
                        solver.psi_boundary = np.zeros_like(_R_cpu)
                        psi_bnd = np.tensordot(
                            solver.greenfunc, solver.jtor, 
                            axes=([1, 2], [0, 1])
                            )
                        solver.psi_boundary[:, 0] = psi_bnd[:self.nx]
                        solver.psi_boundary[:, -1] = psi_bnd[self.nx:2*self.nx]
                        solver.psi_boundary[0, 1:self.ny-1] = psi_bnd[
                            2 * self.nx:2 * self.nx + (self.ny - 2)
                        ]
                        solver.psi_boundary[-1, 1:self.ny-1] = psi_bnd[
                            2 * self.nx + self.ny - 2:
                        ]
                        if ~np.any(np.isnan(solver.psi_boundary)):
                            self.idx_f.append(idx)
                            # self.psi_list_f.append(n_trial_plasma_psi)
                            self.rhs_f.append(solver.rhs)
                            self.bdry_f.append(psi_bnd)
                            self.psi_list_h.append(n_trial_plasma_psi)
                            self.tokamak_psi_h.append(solver.tokamak_psi)
                            self.psi_axis_h.append(profiles.inputs[0])
                            self.psi_bndry_h.append(profiles.psi_bndry)
                            self.flag_limiter_h.append(profiles.flag_limiter)
                            if picard_flag == False:
                                self.idx_g.append(idx)
                                self.psi_list_g.append(trial_plasma_psi)
                                self.tokamak_psi_g.append(solver.tokamak_psi)
                                self.R0_g.append(solver.nksolver.R0)
                                self.constraint_g.append(constraint)
                                self.update_g.append(solver.nksolver.dx)
                                self.G_list_g.append(solver.nksolver.Gn)
                                self.Q_list_g.append(solver.nksolver.Qn)

                            solver.rhs[0, :] = solver.psi_boundary[0, :]
                            solver.rhs[:, 0] = solver.psi_boundary[:, 0]
                            solver.rhs[-1, :] = solver.psi_boundary[-1, :]
                            solver.rhs[:, -1] = solver.psi_boundary[:, -1]
                            
                            new_res0 = n_trial_plasma_psi - (
                                solver.linear_GS_solver(
                                    solver.psi_boundary, solver.rhs
                                    )
                            ).reshape(-1)
                            if ~np.any(np.isnan(solver.psi_boundary)):
                                self.psi_list_f.append(solver.linear_GS_solver(
                                        solver.psi_boundary, solver.rhs
                                    ).reshape(-1))
                            new_norm_rel_change = solver.relative_norm_residual(
                                new_res0, n_trial_plasma_psi
                            )
                            new_rel_change, new_del_psi = solver.relative_del_residual(
                                new_res0, n_trial_plasma_psi
                            )
                            new_residual_flag = False
                    except:
                        update *= 0.75
                        num_update_reduce += 1
                        if num_update_reduce > 10:
                            reduced_failure = True
                            print(f'{idx} | Reduced update failed !!')
                            break

                if new_norm_rel_change < 1.2 * solver.norm_rel_change[-1]:
                    trial_plasma_psi = n_trial_plasma_psi.copy()
                    
                    try:
                        residual_collinearity = np.sum(res0 * new_res0) / (
                            np.linalg.norm(res0) * np.linalg.norm(new_res0)
                        )
                        res0 = 1.0 * new_res0
                        if (residual_collinearity > 0.9) and (picard_flag is False):
                            starting_direction = np.sin(
                                np.linspace(0, 2*np.pi, self.nx)
                            * 1.5 * np.random.random()
                            )[:, np.newaxis]
                            starting_direction = starting_direction * np.sin(
                                    np.linspace(0, 2*np.pi, self.ny)
                                    * 1.5 * np.random.random()
                                )[np.newaxis, :]
                            starting_direction = starting_direction.reshape(-1)
                            strating_direction *= trial_plasma_psi
                        else:
                            starting_direction = np.copy(res0)
                    except:
                        starting_direction = np.copy(res0)
                    rel_change = 1.0 * new_rel_change
                    norm_rel_change = 1.0 * new_norm_rel_change
                    del_psi = 1.0 * new_del_psi
                else:
                    reduce_by = solver.relative_change / new_rel_change                       
                    new_residual_flag = True
                    num_update_reduce = 0
                    while new_residual_flag:
                        try:
                            n_trial_plasma_psi = trial_plasma_psi + update * reduce_by
                            res0 = solver.F_function(
                                n_trial_plasma_psi, solver.tokamak_psi, profiles
                            )
                            new_residual_flag = False
                        except:
                            reduce_by *= 0.75
                            num_update_reduce += 1
                            if num_update_reduce > 10:
                                reduced_failure = True
                                print(f'{idx} | Reduced update failed !!')
                                break
                    
                    starting_direction = np.copy(res0)
                    trial_plasma_psi = n_trial_plasma_psi.copy()
                    norm_rel_change = solver.relative_norm_residual(
                        res0, trial_plasma_psi
                    )
                    rel_change, del_psi = solver.relative_del_residual(
                        res0, trial_plasma_psi
                    )
                    if rel_change < solver.best_relative_change:
                        solver.best_relative_change = 1.0 * rel_change
                        solver.best_psi = np.copy(trial_plasma_psi)
                
                solver.relative_change = 1.0 * rel_change
                solver.norm_rel_change.append(norm_rel_change)
                print(f"{idx} | relative error {rel_change:.4e} ")
                iterations += 1

            if solver.best_relative_change < rel_change:
                solver.relative_change = 1.0 * solver.best_relative_change
                trial_plasma_psi = np.copy(solver.best_psi)
                profiles.Jtor(
                    _R_cpu,
                    _Z_cpu,
                    (solver.tokamak_psi + trial_plasma_psi).reshape(self.nx, self.ny),
                )
            eq.plasma_psi = trial_plasma_psi.reshape(self.nx, self.ny).copy()

            # solver.port_critical(eq=eq, profiles=profiles)

            if rel_change > target_relative_tolerance:
                print(
                    f"{idx} | Forward static solve DID NOT CONVERGE. " \
                    f"Tolerance {rel_change:.2e} "\
                    f"(vs. requested {target_relative_tolerance:.2e}) " \
                    f"reached in {int(iterations)}/{int(max_solving_iterations)} iterations."
                )
            else:
                print(
                    f"{idx} | Forward static solve SUCCESS. " \
                    f"Tolerance {rel_change:.2e} " \
                    f"(vs. requested {target_relative_tolerance:.2e}) "\
                    f"reached in {int(iterations)}/{int(max_solving_iterations)}  iterations."
                )

    def save_data(self, save_path: str) -> None:
        save_path = os.path.join('data', save_path)
        os.makedirs(save_path, exist_ok=True)
        self.idx_f = torch.from_numpy(np.array(self.idx_f[:-1]))
        self.rhs_f = torch.from_numpy(np.array(self.rhs_f[:-1]))
        self.bdry_f = torch.from_numpy(np.array(self.bdry_f[:-1]))
        self.psi_list_f = torch.from_numpy(np.array(self.psi_list_f[1:]))
        self.idx_g = torch.from_numpy(np.array(self.idx_g))
        self.psi_g = torch.from_numpy(np.array(self.psi_list_g))
        self.update_g = torch.from_numpy(np.array(self.update_g))
        self.tokamak_psi_g = torch.from_numpy(np.array(self.tokamak_psi_g))
        self.constraint_g = torch.from_numpy(np.array(self.constraint_g))
        self.R0_g = torch.from_numpy(np.array(self.R0_g))
        self.Q_list_g = torch.from_numpy(np.array(self.Q_list_g))
        self.G_list_g = torch.from_numpy(np.array(self.G_list_g))
        self.psi_h = torch.from_numpy(np.array(self.psi_list_h))
        self.tokamak_psi_h = torch.from_numpy(np.array(self.tokamak_psi_h))
        self.psi_axis_h = torch.from_numpy(np.array(self.psi_axis_h))
        self.psi_bndry_h = torch.from_numpy(np.array(self.psi_bndry_h))
        self.flag_limiter_h = torch.from_numpy(np.array(self.flag_limiter_h))
        torch.save(self.idx_f, os.path.join(save_path, 'index_f.pt'))
        torch.save(self.rhs_f, os.path.join(save_path, 'rhs_f.pt'))
        torch.save(self.bdry_f, os.path.join(save_path, 'bdry_f.pt'))
        torch.save(self.psi_list_f, os.path.join(save_path, 'psi_f.pt'))
        torch.save(self.idx_g, os.path.join(save_path, 'index_g.pt'))
        torch.save(self.psi_g, os.path.join(save_path, 'psi_g.pt'))
        torch.save(self.update_g, os.path.join(save_path, 'update_g.pt'))
        torch.save(self.tokamak_psi_g, os.path.join(save_path, 'tokamak_psi_g.pt'))
        torch.save(self.G_list_g, os.path.join(save_path, 'G_list_g.pt'))
        torch.save(self.constraint_g, os.path.join(save_path, 'constraint_g.pt'))
        torch.save(self.Q_list_g, os.path.join(save_path, 'Q_list_g.pt'))
        torch.save(self.R0_g, os.path.join(save_path, 'R0_g.pt'))
        torch.save(self.psi_h, os.path.join(save_path, 'psi_h.pt'))
        torch.save(self.tokamak_psi_h, os.path.join(save_path, 'tokamak_psi_h.pt'))
        torch.save(self.psi_axis_h, os.path.join(save_path, 'psi_axis_h.pt'))
        torch.save(self.psi_bndry_h, os.path.join(save_path, 'psi_bndry_h.pt'))
        torch.save(self.flag_limiter_h, os.path.join(save_path, 'flag_limiter_h.pt'))
        return None

    def __len__(self):
        return len(self.rhs_f)
    
    def __getitem__(self, index: int) -> Tuple[Tensor]:
        return self.idx_f[index], self.rhs_f[index], \
            self.bdry_f[index], self.psi_list_f[index]
        
class GSrhsdatasetMASTU_g(GSrhsdatasetMASTU_f, Dataset):
    def __init__(
        self, Rmin: float, Rmax: float, Zmin: float, Zmax: float,
        nR: int, nZ: int, num: int, max_iter: int,
        load_path: Optional[str] = None, save_path: Optional[str] = None
        ) -> None:
        super().__init__(
            Rmin, Rmax, Zmin, Zmax,
            nR, nZ, num, max_iter,
            load_path, save_path
        )
    
    def __len__(self):
        return len(self.tokamak_psi_g)
    
    def __getitem__(self, index: int) -> Tuple[Tensor]:
        return self.idx_g[index], self.psi_g[index], \
            self.tokamak_psi_g[index], self.constraint_g[index], \
            self.R0_g[index], self.Q_list_g[index], self.G_list_g[index]
            # self.update_g[index]
            # self.update_g[index], self.Q_list_g[index]
            
class GSrhsdatasetMASTU_separatrix(GSrhsdatasetMASTU_f, Dataset):
    def __init__(
        self, Rmin: float, Rmax: float, Zmin: float, Zmax: float,
        nR: int, nZ: int, num: int, max_iter: int,
        load_path: Optional[str] = None, save_path: Optional[str] = None
        ) -> None:
        super().__init__(
            Rmin, Rmax, Zmin, Zmax,
            nR, nZ, num, max_iter,
            load_path, save_path
        )
    
    def __len__(self):
        return len(self.idx_f)
    
    def __getitem__(self, index: int) -> Tuple[Tensor]:
        return self.idx_f[index], self.psi_list_h[index], \
            self.tokamak_psi_h[index], self.psi_axis_h[index], \
            self.psi_bndry_h[index], self.flag_limiter_h[index]
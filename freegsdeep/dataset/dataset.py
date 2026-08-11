import os
import pickle
import torch
# import freegs.freegs as freegs
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
from freegs4e.gradshafranov import Greens
from freegsdeep.utils.typing import *
from scipy.integrate import romb

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
            self.generate_data(num, max_iter, save_path)
            self.save_data(save_path)

    def load_path(self, load_path: str) -> None:
        load_path = os.path.join('data', load_path)
        self.idx_f = torch.load(os.path.join(load_path, 'index_f.pt'), weights_only=True)
        self.rhs_f = torch.load(os.path.join(load_path, 'rhs_f.pt'), weights_only=True)
        self.bdry_f = torch.load(os.path.join(load_path, 'bdry_f.pt'), weights_only=True)
        self.psi_list_f = torch.load(os.path.join(load_path, 'psi_f.pt'), weights_only=True)
        for name in [
            'constraint_f',
            'tokamak_psi_f',
            'psi_axis_f',
            'psi_bndry_f',
            'flag_limiter_f',
            'diverted_psi_bndry_f',
            'psi_on_limiter_f',
            'limiter_margin_f',
            'mask_size_f',
        ]:
            path = os.path.join(load_path, f'{name}.pt')
            if os.path.exists(path):
                setattr(self, name, torch.load(path, weights_only=True))
        row_ok = (
            torch.isfinite(self.rhs_f.flatten(1)).all(dim=1) &
            torch.isfinite(self.bdry_f.flatten(1)).all(dim=1) &
            torch.isfinite(self.psi_list_f.flatten(1)).all(dim=1)
        )
        if not bool(row_ok.all()):
            print(
                f"Dropping {int((~row_ok).sum())} non-finite F samples "
                f"from {load_path}."
            )
            n_f = len(row_ok)
            self.idx_f = self.idx_f[row_ok]
            self.rhs_f = self.rhs_f[row_ok]
            self.bdry_f = self.bdry_f[row_ok]
            self.psi_list_f = self.psi_list_f[row_ok]
            for name in [
                'constraint_f',
                'tokamak_psi_f',
                'psi_axis_f',
                'psi_bndry_f',
                'flag_limiter_f',
                'diverted_psi_bndry_f',
                'psi_on_limiter_f',
                'limiter_margin_f',
                'mask_size_f',
            ]:
                value = getattr(self, name, None)
                if torch.is_tensor(value) and value.ndim > 0 and len(value) == n_f:
                    setattr(self, name, value[row_ok])
        self.idx_g = torch.load(os.path.join(load_path, 'index_g.pt'), weights_only=True)
        self.psi_g = torch.load(os.path.join(load_path, 'psi_g.pt'), weights_only=True)
        self.update_g = torch.load(os.path.join(load_path, 'update_g.pt'), weights_only=True)
        self.tokamak_psi_g = torch.load(os.path.join(load_path, 'tokamak_psi_g.pt'), weights_only=True)
        self.R0_g = torch.load(os.path.join(load_path, 'R0_g.pt'), weights_only=True)
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
        self, num: int, max_iter: int, save_path: Optional[str],
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
        self.res0_f = []
        self.constraint_f = []
        self.tokamak_psi_f = []
        self.psi_axis_f = []
        self.psi_bndry_f = []
        self.flag_limiter_f = []
        self.diverted_psi_bndry_f = []
        self.psi_on_limiter_f = []
        self.limiter_margin_f = []
        self.mask_size_f = []
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
        beta0 = 8e3 / 6e5 ** 2
        diverted_num = int(0.5 * num)
        transition_num = int(0.2 * num)
        limited_num = num - diverted_num - transition_num
        Ip = np.concatenate([
            np.random.uniform(3.5e5, 6.5e5, (diverted_num)),
            np.random.uniform(3.5e5, 6.0e5, (transition_num)),
            np.random.uniform(3.5e5, 5.5e5, (limited_num))
        ])
        beta_mult = np.concatenate([
            np.random.uniform(0.8, 1.4, (diverted_num)),
            np.random.uniform(1.4, 2.0, (transition_num)),
            np.random.uniform(2.0, 2.4, (limited_num))
        ])
        paxis = beta_mult * beta0 * Ip ** 2
        fvac = np.ones((num)) * 0.5

        phy_list = []
        for p, i, f in zip(paxis, Ip, fvac):
            phy_list.append(
                (p, i, f, alpha_m, alpha_n)
            )

        target_relative_tolerance = 1e-9
        max_solving_iterations = max_iter
        Picard_handover = 0.1
        max_rel_update_size = 0.15

        for idx, (paxis, Ip, fvac, alpm, alpn) in enumerate(phy_list):
            eq = equilibrium_update.Equilibrium(
                tokamak=tokamak,
                Rmin=self.Rmin, Rmax=self.Rmax, Zmin=self.Zmin,
                Zmax=self.Zmax, nx=self.nx, ny=self.ny,
            )
            # currents_dict = currents_dict_diverted if limited_or_diverted[idx] == 1 else currents_dict_limited 
            currents_dict = currents_dict_diverted
            # currents_dict_perturb = np.random.uniform(
            #     low=0.5, high=1.75, size=len(currents_dict)
            # )
            currents_dict_perturb = np.ones(len(currents_dict))
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

            def F_function(plasma_psi, tokamak_psi, profiles):
                solver.jtor = profiles.Jtor(
                    _R_cpu, _Z_cpu, (
                        solver.tokamak_psi + plasma_psi
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
                solver.rhs[0, :] = solver.psi_boundary[0, :]
                solver.rhs[:, 0] = solver.psi_boundary[:, 0]
                solver.rhs[-1, :] = solver.psi_boundary[-1, :]
                solver.rhs[:, -1] = solver.psi_boundary[:, -1]
                psi_list_f = solver.linear_GS_solver(
                    solver.psi_boundary, solver.rhs
                ).reshape(-1)

                if (
                    (not np.isfinite(solver.rhs).all()) or
                    (not np.isfinite(psi_bnd).all()) or
                    (not np.isfinite(psi_list_f).all())
                ):
                    raise FloatingPointError("Non-finite F sample generated.")
                assert np.isclose(
                    solver.F_function(plasma_psi, tokamak_psi, profiles),
                    plasma_psi - psi_list_f, atol=1e-6, rtol=1e-4
                ).all()

                diverted_psi_bndry = getattr(
                    profiles, 'diverted_psi_bndry', np.nan
                )
                interpolated = getattr(
                    profiles.limiter_handler, 'interpolated_on_limiter', []
                )
                if len(interpolated):
                    psi_on_limiter = float(np.amax(np.asarray(interpolated)))
                else:
                    psi_on_limiter = np.nan
                if np.isfinite(psi_on_limiter) and np.isfinite(diverted_psi_bndry):
                    limiter_margin = psi_on_limiter - diverted_psi_bndry
                else:
                    limiter_margin = np.nan

                self.idx_f.append(idx)
                self.rhs_f.append(solver.rhs)
                self.bdry_f.append(psi_bnd)
                self.psi_list_f.append(psi_list_f)
                self.constraint_f.append(constraint)
                self.tokamak_psi_f.append(solver.tokamak_psi)
                self.psi_axis_f.append(profiles.inputs[0])
                self.psi_bndry_f.append(profiles.psi_bndry)
                self.flag_limiter_f.append(profiles.flag_limiter)
                self.diverted_psi_bndry_f.append(diverted_psi_bndry)
                self.psi_on_limiter_f.append(psi_on_limiter)
                self.limiter_margin_f.append(limiter_margin)
                self.mask_size_f.append(
                    int(np.sum(np.asarray(profiles.limiter_core_mask).astype(bool)))
                )

                return plasma_psi - psi_list_f

            while (control_trial_psi is False) and (n_up < 10):
                try:
                    res0 = F_function(
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
            # while (rel_change > target_relative_tolerance) * (
            #     iterations < max_solving_iterations
            # ) and reduced_failure == False:
            while (iterations < max_solving_iterations) and reduced_failure == False:
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
                        F_function=F_function,
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
                        
                        new_res0 = F_function(n_trial_plasma_psi, solver.tokamak_psi, profiles)
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
                            starting_direction *= trial_plasma_psi
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
                if rel_change < target_relative_tolerance:
                    print(
                        f"{idx} | Converged in {int(iterations)} iterations.")
                    break
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
        
            if ((idx + 1) % 10 == 0) & (save_path is not None):
                print(f"Saving data at index {idx} ...")
                self.save_data(save_path)

    def save_data(self, save_path: str) -> None:
        save_path = os.path.join('data', save_path)
        os.makedirs(save_path, exist_ok=True)
        idx_f = torch.from_numpy(np.array(self.idx_f))
        rhs_f = torch.from_numpy(np.array(self.rhs_f))
        bdry_f = torch.from_numpy(np.array(self.bdry_f))
        res0_f = torch.from_numpy(np.array(self.res0_f))
        psi_list_f = torch.from_numpy(np.array(self.psi_list_f))
        constraint_f = torch.from_numpy(np.array(self.constraint_f))
        tokamak_psi_f = torch.from_numpy(np.array(self.tokamak_psi_f))
        psi_axis_f = torch.from_numpy(np.array(self.psi_axis_f))
        psi_bndry_f = torch.from_numpy(np.array(self.psi_bndry_f))
        flag_limiter_f = torch.from_numpy(np.array(self.flag_limiter_f))
        diverted_psi_bndry_f = torch.from_numpy(
            np.array(self.diverted_psi_bndry_f)
        )
        psi_on_limiter_f = torch.from_numpy(np.array(self.psi_on_limiter_f))
        limiter_margin_f = torch.from_numpy(np.array(self.limiter_margin_f))
        mask_size_f = torch.from_numpy(np.array(self.mask_size_f))
        idx_g = torch.from_numpy(np.array(self.idx_g))
        psi_g = torch.from_numpy(np.array(self.psi_list_g))
        update_g = torch.from_numpy(np.array(self.update_g))
        tokamak_psi_g = torch.from_numpy(np.array(self.tokamak_psi_g))
        constraint_g = torch.from_numpy(np.array(self.constraint_g))
        R0_g = torch.from_numpy(np.array(self.R0_g))
        Q_list_g = torch.from_numpy(np.array(self.Q_list_g))
        G_list_g = torch.from_numpy(np.array(self.G_list_g))
        psi_h = torch.from_numpy(np.array(self.psi_list_h))
        tokamak_psi_h = torch.from_numpy(np.array(self.tokamak_psi_h))
        psi_axis_h = torch.from_numpy(np.array(self.psi_axis_h))
        psi_bndry_h = torch.from_numpy(np.array(self.psi_bndry_h))
        flag_limiter_h = torch.from_numpy(np.array(self.flag_limiter_h))
        torch.save(idx_f, os.path.join(save_path, 'index_f.pt'))
        torch.save(rhs_f, os.path.join(save_path, 'rhs_f.pt'))
        torch.save(bdry_f, os.path.join(save_path, 'bdry_f.pt'))
        torch.save(psi_list_f, os.path.join(save_path, 'psi_f.pt'))
        torch.save(constraint_f, os.path.join(save_path, 'constraint_f.pt'))
        torch.save(tokamak_psi_f, os.path.join(save_path, 'tokamak_psi_f.pt'))
        torch.save(psi_axis_f, os.path.join(save_path, 'psi_axis_f.pt'))
        torch.save(psi_bndry_f, os.path.join(save_path, 'psi_bndry_f.pt'))
        torch.save(flag_limiter_f, os.path.join(save_path, 'flag_limiter_f.pt'))
        torch.save(
            diverted_psi_bndry_f,
            os.path.join(save_path, 'diverted_psi_bndry_f.pt')
        )
        torch.save(
            psi_on_limiter_f, os.path.join(save_path, 'psi_on_limiter_f.pt')
        )
        torch.save(
            limiter_margin_f, os.path.join(save_path, 'limiter_margin_f.pt')
        )
        torch.save(mask_size_f, os.path.join(save_path, 'mask_size_f.pt'))
        torch.save(idx_g, os.path.join(save_path, 'index_g.pt'))
        torch.save(psi_g, os.path.join(save_path, 'psi_g.pt'))
        torch.save(update_g, os.path.join(save_path, 'update_g.pt'))
        torch.save(tokamak_psi_g, os.path.join(save_path, 'tokamak_psi_g.pt'))
        torch.save(G_list_g, os.path.join(save_path, 'G_list_g.pt'))
        torch.save(constraint_g, os.path.join(save_path, 'constraint_g.pt'))
        torch.save(Q_list_g, os.path.join(save_path, 'Q_list_g.pt'))
        torch.save(R0_g, os.path.join(save_path, 'R0_g.pt'))
        torch.save(psi_h, os.path.join(save_path, 'psi_h.pt'))
        torch.save(tokamak_psi_h, os.path.join(save_path, 'tokamak_psi_h.pt'))
        torch.save(psi_axis_h, os.path.join(save_path, 'psi_axis_h.pt'))
        torch.save(psi_bndry_h, os.path.join(save_path, 'psi_bndry_h.pt'))
        torch.save(flag_limiter_h, os.path.join(save_path, 'flag_limiter_h.pt'))
        torch.save(res0_f, os.path.join(save_path, 'res0_f.pt'))
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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pickle
import torch
import jax
from jax import vmap
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
import freegs.freegs as freegs
from freegsnke import (
    build_machine,
    equilibrium_update,
    GSstaticsolver,
    jtor_update,
)

import numpy as np
from freegs.freegs.gradshafranov import Greens
from freegsdeep.typing import *
from freegsdeep.model import Integratednet_jax, FNO2D
# from freegsdeep.utils.utils import vectorized_nksolver, vectorized_solver, PCA_preconditioner
from freegsdeep.utils.jtor import ConstrainPaxisIp
from freegsdeep.utils.utils import Greens_jax
import matplotlib.pyplot as plt
from datetime import datetime


class main():

    def __init__(
        self, Rmin: float, Rmax: float, Zmin: float, Zmax: float, 
        nR: int, nZ: int,
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
        self.residual = jnp.asarray(self.residual)
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
        self.boundary = jnp.asarray(self.boundary)

        solver_deep = Integratednet_jax(
            Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax, 
            nx=nR, ny=nZ, hidden_dim=15, key=jax.random.PRNGKey(0)
            )
        with open(
            'logs/debug_jax_transfinite_integrated/model/model_506.eqx', 'rb'
            ) as f:
            solver_deep = eqx.tree_deserialise_leaves(f, solver_deep)
        self.solver_deep = vmap(solver_deep, in_axes=(0, 0, None, None, None))
    
    def freeboundary(self, plasma_psi, tokamak_psi, profiles):
        self.jtor = profiles.Jtor(
            self.R, self.Z, (plasma_psi + tokamak_psi).reshape(self.nx, self.ny)
        )
        self.rhs = self.rhs_before_jtor * self.jtor
        self.psi_boundary = jnp.zeros_like(self.R)
        psi_bnd = jnp.tensordot(self.greenfunc, self.jtor, axes=([1, 2], [0, 1]))

        self.psi_boundary = self.psi_boundary.at[:, 0].set(psi_bnd[: self.nx])
        self.psi_boundary = self.psi_boundary.at[:, -1].set(psi_bnd[self.nx : 2 * self.nx])
        self.psi_boundary = self.psi_boundary.at[0, 1 : self.ny - 1].set(psi_bnd[
            2 * self.nx : 2 * self.nx + self.ny - 2
        ])
        self.psi_boundary = self.psi_boundary.at[-1, 1 : self.ny - 1].set(psi_bnd[2 * self.nx + self.ny - 2 :])

        self.rhs = self.rhs.at[0, :].set(self.psi_boundary[0, :])
        self.rhs = self.rhs.at[:, 0].set(self.psi_boundary[:, 0])
        self.rhs = self.rhs.at[-1, :].set(self.psi_boundary[-1, :])
        self.rhs = self.rhs.at[:, -1].set(self.psi_boundary[:, -1])


    def F_function(
        self, plasma_psi: Array, tokamak_psi: Array, profiles: Array,
        ) -> Array:
        self.freeboundary(plasma_psi, tokamak_psi, profiles)
        rhs = self.rhs
        psi_bnd = jnp.zeros((len(self.boundary), 1))
        psi_bnd = psi_bnd.at[:self.nx, 0].set(rhs[:, 0]) 
        psi_bnd = psi_bnd.at[self.nx:2*self.nx, 0].set(rhs[:, -1])
        psi_bnd = psi_bnd.at[2*self.nx:2*self.nx+self.ny-2, 0].set(rhs[0, 1:-1])
        psi_bnd = psi_bnd.at[2*self.nx+self.ny-2:, 0].set(rhs[-1, 1:-1])

        rhs = rhs[None, :, :]
        residual = plasma_psi - self.solver_deep(
                self.residual[:, 0:1], self.residual[:, 1:2],
                rhs, self.boundary, psi_bnd
        ).reshape(-1)
        return residual
    
    def simulation(
        self, image_path: str, 
        alpha_m: float = 1.8, alpha_n: float = 1.2, picard_handover: float = 0.1,
        ) -> None:
        _R_cpu, _Z_cpu = np.meshgrid(
            np.linspace(self.Rmin, self.Rmax, self.nx),
            np.linspace(self.Zmin, self.Zmax, self.ny),
            indexing='ij'
        )
        
        self.num = self.nx * self.ny
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
        # with open('freegsnke/examples/data/simple_diverted_currents_PaxisIp.pk', 'rb') as f:
        with open('freegsnke/examples/data/simple_diverted_currents_PaxisIp.pk', 'rb') as f:
            currents_dict = pickle.load(f)

        paxis = 8e3
        Ip = 6e5
        fvac = 0.5

        target_relative_tolerance = 1e-6
        max_solving_iterations = 200
        Picard_handover = picard_handover
        max_rel_update_size = 0.15

        eq = equilibrium_update.Equilibrium(
            tokamak=tokamak,
            Rmin=self.Rmin, Rmax=self.Rmax, Zmin=self.Zmin, Zmax=self.Zmax,
            nx=self.nx, ny=self.ny,
        )
        self.R = jnp.asarray(eq.R)
        self.Z = jnp.asarray(eq.Z)
        self.rhs_before_jtor = -4e-7 * jnp.pi * self.R
        # matrices of responses of boundary locations to each grid positions
        dR = self.R[1, 0] - self.R[0, 0]
        dZ = self.Z[0, 1] - self.Z[0, 0]
        self.dRdZ = dR * dZ
        R_1D = self.R[:, 0]
        Z_1D = self.Z[0, :]
        bndry_indices = np.concatenate(
            [
                [(x, 0) for x in range(self.nx)],
                [(x, self.ny - 1) for x in range(self.nx)],
                [(0, y) for y in np.arange(1, self.ny - 1)],
                [(self.nx - 1, y) for y in np.arange(1, self.ny - 1)],
            ]
        )
        greenfunc = Greens_jax(
            self.R[jnp.newaxis, :, :],
            self.Z[jnp.newaxis, :, :],
            R_1D[bndry_indices[:, 0]][:, jnp.newaxis, jnp.newaxis],
            Z_1D[bndry_indices[:, 1]][:, jnp.newaxis, jnp.newaxis],
        )
        # Prevent infinity/nan by removing Greens(x,y;x,y)
        zeros = jnp.ones_like(greenfunc)
        zeros = zeros.at[
            jnp.arange(len(bndry_indices)), bndry_indices[:, 0], bndry_indices[:, 1]
        ].set(0)
        self.greenfunc = greenfunc * zeros * self.dRdZ
        # currents_dict_perturb = np.random.uniform(
        #     low=0.5, high=1.75, size=len(currents_dict)
        # )
        # print('currents_perturb', currents_dict_perturb)
        # for idx2, key in enumerate(currents_dict.keys()):
        #     eq.tokamak.set_coil_current(
        #         coil_label=key,
        #         current_value=currents_dict[key] * currents_dict_perturb[idx2]
        #         )
        for key in currents_dict.keys():
            eq.tokamak.set_coil_current(coil_label=key, current_value=currents_dict[key])
        # profiles = jtor_update.ConstrainPaxisIp(
        #     eq, paxis, Ip, fvac, alpha_m, alpha_n
        # )
        profiles = ConstrainPaxisIp(
            eq, paxis, Ip, fvac, alpha_m, alpha_n
        )
        self.solver = GSstaticsolver.NKGSsolver(eq)
        # self.vectorized_solver = vectorized_solver(eq, tokamak.limiter)
        # self.vectorized_nksolver = vectorized_nksolver(problem_dimension=self.num)
        start_time = datetime.now()
        picard_flag = 1 
        trial_plasma_psi = np.copy(eq.plasma_psi).reshape(-1)
        self.solver.tokamak_psi = eq.tokamak.getPsitokamak(
            vgreen=eq._vgreen
            ).reshape(-1)
        control_trial_psi = False
        n_up = 0.0 + 4 * eq.solved
        while (control_trial_psi is False) and (n_up < 10):
            try:
                res0 = self.solver.F_function(
                    trial_plasma_psi, self.solver.tokamak_psi, profiles
                    )
                control_trial_psi = True
            except:
                trial_plasma_psi /= 0.8
                n_up += 1
        if control_trial_psi is False:
            eq.plasma_psi = trial_plasma_psi = eq.create_psi_plasma_default(
                adaptive_centre=True
            )
            eq.adjust_psi_plasma()
            trial_plasma_psi = np.copy(eq.plasma_psi).reshape(-1)
            res0 = self.solver.F_function(
                trial_plasma_psi, self.solver.tokamak_psi, profiles
                )
            
            control_trial_psi = True
        
        self.solver.jtor_at_start = profiles.jtor.copy()
        norm_rel_change = self.solver.relative_norm_residual(res0, trial_plasma_psi)
        rel_change, del_psi = self.solver.relative_del_residual(res0, trial_plasma_psi)
        self.solver.relative_change = 1.0 * rel_change
        self.solver.norm_rel_change = [1.0 * norm_rel_change]
        self.solver.best_relative_change = rel_change
        self.solver.best_psi = trial_plasma_psi
        starting_direction = np.copy(res0)

        self.solver.initial_rel_residual = 1.0 * rel_change
        iterations = 0
        reduced_failure = False
        while (rel_change > target_relative_tolerance) * (
            iterations < max_solving_iterations
        ) and reduced_failure == False:
            if rel_change > Picard_handover and picard_flag:
                print("Picard iteration: " + str(iterations))
                # if picard_flag < min(max_solving_iterations - 1, 3):
                #     res0_2d = res0.reshape(self.nx, self.ny)
                #     res0 = 0.5 * (res0_2d + res0_2d[:, ::-1]).reshape(-1)
                #     picard_flag += 1
                # else:
                #     picard_flag = 1
                update = -1.0 * res0
            else:
                # print("Neural operator iteration: " + str(iterations))
                # picard_flag = False
                # neural_flag = True
                # if picard_flag < min(max_solving_iterations - 1, 3):
                #     res0_2d = res0.reshape(self.nx, self.ny)
                #     res0 = 0.5 * (res0_2d + res0_2d[:, ::-1]).reshape(-1)
                #     picard_flag += 1
                # else:
                #     picard_flag = 1
                # update = -1.0 * res0
                res0 = np.asarray(res0)
                print("Newton-Krylov iteration: " + str(iterations))
                picard_flag = False
                profiles = jtor_update.ConstrainPaxisIp(
                    eq, paxis, Ip, fvac, alpha_m, alpha_n
                )
                args = [self.solver.tokamak_psi, profiles]
                # output = self.solver_nk(
                #     jnp.concatenate([
                #         jnp.asarray(trial_plasma_psi.reshape(1, self.nx, self.ny)),
                #         jnp.asarray(self.solver.tokamak_psi.reshape(1, self.nx, self.ny)),
                #         jnp.asarray(res0.reshape(1, self.nx, self.ny)),
                #         ], axis=0),
                #     # jnp.asarray([8e3, 6e5, 0.5])
                #     # jnp.asarray([1e4, 7e4, 0.5])
                #     jnp.asarray([1e3, 5e5, 0.5])
                # )
                # output = jnp.permute_dims(output, (1, 2, 0)).reshape(
                #     self.nx * self.ny, -1
                # )
                # Q_pred_neural, _ = jnp.linalg.qr(output, mode='reduced')

                # self.vectorized_nksolver.Arnoldi_iteration(
                #     x0=trial_plasma_psi.copy(),
                #     Q_pred=Q_pred_neural.copy(),
                #     R0=res0.copy(),
                #     F_function=self.vectorized_solver.F_function,
                #     args=args,
                #     step_size=2.5,
                #     scaling_with_n=-1.0,
                #     target_relative_unexplained_residual=0.3,
                #     max_n_directions=17,
                #     clip=10,
                #     true_solver=self.solver,
                #     vectorized_solver=self.vectorized_solver,
                # )
                # preconditioner = 1.0 * self.vectorized_nksolver.dx
                # update = 1.0 * self.vectorized_nksolver.dx

                # self.solver.nksolver.Arnoldi_iteration(
                #     x0=np.array(trial_plasma_psi.copy()),
                #     dx=np.array(starting_direction.copy()),
                #     # dx=np.array(preconditioner.copy()),
                #     R0=np.array(res0.copy()),
                #     F_function=self.solver.F_function,
                #     args=args,
                #     step_size=2.5,
                #     scaling_with_n=-1.0,
                #     target_relative_unexplained_residual= \
                #         0.1,
                #     max_n_directions=5,
                #     explore_all=True,
                #     clip=10,
                # )

                # # PCA test
                # F_data = self.solver.nksolver.G[:, :self.solver.nksolver.n_it + 1]
                # X_data = self.solver.nksolver.Qn[:, :self.solver.nksolver.n_it + 1]
                # F_data_mean = F_data.mean(axis=1)

                # P, _, _ = np.linalg.svd(F_data - F_data_mean[:, None], full_matrices=False)
                # Q, _, _ = np.linalg.svd(X_data - X_data.mean(axis=1, keepdims=True), full_matrices=False)
                # preconditioner = PCA_preconditioner(
                #     P, Q, F_data_mean, trial_plasma_psi.copy(), 
                #     res0.copy(), self.solver.F_function, args,
                #     )

                self.solver.nksolver.Arnoldi_iteration(
                    x0=np.array(trial_plasma_psi.copy()),
                    dx=np.array(starting_direction.copy()),
                    # dx=np.array(preconditioner.copy()),
                    R0=np.array(res0.copy()),
                    F_function=self.solver.F_function,
                    args=args,
                    step_size=2.5,
                    scaling_with_n=-1.0,
                    target_relative_unexplained_residual= \
                        0.1,
                    max_n_directions=16,
                    explore_all=False,
                    clip=10,
                )
                
                update = 1.0 * self.solver.nksolver.dx
                # breakpoint()
                print(f"Krylov subspace size: {self.solver.nksolver.n_it + 1}")
                print(f'R0 norm: {np.linalg.norm(res0):.4e}, update norm: {np.linalg.norm(update):.4e}')


            del_update = np.amax(update) - np.amin(update)
            if del_update / del_psi > max_rel_update_size:
                update *= np.abs(max_rel_update_size * del_psi / del_update)
            new_residual_flag = True
            num_update_reduce = 0

            fig, ax = plt.subplots(1, 2, figsize=(12, 10))
            psi = np.asarray(trial_plasma_psi).reshape(self.nx, self.ny) + \
                self.solver.tokamak_psi.reshape(self.nx, self.ny)
            levels = np.linspace(psi.min(), psi.max(), 50)
            ax[0].contour(eq.R, eq.Z, psi, levels=levels)
            ax[0].contour(
                eq.R, eq.Z, psi, levels=[profiles.psi_bndry], colors='r', linestyles='--'
                )
            if eq.tokamak.wall and len(eq.tokamak.wall.R):
                ax[0].plot(
                    list(eq.tokamak.wall.R) + [eq.tokamak.wall.R[0]],
                    list(eq.tokamak.wall.Z) + [eq.tokamak.wall.Z[0]],
                    "k",
                )
            if eq.tokamak.limiter and len(eq.tokamak.limiter.R):
                ax[0].plot(
                    list(eq.tokamak.limiter.R) + [eq.tokamak.limiter.R[0]],
                    list(eq.tokamak.limiter.Z) + [eq.tokamak.limiter.Z[0]],
                    "k--",
                    lw=0.5,
                )
            c1 = ax[1].imshow(update.reshape(self.nx, self.ny).T, origin='lower')
            fig.colorbar(c1, ax=ax[1])
            save_path = os.path.join('image', 'MAST-U', image_path)
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(save_path + '/iteration_' + str(iterations) + '.png')
            plt.close()

            while new_residual_flag:

                    # n_trial_plasma_psi = trial_plasma_psi + update
                    # new_res0 = self.vectorized_solver.F_function(
                    #     n_trial_plasma_psi[:, None], self.solver.tokamak_psi, profiles
                    # ).squeeze(1)

                n_trial_plasma_psi = trial_plasma_psi + update
                if picard_flag > 0: 
                    new_res0 = self.solver.F_function(
                        np.asarray(n_trial_plasma_psi), self.solver.tokamak_psi, profiles
                    )
                    # new_res0 = self.F_function(
                    #     jnp.asarray(n_trial_plasma_psi), self.solver.tokamak_psi, profiles
                    # )
                else:
                    new_res0 = self.solver.F_function(
                        np.asarray(n_trial_plasma_psi), self.solver.tokamak_psi, profiles
                    )
                    # new_res0 = self.F_function(
                    #     jnp.asarray(n_trial_plasma_psi), self.solver.tokamak_psi, profiles
                    # )
                new_norm_rel_change = self.solver.relative_norm_residual(
                    new_res0, n_trial_plasma_psi
                )
                new_rel_change, new_del_psi = self.solver.relative_del_residual(
                    new_res0, n_trial_plasma_psi
                )
                try:
                    new_residual_flag = False
                except:
                    update *= 0.75
                    num_update_reduce += 1
                    if num_update_reduce > 10:
                        reduced_failure = True
                        print(f'Reduced update failed !!')
                        break

            if new_norm_rel_change < 1.2 * self.solver.norm_rel_change[-1]:
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
                reduce_by = self.solver.relative_change / new_rel_change                       
                new_residual_flag = True
                num_update_reduce = 0
                while new_residual_flag:
                    n_trial_plasma_psi = trial_plasma_psi + update * reduce_by
                    res0 = self.solver.F_function(
                        np.asarray(n_trial_plasma_psi), self.solver.tokamak_psi, profiles
                    )
                    # res0 = np.asarray(self.vectorized_solver.F_function(
                    #     n_trial_plasma_psi[:, None], self.solver.tokamak_psi, profiles
                    # ).squeeze(1))
                    try:
                        new_residual_flag = False
                    except:
                        reduce_by *= 0.75
                        num_update_reduce += 1
                        if num_update_reduce > 10:
                            reduced_failure = True
                            print(f'Reduced update failed')
                            break
                        
                
                starting_direction = np.copy(res0)
                trial_plasma_psi = n_trial_plasma_psi.copy()
                norm_rel_change = self.solver.relative_norm_residual(
                    res0, trial_plasma_psi
                )
                rel_change, del_psi = self.solver.relative_del_residual(
                    res0, trial_plasma_psi
                )
                if rel_change < self.solver.best_relative_change:
                    self.solver.best_relative_change = 1.0 * rel_change
                    self.solver.best_psi = np.copy(trial_plasma_psi)
            
            self.solver.relative_change = 1.0 * rel_change
            self.solver.norm_rel_change.append(norm_rel_change)
            print(f"relative error {rel_change:.4e} ")
            iterations += 1

        if self.solver.best_relative_change < rel_change:
            self.solver.relative_change = 1.0 * self.solver.best_relative_change
            trial_plasma_psi = np.copy(self.solver.best_psi)
            profiles.Jtor(
                _R_cpu, _Z_cpu,
                (self.solver.tokamak_psi + trial_plasma_psi).reshape(
                    self.nx, self.ny
                    ),
            )
        eq.plasma_psi = trial_plasma_psi.reshape(self.nx, self.ny).copy()

        # solver.port_critical(eq=eq, profiles=profiles)

        if rel_change > target_relative_tolerance:
            print(
                f"Forward static solve DID NOT CONVERGE. " \
                f"Tolerance {rel_change:.2e} "
                f"(vs. requested {target_relative_tolerance:.2e}) " \
                f"reached in {int(iterations)}/{int(max_solving_iterations)} iterations."
            )
        else:
            print(
                f"Forward static solve SUCCESS. Tolerance {rel_change:.2e} (vs. requested {target_relative_tolerance:.2e}) reached in {int(iterations)}/{int(max_solving_iterations)} iterations."
            )
        print(f"Total solving time: {datetime.now() - start_time}")
        return None
    
if __name__ == "__main__":
    sim = main(
        Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2,
        nR=65, nZ=129
    )
    # import cProfile
    # import pstats
    # import io
    
    # with cProfile.Profile() as pr:
    #     sim.simulation()
    # pr.dump_stats('profile/profile_MAST-U_deeponet.prof')
    # sim.simulation(image_path='simulation_numerical_nk', picard_handover=0.1)
    sim.simulation(image_path='simulation_nk_numerical_fixed_numerical', picard_handover=0.1)
        
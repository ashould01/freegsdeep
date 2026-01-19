import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
from freegsdeep.model import Integratednet_jax

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
            nx=nR, ny=nZ, key=jax.random.PRNGKey(0)
            )

        with open('logs/debug_resi_jax/model/model_39.eqx', 'rb') as f:
            solver_deep = eqx.tree_deserialise_leaves(f, solver_deep)
        
        self.solver_deep = vmap(solver_deep, in_axes=(0, 0, None, None, None))

    def F_function(
        self, solver: GSstaticsolver.NKGSsolver, plasma_psi: Array,
        tokamak_psi: Array, profiles: Array, neural_hangover: bool
        ) -> Array:
        solver.freeboundary(plasma_psi, tokamak_psi, profiles)
        if neural_hangover:
            rhs = jnp.asarray(solver.rhs)
            psi_bnd = jnp.zeros((len(self.boundary), 1))
            psi_bnd = psi_bnd.at[:self.nx, 0].set(rhs[:, 0]) 
            psi_bnd = psi_bnd.at[self.nx:2*self.nx, 0].set(rhs[:, -1])
            psi_bnd = psi_bnd.at[2*self.nx:2*self.nx+self.ny-2, 0].set(rhs[0, 1:-1])
            psi_bnd = psi_bnd.at[2*self.nx+self.ny-2:, 0].set(rhs[-1, 1:-1])

            rhs = rhs[None, :, :]
            residual = plasma_psi - (self.solver_deep(
                    self.residual[:, 0:1], self.residual[:, 1:2],
                    rhs, self.boundary, psi_bnd
            ))
            residual = np.asarray(residual)
            return residual
        else:
            residual = plasma_psi - solver.linear_GS_solver(
                solver.psi_boundary, solver.rhs
            ).reshape(-1)
            return residual
    
    def simulation(
        self, alpha_m: float = 1.8, alpha_n: float = 1.2
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
        Picard_handover = 0.1
        max_rel_update_size = 0.15

        eq = equilibrium_update.Equilibrium(
            tokamak=tokamak,
            Rmin=self.Rmin, Rmax=self.Rmax, Zmin=self.Zmin, Zmax=self.Zmax,
            nx=self.nx, ny=self.ny,
        )
        for idx2, key in enumerate(currents_dict.keys()):
            eq.tokamak.set_coil_current(
                coil_label=key,
                current_value=currents_dict[key]
                )
        profiles = jtor_update.ConstrainPaxisIp(
            eq, paxis, Ip, fvac, alpha_m, alpha_n
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

        solver.initial_rel_residual = 1.0 * rel_change
        iterations = 0
        reduced_failure = False
        while (rel_change > target_relative_tolerance) * (
            iterations < max_solving_iterations
        ) and reduced_failure == False:
            if rel_change > Picard_handover:
                print("Picard iteration: " + str(iterations))
                neural_flag = False
                if picard_flag < min(max_solving_iterations - 1, 3):
                    res0_2d = res0.reshape(self.nx, self.ny)
                    res0 = 0.5 * (res0_2d + res0_2d[:, ::-1]).reshape(-1)
                    picard_flag += 1
                else:
                    picard_flag = 1
                update = -1.0 * res0
            else:
                print("Neural operator iteration: " + str(iterations))
                picard_flag = False
                neural_flag = True
                if picard_flag < min(max_solving_iterations - 1, 3):
                    res0_2d = res0.reshape(self.nx, self.ny)
                    res0 = 0.5 * (res0_2d + res0_2d[:, ::-1]).reshape(-1)
                    picard_flag += 1
                else:
                    picard_flag = 1
                update = -1.0 * res0
                # print("Newton-Kylrov iteration: " + str(iterations))
                # picard_flag = False
                # neural_flag = False
                # solver.nksolver.Arnoldi_iteration(
                #     x0=trial_plasma_psi.copy(),
                #     dx=starting_direction.copy(),
                #     R0=res0.copy(),
                #     F_function=solver.F_function,
                #     args=args,
                #     step_size=2.5,
                #     scaling_with_n=-1.0,
                #     target_relative_unexplained_residual= \
                #         0.3,
                #     max_n_directions=16,
                #     clip=10,
                # )
                # update = 1.0 * solver.nksolver.dx

            del_update = np.amax(update) - np.amin(update)
            if del_update / del_psi > max_rel_update_size:
                update *= np.abs(max_rel_update_size * del_psi / del_update)
            new_residual_flag = True
            num_update_reduce = 0
            while new_residual_flag:

                n_trial_plasma_psi = trial_plasma_psi + update
                
                new_res0 = self.F_function(
                    solver, n_trial_plasma_psi,
                    solver.tokamak_psi, profiles, neural_hangover=neural_flag
                )
                try:
                    # n_trial_plasma_psi = trial_plasma_psi + update
                    # new_res0 = solver.F_function(
                    #     n_trial_plasma_psi, solver.tokamak_psi, profiles
                    # )
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
                            print(f'Reduced update failed')
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
            print(f"relative error {rel_change:.4e} ")
            iterations += 1

        if solver.best_relative_change < rel_change:
            solver.relative_change = 1.0 * solver.best_relative_change
            trial_plasma_psi = np.copy(solver.best_psi)
            profiles.Jtor(
                _R_cpu, _Z_cpu,
                (solver.tokamak_psi + trial_plasma_psi).reshape(
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
        
        return None
    
if __name__ == "__main__":
    sim = main(
        Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2,
        nR=65, nZ=129
    )
    sim.simulation(
        
    )
        
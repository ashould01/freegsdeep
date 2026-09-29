import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
from functools import partial
import torch
import jax
from jax import vmap
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from freegsnke import (
    build_machine,
    equilibrium_update,
)

import numpy as np
from freegsdeep.utils.typing import *
from freegsdeep.model import Integratednet_jax, XPlimnet
from freegsdeep.neural_boundary import (
    NeuralPlasmaBoundary,
    build_neural_jtor,
    build_neural_jtor_jit,
)
from freegsdeep.utils.utils import Greens_jax
from freegsdeep.utils.jtor import (
    ConstrainPaxisIp, _jtor_build_kernel,
    nksolver, _arnoldi_iteration_kernel
)
import matplotlib.pyplot as plt
from datetime import datetime

@eqx.filter_jit
def _F_function(
    R: jax.Array, Z: jax.Array, plasma_psi: jax.Array, tokamak_psi: jax.Array, 
    jtor: jax.Array, rhs_before_jtor: jax.Array, greenfunc: jax.Array, 
    nx: int, ny: int, residual: jax.Array, boundary: jax.Array,
    solver_deep: Integratednet_jax
    ) -> jax.Array:

    rhs = rhs_before_jtor * jtor
    psi_boundary = jnp.zeros_like(R)
    psi_bnd = jnp.tensordot(greenfunc, jtor, axes=([1, 2], [0, 1]))

    psi_boundary = psi_boundary.at[:, 0].set(psi_bnd[: nx])
    psi_boundary = psi_boundary.at[:, -1].set(psi_bnd[nx : 2 * nx])
    psi_boundary = psi_boundary.at[0, 1 : ny - 1].set(
        psi_bnd[2 * nx : 2 * nx + ny - 2]
        )
    psi_boundary = psi_boundary.at[-1, 1 : ny - 1].set(
        psi_bnd[2 * nx + ny - 2 :]
        )

    rhs = rhs.at[0, :].set(psi_boundary[0, :])
    rhs = rhs.at[:, 0].set(psi_boundary[:, 0])
    rhs = rhs.at[-1, :].set(psi_boundary[-1, :])
    rhs = rhs.at[:, -1].set(psi_boundary[:, -1])
    rhs_neural = rhs[None, :, :]
    # residual, boundary = residual.astype(jnp.float32), boundary.astype(jnp.float32)
    # psi_bnd, rhs_neural = psi_bnd.astype(jnp.float32), rhs[None, :, :].astype(jnp.float32)
    plasma_psi_neural = solver_deep(
        residual[:, 0:1], residual[:, 1:2], 
        rhs_neural, boundary, psi_bnd[:, None]
    ).reshape(nx, ny)

    residual_neural = plasma_psi - plasma_psi_neural.reshape(-1)
    return residual_neural


@eqx.filter_jit
def _nk_residual_kernel(
    R: jax.Array,
    Z: jax.Array,
    plasma_psi: jax.Array,
    tokamak_psi: jax.Array,
    rhs_before_jtor: jax.Array,
    greenfunc: jax.Array,
    residual: jax.Array,
    boundary: jax.Array,
    mask_inside_limiter: jax.Array,
    mask_limiter_cells: jax.Array,
    fine_cell_ids: jax.Array,
    fine_weights_r: jax.Array,
    fine_weights_z: jax.Array,
    d_r_d_z: jax.Array,
    ip: jax.Array,
    paxis: jax.Array,
    Ip: jax.Array,
    Raxis: jax.Array,
    alpha_m: jax.Array,
    alpha_n: jax.Array,
    los_top_k: int,
    solver_deep: Integratednet_jax,
    boundary_model: XPlimnet | None,
    boundary_limiter_mask: jax.Array,
) -> jax.Array:
    """Pure residual used as the Newton--Krylov operator.

    Every varying input is a JAX array (or an Equinox model pytree).  In
    particular, there is no ``profiles`` object mutation in this call, so it
    can be traced from ``_arnoldi_iteration_lax``.
    """
    nx, ny = R.shape
    psi = (plasma_psi + tokamak_psi).reshape(nx, ny)
    if boundary_model is None:
        supplied_psi_bndry = psi[0, 0]
        use_supplied_psi_bndry = jnp.array(False)
        jtor = _jtor_build_kernel(
            R, Z, psi, mask_inside_limiter, mask_limiter_cells, fine_cell_ids,
            fine_weights_r, fine_weights_z, d_r_d_z, ip, supplied_psi_bndry,
            use_supplied_psi_bndry, paxis, Ip, Raxis, alpha_m, alpha_n,
            nx + ny - 2, los_top_k,
        )[0]
    else:
        jtor, _, _ = build_neural_jtor(
            boundary_model, boundary_limiter_mask, R, Z, psi,
            paxis=paxis, Ip=Ip, Raxis=Raxis,
            alpha_m=alpha_m, alpha_n=alpha_n,
        )
    return _F_function(
        R=R,
        Z=Z,
        plasma_psi=plasma_psi,
        tokamak_psi=tokamak_psi,
        jtor=jtor,
        rhs_before_jtor=rhs_before_jtor,
        greenfunc=greenfunc,
        nx=nx,
        ny=ny,
        residual=residual,
        boundary=boundary,
        solver_deep=solver_deep,
    )


@partial(jax.jit, static_argnames=("nx", "ny"))
def _picard_update_kernel(
    residual: jax.Array,
    symmetrise: jax.Array,
    *,
    nx: int,
    ny: int,
) -> tuple[jax.Array, jax.Array]:
    """Build the Picard direction without eager elementwise JAX dispatches."""

    def symmetrised_residual(value):
        value_2d = value.reshape(nx, ny)
        return (0.5 * (value_2d + value_2d[:, ::-1])).reshape(-1)

    residual = jax.lax.cond(
        symmetrise,
        symmetrised_residual,
        lambda value: value,
        residual,
    )
    return residual, -residual


@jax.jit
def _clip_update_kernel(
    update: jax.Array,
    delta_psi: jax.Array,
    max_relative_update: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply the existing update-size limiter as one fused device kernel."""

    delta_update = jnp.amax(update) - jnp.amin(update)
    resized = delta_update / delta_psi > max_relative_update

    def resize(value):
        return value * jnp.abs(max_relative_update * delta_psi / delta_update)

    update = jax.lax.cond(resized, resize, lambda value: value, update)
    return update, resized


@jax.jit
def _candidate_metrics_kernel(
    current_residual: jax.Array,
    candidate_residual: jax.Array,
    candidate_psi: jax.Array,
    previous_norm_relative: jax.Array,
    previous_relative: jax.Array,
    restart_enabled: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate outer-loop acceptance quantities in one JIT compilation."""

    new_norm_relative = jnp.linalg.norm(candidate_residual) / jnp.linalg.norm(candidate_psi)
    new_delta_psi = jnp.amax(candidate_psi) - jnp.amin(candidate_psi)
    new_relative = (
        jnp.amax(candidate_residual) - jnp.amin(candidate_residual)
    ) / new_delta_psi
    accepted = new_norm_relative < 1.2 * previous_norm_relative
    residual_collinearity = jnp.sum(current_residual * candidate_residual) / (
        jnp.linalg.norm(current_residual) * jnp.linalg.norm(candidate_residual)
    )
    restart_direction = accepted & restart_enabled & (residual_collinearity > 0.9)
    reduction = previous_relative / new_relative
    return (
        new_norm_relative,
        new_delta_psi,
        new_relative,
        accepted,
        restart_direction,
        reduction,
    )

class main():

    def __init__(
        self, Rmin: float, Rmax: float, Zmin: float, Zmax: float, 
        nR: int, nZ: int, params: Tuple[float, float, float], 
        model_path: str,
        ) -> None:
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.nx = nR
        self.ny = nZ
        self.paxis, self.Ip, self.fvac = params

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

        self.solver_deep_no_vmap = Integratednet_jax(
            Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax, 
            nx=nR, ny=nZ, hidden_dim=30, key=jax.random.PRNGKey(0), freqs=5
            )
        def deserialise_cast_arrays(file, like_leaf):
            value = eqx.default_deserialise_filter_spec(file, like_leaf)

            if eqx.is_array(like_leaf):
                return value.astype(like_leaf.dtype)

            return value

        with open(
            model_path, 'rb'
            ) as f:
            self.solver_deep_no_vmap = eqx.tree_deserialise_leaves(
                f, self.solver_deep_no_vmap,
                filter_spec=deserialise_cast_arrays
                )
        self.solver_deep = vmap(self.solver_deep_no_vmap, in_axes=(0, 0, None, None, None))
        self.i = -1

    def F_function(
        self,
        plasma_psi: jax.Array,
        tokamak_psi: jax.Array,
        profiles: ConstrainPaxisIp,
        ) -> jax.Array:
        """Public residual, with optional phase timing for cold-start diagnosis."""
        psi = (plasma_psi + tokamak_psi).reshape(self.nx, self.ny)
        if self.boundary_model is None:
            jtor = profiles.Jtor(self.R, self.Z, psi)
        else:
            jtor = build_neural_jtor_jit(
                self.boundary_model,
                self.boundary_limiter_mask,
                self.R,
                self.Z,
                psi,
                profiles.paxis,
                profiles.Ip,
                profiles.Raxis,
                profiles.alpha_m,
                profiles.alpha_n,
            )
        jtor = jtor.reshape(self.nx, self.ny)

        residual = _F_function(
            R=self.R, Z=self.Z, plasma_psi=plasma_psi, tokamak_psi=tokamak_psi, 
            jtor=jtor, rhs_before_jtor=self.rhs_before_jtor, greenfunc=self.greenfunc, 
            nx=self.nx, ny=self.ny, residual=self.residual, boundary=self.boundary,
            solver_deep=self.solver_deep
        )
        return residual

    def F_function_pure(
        self, plasma_psi: jax.Array, tokamak_psi: jax.Array, profiles: ConstrainPaxisIp
    ) -> jax.Array:
        """Residual path suitable for the JIT-compiled Newton--Krylov loop.

        Unlike ``F_function``, this does not update fields on ``profiles``.
        Those diagnostics are still refreshed by the public path after the
        simulation has selected its final equilibrium.
        """
        limiter = profiles.limiter_handler
        return _nk_residual_kernel(
            R=self.R,
            Z=self.Z,
            plasma_psi=plasma_psi,
            tokamak_psi=tokamak_psi,
            rhs_before_jtor=self.rhs_before_jtor,
            greenfunc=self.greenfunc,
            residual=self.residual,
            boundary=self.boundary,
            mask_inside_limiter=limiter.mask_inside_limiter,
            mask_limiter_cells=limiter.mask_limiter_cells,
            fine_cell_ids=limiter.fine_cell_ids,
            fine_weights_r=limiter.fine_weights_r,
            fine_weights_z=limiter.fine_weights_z,
            d_r_d_z=limiter.dRdZ,
            ip=profiles.Ip,
            paxis=profiles.paxis,
            Ip=profiles.Ip,
            Raxis=profiles.Raxis,
            alpha_m=profiles.alpha_m,
            alpha_n=profiles.alpha_n,
            los_top_k=profiles.los_top_k,
            solver_deep=self.solver_deep,
            boundary_model=self.boundary_model,
            boundary_limiter_mask=self.boundary_limiter_mask,
        )

    def relative_norm_residual(
        self, residual: jax.Array, plasma_psi: jax.Array
        ) -> float:
        return jnp.linalg.norm(residual) / jnp.linalg.norm(plasma_psi)
    
    def relative_del_residual(
        self, residual: jax.Array, plasma_psi: jax.Array
        ) -> Tuple[float, float]:
        del_psi = jnp.amax(plasma_psi) - jnp.amin(plasma_psi)
        del_res = jnp.amax(residual) - jnp.amin(residual)
        return del_res / del_psi, del_psi
    
    def warmup(
        self, plasma_psi: jax.Array, tokamak_psi: jax.Array, starting_direction: jax.Array,
        res0: jax.Array, nk_residual: Callable, delta_psi: jax.Array,
        norm_relative: jax.Array, relative: jax.Array, max_relative_update: float,
        ) -> None:
        res0.block_until_ready()
        
        _arnoldi_iteration_kernel.lower(
            plasma_psi,
            starting_direction,
            res0,
            nk_residual,
            (tokamak_psi, ),
            l2_reg=1e-6,
            collinearity_reg=1e-6,
            step_size=2.5,
            scaling_with_n=-1.0,
            target_relative_unexplained_residual= \
                0.3,
            max_n_directions=16,
            explore_all=False,
            clip=10,
        ).compile()
        dummy_result = _arnoldi_iteration_kernel(
            plasma_psi,
            starting_direction,
            res0,
            nk_residual,
            (tokamak_psi,),
            l2_reg=1e-6,
            collinearity_reg=1e-6,
            step_size=2.5,
            scaling_with_n=-1.0,
            target_relative_unexplained_residual=0.3,
            max_n_directions=16,
            explore_all=False,
            clip=10,
        )
        dummy_result.dx.block_until_ready()

        # Execute the outer Picard kernels as well as compiling them.  The
        # first physical iteration can then reuse their executable cache.
        warm_residual, warm_update = _picard_update_kernel(
            res0, jnp.asarray(True), nx=self.nx, ny=self.ny
        )
        warm_update, _ = _clip_update_kernel(
            warm_update, delta_psi, max_relative_update
        )
        warm_metrics = _candidate_metrics_kernel(
            warm_residual,
            warm_residual,
            plasma_psi,
            norm_relative,
            relative,
            jnp.asarray(False),
        )
        warm_update.block_until_ready()
        warm_metrics[0].block_until_ready()
        return None

    def simulation(
        self, exp_path: str, image_path: str, 
        alpha_m: float = 1.8, alpha_n: float = 1.2, picard_handover: float = 0.1,
        target_relative_tolerance = 1e-6, los_top_k: int = 20,
        boundary_checkpoint: str | None = None,
        ) -> None:
        _R_cpu, _Z_cpu = np.meshgrid(
            np.linspace(self.Rmin, self.Rmax, self.nx),
            np.linspace(self.Zmin, self.Zmax, self.ny),
            indexing='ij'
        )
        self.exp_path = exp_path
        self.image_path = image_path
        
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

        max_solving_iterations = 200
        Picard_handover = picard_handover
        max_rel_update_size = 0.15

        eq = equilibrium_update.Equilibrium(
            tokamak=tokamak,
            Rmin=self.Rmin, Rmax=self.Rmax, Zmin=self.Zmin, Zmax=self.Zmax,
            nx=self.nx, ny=self.ny,
        )
        limiter_handler = eq.limiter_handler
        self.boundary_model = None
        self.boundary_limiter_mask = jnp.asarray(limiter_handler.mask_inside_limiter)
        if boundary_checkpoint is not None:
            boundary_predictor = NeuralPlasmaBoundary.from_checkpoint(
                boundary_checkpoint,
                self.nx,
                self.ny,
                self.boundary_limiter_mask,
            )
            self.boundary_model = boundary_predictor.model
        self.nksolver = nksolver(
            problem_dimension=self.nx * self.ny,
            l2_reg=1e-6,
            collinearity_reg=1e-6,
        )
        self.R_np = np.asarray(eq.R)
        self.Z_np = np.asarray(eq.Z)
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
        bndry_indices = jnp.asarray(bndry_indices)
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

        profiles = ConstrainPaxisIp(
            eq, self.paxis, self.Ip, self.fvac, alpha_m, alpha_n,
            los_top_k=los_top_k,
        )

        # Build this closure once per simulation.  The NK kernel treats the
        # callable as static and only ``tokamak_psi`` as dynamic array input.
        def nk_residual(plasma_psi, tokamak_psi_arg):
            return self.F_function_pure(plasma_psi, tokamak_psi_arg, profiles)

        with open('freegsnke/examples/data/simple_diverted_currents_PaxisIp.pk', 'rb') as f:
            currents_dict = pickle.load(f)
        for key in currents_dict.keys():
            eq.tokamak.set_coil_current(coil_label=key, current_value=currents_dict[key])
        
            
        picard_flag = 1 
        trial_plasma_psi = jnp.asarray(np.copy(eq.plasma_psi).reshape(-1)) 
        tokamak_psi = jnp.asarray(eq.tokamak.getPsitokamak(
            vgreen=eq._vgreen
            ).reshape(-1))


        control_trial_psi = False
        n_up = 0.0 + 4 * eq.solved
        self.image_index = 0
        while (control_trial_psi is False) and (n_up < 10):
            try:
                res0 = self.F_function(
                    trial_plasma_psi, tokamak_psi, profiles
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
            res0 = self.F_function(
                trial_plasma_psi, tokamak_psi, profiles
                )
            
            control_trial_psi = True
        
        norm_rel_change = self.relative_norm_residual(res0, trial_plasma_psi)
        rel_change, del_psi = self.relative_del_residual(res0, trial_plasma_psi)
        self.relative_change = 1.0 * rel_change
        self.norm_rel_change = [1.0 * norm_rel_change]
        self.best_relative_change = rel_change
        self.best_psi = trial_plasma_psi
        starting_direction = jnp.copy(res0)
        self.warmup(
            trial_plasma_psi,
            tokamak_psi,
            starting_direction,
            res0,
            nk_residual,
            del_psi,
            norm_rel_change,
            rel_change,
            max_rel_update_size,
            )

        self.initial_rel_residual = 1.0 * rel_change
        iterations = 0
        reduced_failure = False
        res0.block_until_ready()
        start_time = datetime.now()
        while (rel_change > target_relative_tolerance) * (
            iterations < max_solving_iterations
        ) and reduced_failure == False:
            if rel_change > Picard_handover and picard_flag:
                iteration_kind = "Picard"
                # print("Picard iteration: " + str(iterations))
                symmetrise = picard_flag < min(max_solving_iterations - 1, 3)
                if symmetrise:
                    picard_flag += 1
                else:
                    picard_flag = 1
                res0, update = _picard_update_kernel(
                    res0, symmetrise, nx=self.nx, ny=self.ny
                )
            else:
                iteration_kind = "Newton-Krylov"
                res0 = jnp.asarray(res0)
                # print("Newton-Krylov iteration: " + str(iterations))
                picard_flag = False
                # profiles = jtor_update.ConstrainPaxisIp(
                #     eq, self.paxis, self.Ip, self.fvac, alpha_m, alpha_n
                # )
                args = (tokamak_psi,)

                self.nksolver.Arnoldi_iteration(
                    x0=trial_plasma_psi,
                    dx=starting_direction,
                    R0=res0,
                    F_function=nk_residual,
                    args=args,
                    step_size=2.5,
                    scaling_with_n=-1.0,
                    target_relative_unexplained_residual= \
                        0.3,
                    max_n_directions=16,
                    explore_all=False,
                    clip=10,
                )
                
                update = 1.0 * self.nksolver.dx
                # print(f"Krylov subspace size: {self.nksolver.n_it + 1}")
                # print(f'R0 norm: {jnp.linalg.norm(res0):.4e}, update norm: {jnp.linalg.norm(update):.4e}')

            update, update_resized = _clip_update_kernel(
                update, del_psi, max_rel_update_size
            )
            # if bool(update_resized):
            #     print('Update too large, resized')
            new_residual_flag = True
            num_update_reduce = 0

            while new_residual_flag:
                n_trial_plasma_psi = trial_plasma_psi + update
                new_res0 = self.F_function(
                    n_trial_plasma_psi, tokamak_psi, profiles
                    )
                (
                    new_norm_rel_change,
                    new_del_psi,
                    new_rel_change,
                    update_accepted,
                    restart_direction,
                    reduce_by,
                ) = _candidate_metrics_kernel(
                    res0,
                    new_res0,
                    n_trial_plasma_psi,
                    self.norm_rel_change[-1],
                    self.relative_change,
                    jnp.asarray(picard_flag is False),
                )
                try:
                    new_residual_flag = False
                except:
                    update *= 0.75
                    num_update_reduce += 1
                    if num_update_reduce > 10:
                        reduced_failure = True
                        # print(f'Reduced update failed !!')
                        break

            if bool(update_accepted):
                trial_plasma_psi = n_trial_plasma_psi.copy()
                
                try:
                    res0 = 1.0 * new_res0
                    if bool(restart_direction):
                        starting_direction = jnp.sin(
                            jnp.linspace(0, 2*jnp.pi, self.nx)
                        * 1.5 * jnp.random.random()
                        )[:, None]
                        starting_direction = starting_direction * jnp.sin(
                                jnp.linspace(0, 2*jnp.pi, self.ny)
                                * 1.5 * jnp.random.random()
                            )[None, :]
                        starting_direction = starting_direction.reshape(-1)
                        starting_direction *= trial_plasma_psi
                    else:
                        starting_direction = jnp.copy(res0)
                except:
                    starting_direction = jnp.copy(res0)
                rel_change = 1.0 * new_rel_change
                norm_rel_change = 1.0 * new_norm_rel_change
                del_psi = 1.0 * new_del_psi
            else:
                new_residual_flag = True
                num_update_reduce = 0
                while new_residual_flag:
                    try:
                        n_trial_plasma_psi = trial_plasma_psi + update * reduce_by
                        res0 = self.F_function(
                            n_trial_plasma_psi, tokamak_psi, profiles
                        )
                        new_residual_flag = False
                    except:
                        reduce_by *= 0.75
                        num_update_reduce += 1
                        if num_update_reduce > 10:
                            reduced_failure = True
                            # print(f'Reduced update failed')
                            break
                        
                
                starting_direction = jnp.copy(res0)
                trial_plasma_psi = n_trial_plasma_psi.copy()
                norm_rel_change = self.relative_norm_residual(
                    res0, trial_plasma_psi
                )
                rel_change, del_psi = self.relative_del_residual(
                    res0, trial_plasma_psi
                )
                if rel_change < self.best_relative_change:
                    self.best_relative_change = 1.0 * rel_change
                    self.best_psi = jnp.copy(trial_plasma_psi)
            
            self.relative_change = 1.0 * rel_change
            self.norm_rel_change.append(norm_rel_change)
            # JAX dispatch is asynchronous.  Synchronise at the iteration
            # boundary so this duration belongs to this iteration alone.
            # print(f"relative error {rel_change:.4e} ")
            eq.plasma_psi = trial_plasma_psi.reshape(self.nx, self.ny).copy()
            # self.port_critical(eq=eq, profiles=profiles)
            # eq._profiles.opt = profiles.opt.copy()
            # eq._profiles.xpt = profiles.xpt.copy()
            # eq._profiles.psi_bndry = profiles.psi_bndry
            # eq._profiles.flag_limiter = profiles.flag_limiter
            iterations += 1

        if self.best_relative_change < rel_change:
            self.relative_change = 1.0 * self.best_relative_change
            trial_plasma_psi = np.copy(self.best_psi)
            final_psi = (tokamak_psi + trial_plasma_psi).reshape(self.nx, self.ny)
            if self.boundary_model is None:
                profiles.Jtor(_R_cpu, _Z_cpu, final_psi)
            else:
                profiles.jtor, profiles.jtorshape, prediction = build_neural_jtor(
                    self.boundary_model,
                    self.boundary_limiter_mask,
                    _R_cpu,
                    _Z_cpu,
                    final_psi,
                    paxis=profiles.paxis,
                    Ip=profiles.Ip,
                    Raxis=profiles.Raxis,
                    alpha_m=profiles.alpha_m,
                    alpha_n=profiles.alpha_n,
                )
                profiles.opt = jnp.array([[jnp.nan, jnp.nan, prediction.psi_axis]])
                profiles.xpt = jnp.full((1, 3), jnp.nan, dtype=final_psi.dtype)
                profiles.psi_bndry = prediction.psi_bndry
                profiles.diverted_core_mask = prediction.core_mask
                profiles.limiter_core_mask = prediction.core_mask
                profiles.flag_limiter = jnp.array(False)
                profiles.flood_fill_steps = jnp.array(0, dtype=jnp.int2)
        eq.plasma_psi = trial_plasma_psi.reshape(self.nx, self.ny).copy()
        # self.solver.port_critical(eq=eq, profiles=profiles)

        print(f"Total solving time: {datetime.now() - start_time}")
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
    
    def port_critical(self, eq, profiles):
        eq.xpt = np.copy(profiles.xpt)
        eq.opt = np.copy(profiles.opt)
        eq.psi_axis = eq.opt[0, 2]

        eq.psi_bndry = profiles.psi_bndry
        eq.flag_limiter = profiles.flag_limiter

        eq._current = np.sum(profiles.jtor) * self.dRdZ
        if type(profiles.jtor) is not np.ndarray:
            profiles.jtor = np.asarray(profiles.jtor)
            profiles.limiter_core_mask = np.asarray(profiles.limiter_core_mask)
            eq._profiles = profiles.copy()
            profiles.jtor = jnp.asarray(profiles.jtor)
            profiles.limiter_core_mask = jnp.asarray(profiles.limiter_core_mask)
        else:
            eq._profiles = profiles.copy()
        try:
            eq.tokamak_psi = self.tokamak_psi.reshape(self.nx, self.ny)
        except:
            pass

    def plot(self):
        pass
    
if __name__ == "__main__":
    sim = main(
        Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2,
        nR=65, nZ=129, params=(8e3, 6e5, 0.5),
        model_path='logs/260702_integratednet_dataset_more_limited/model/model_999.eqx'
    )
    sim.simulation(
        exp_path='MAST-U_jax_1',
        image_path='simulation_neural', picard_handover=0.1,
        target_relative_tolerance=1e-5
        )
        

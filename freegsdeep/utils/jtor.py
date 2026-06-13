from functools import partial

import os
import jax
import jax.numpy as jnp
from jax import vmap
jax.config.update("jax_enable_x64", True)
import equinox as eqx

from freegsdeep.model import XPlimnet
from freegsdeep.dataset import GSrhsdatasetMASTU_separatrix
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import beta
from freegsnke.freegsnke.limiter_func import Limiter_handler
from freegs4e.gradshafranov import Greens
from freegsdeep.model import XPlimnet
from freegsnke.freegsnke.copying import copy_into


class ConstrainPaxisIp():
    
    def __init__(self, eq, paxis, Ip, fvac, alpha_m=1.0, alpha_n=2.0, Raxis=1.0):

        self.set_masks(eq=eq)
        model = XPlimnet(nx=eq.nx, ny=eq.ny, key=jax.random.PRNGKey(0))
        load_name = {'name' : 'debug_jax_separatrix_4', 'epoch' : 299}
        load_path = os.path.join(
            'logs', load_name['name'], 'model', f'model_{load_name["epoch"]}.eqx'
            )
        with open(load_path, 'rb') as f:
            self.model = eqx.tree_deserialise_leaves(f, model)
        
        self.paxis = paxis
        self.Ip = Ip
        self.fvac = fvac
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n
        self.Raxis = Raxis

    def set_masks(self, eq):
        """Universal function to set all masks related to the limiter.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Specifies the domain properties
        """
        self.dR = eq.R_1D[1] - eq.R_1D[0]
        self.dZ = eq.Z_1D[1] - eq.Z_1D[0]
        self.dR_dZ = np.array([self.dR, self.dZ])
        self.R0Z0 = np.array([eq.R_1D[0], eq.Z_1D[0]])
        self.dRdZ = self.dR * self.dZ
        self.grid_points = np.concatenate(
            (eq.R[:, :, np.newaxis], eq.Z[:, :, np.newaxis]), axis=-1
        )
        self.nx, self.ny = np.shape(eq.R)
        self.eqRidx = np.tile(np.arange(self.nx)[:, np.newaxis], (1, self.ny))
        self.eqZidx = np.tile(np.arange(self.ny)[:, np.newaxis], (1, self.nx)).T
        self.idx_grid_points = np.concatenate(
            (self.eqRidx[:, :, np.newaxis], self.eqZidx[:, :, np.newaxis]), axis=-1
        ).reshape(-1, 2)

        self.limiter_handler = eq.limiter_handler

        # self.core_mask_limiter = eq.limiter_handler.core_mask_limiter

        self.mask_inside_limiter = eq.limiter_handler.mask_inside_limiter

        mask_outside_limiter = np.logical_not(eq.limiter_handler.mask_inside_limiter)
        # Note the factor 2 is not a typo: used in critical.inside_mask
        self.mask_outside_limiter = (2 * mask_outside_limiter).astype(float)

        self.limiter_mask_out = eq.limiter_handler.limiter_mask_out

        self.limiter_mask_for_plotting = (
            eq.limiter_handler.mask_inside_limiter
            + eq.limiter_handler.make_layer_mask(
                eq.limiter_handler.mask_inside_limiter, layer_size=1
            )
        ) > 0

        # set mask of the edge domain pixels
        self.edge_mask = np.zeros_like(eq.R)
        self.edge_mask[0, :] = self.edge_mask[:, 0] = self.edge_mask[-1, :] = (
            self.edge_mask[:, -1]
        ) = 1
        
    def Jtor_build(
        self, Jtor_part1, Jtor_part2, 
        core_mask_limiter, R, Z, psi,
        mask_outside_limiter, limiter_mask_out,
    ):
        psi_axis, diverted_core_mask, self.diverted_psi_bndry = Jtor_part1(
            psi, mask_outside_limiter
        )

        if diverted_core_mask is None:
            psi_bndry, limiter_core_mask, flag_limiter = (
                self.diverted_psi_bndry,
                None,
                False,
            )
        
        else:
            psi_bndry, limiter_core_mask, flag_limiter = core_mask_limiter(
                psi,
                self.diverted_psi_bndry,
                diverted_core_mask * self.mask_inside_limiter,
                limiter_mask_out,
            )
            if np.sum(limiter_core_mask * self.mask_inside_limiter) == 0:
                limiter_core_mask = diverted_core_mask * self.mask_inside_limiter
                psi_bndry = 1.0 * self.diverted_psi_bndry
        
        jtor = Jtor_part2(
            R, Z, psi, psi_axis, psi_bndry, limiter_core_mask
        )
        return np.asarray(jtor)
    
    def Jtor_part1(self, psi, mask_inside_limiter):
        psi = psi * mask_inside_limiter
        scaling = jnp.max(psi) - jnp.min(psi)
        psi_scale = (psi - jnp.min(psi)) / scaling
        psi_scale *= mask_inside_limiter
        psi_scale = psi_scale.reshape(1, *psi_scale.shape)
        psi_recon_pred = self.model(psi_scale)[0] * scaling + \
            jnp.min(psi)
        mask = psi > psi_recon_pred[1]

        return psi_recon_pred[0], mask, psi_recon_pred[1] 
        
    def Jtor_part2(self, R, Z, psi, psi_axis, psi_bndry, mask):
        if psi_bndry is None:
            psi_bndry = psi[0, 0]
        self.psi_bndry = psi_bndry
        self.psi_axis = psi_axis

        # grid sizes
        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # calculate normalised psi
        self.psi_norm = np.clip((psi - psi_axis) / (psi_bndry - psi_axis), 0.0, 1.0)

        # shape function
        jtorshape = (
            1.0 - self.psi_norm ** self.alpha_m
        ) ** self.alpha_n

        # if there is a masking function, use it
        if mask is not None:
            jtorshape *= mask
            self.mask = mask

        # now apply constraints to define constants
        self.shapeintegral = (
            beta(1.0 / self.alpha_m, 1.0 + self.alpha_n) / self.alpha_m
        )
        self.shapeintegral *= psi_bndry - psi_axis

        # integrate current density components
        self.IR = (
            np.sum(jtorshape * R / self.Raxis) * dR * dZ
        )  # romb(romb(jtorshape * R / self.Raxis)) * dR * dZ
        self.I_R = (
            np.sum(jtorshape * self.Raxis / R) * dR * dZ
        )  # romb(romb(jtorshape * self.Raxis / R)) * dR * dZ

        # find L scaling parameter and scaled beta
        self.LBeta0 = -self.paxis * self.Raxis / self.shapeintegral
        self.L = self.Ip / self.I_R - self.LBeta0 * (self.IR / self.I_R - 1)
        self.Beta0 = self.LBeta0 / self.L

        # calculate final toroidal current density
        Jtor = (
            self.L
            * (self.Beta0 * R / self.Raxis + (1 - self.Beta0) * self.Raxis / R)
            * jtorshape
        )

        # store parameters
        self.jtor = Jtor
        self.jtorshape = jtorshape
        return Jtor

    def Jtor(self, R, Z, psi):
        return self.Jtor_build(
            self.Jtor_part1, self.Jtor_part2, 
            self.limiter_handler.core_mask_limiter, R, Z, psi, 
            self.mask_outside_limiter, self.limiter_mask_out,
        )

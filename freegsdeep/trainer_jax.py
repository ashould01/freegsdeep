import os
tex_bin = "/data1/home/ahn40200393/texlive/2025/bin/x86_64-linux"
os.environ["PATH"] = tex_bin + ":" + os.environ.get("PATH", "")
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from matplotlib import colors
from matplotlib.ticker import FuncFormatter
import torch
from torch.utils.data import DataLoader, random_split, Subset
import math
import jax
from jax import (
    vmap, grad, jacfwd, jacrev, hessian
)
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
import optax
from freegsdeep.dataset import GSrhsdatasetMASTU
from freegsdeep.model import *
from jaxtyping import Float64
from freegsdeep.typing import *

class Trainer:
    
    def __init__(
        self, Rmin: float, Rmax: float, Zmin: float, Zmax: float,
        nR: int, nZ: int, batch_size: int, epoch: int, 
        len_data: Optional[int], data_load_path: Optional[str], 
        data_save_path: Optional[str]
        ) -> None:
        assert ((len_data != None) and (data_save_path != None)) or \
            ((len_data == None) and (data_load_path != None)), \
                "Either load data or generate data properly."
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.nR = nR
        self.nZ = nZ
        self.len_data = len_data
        self.batch_size = batch_size
        self.epoch = epoch
        self.data_load_path = data_load_path

        self.dataset = GSrhsdatasetMASTU(
            Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax,
            nR=nR, nZ=nZ, num=len_data, max_iter=150,
            load_path=data_load_path, save_path=data_save_path
        )

        self.resi = jnp.asarray(self.dataset.residual)
        self.bdry = jnp.asarray(self.dataset.boundary)

        self.model = Integratednet_jax(
            Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax,
            nx=nR, ny=nZ, key=jax.random.PRNGKey(0)
            )

    def train(
        self,
        save_name: str,
        load_name: Optional[Dict[int, str]] = None
        ) -> None:
        save_path = os.path.join('logs', save_name)
        os.makedirs(save_path, exist_ok=True)
        data_size = len(self.dataset)
        train_size = int(0.9 * data_size)
        test_size = data_size - train_size

        if load_name:
            load_epoch = load_name['epoch']
            self.model, train_dataset, test_dataset = self.load(load_name)
        else:
            load_epoch = 0
            train_dataset, test_dataset = random_split(
                self.dataset, [train_size, test_size]
                )
            del_idx = torch.where(
                self.dataset.idx[1:] != self.dataset.idx[:-1]
                )[0]
            new_idx = [
                idx for idx in test_dataset.indices if idx not in del_idx
                    ]
            test_dataset = Subset(test_dataset.dataset, new_idx)
        splits = {
            'train': train_dataset.indices,
            'test' : test_dataset.indices
        }
        torch.save(splits, os.path.join(save_path, 'split_idx.pt'))

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, 
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size * 5, shuffle=False,
        )
        print("Length for the dataloader :", len(train_dataloader))

        linesearch_fn = optax.scale_by_zoom_linesearch(max_linesearch_steps=15)
        # optim_resi = optax.lbfgs(
        #     memory_size=10, 
        #     linesearch=linesearch_fn
        #     )
        # optim_bdry = optax.lbfgs(
        #     memory_size=10, 
        #     linesearch=linesearch_fn
        #     )
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        n_resi = 400
        self.R_bdry_resi = jax.random.uniform(
            key=key1, shape=(n_resi, 1), minval=self.Rmin, maxval=self.Rmax
            )
        self.Z_bdry_resi = jax.random.uniform(
            key=key2, shape=(n_resi, 1), minval=self.Zmin, maxval=self.Zmax
            )
        optim_resi = optax.adam(learning_rate=1e-3)
        optim_bdry = optax.adam(learning_rate=1e-3)
        params_resi = eqx.filter(self.model.resi_net, eqx.is_array)
        params_bdry = eqx.filter(self.model.bdry_net, eqx.is_array)
        opt_state_resi = optim_resi.init(params_resi)
        opt_state_bdry = optim_bdry.init(params_bdry)
        for epoch in range(load_epoch, self.epoch):
            loss = psi_resi_loss = psi_bdry_loss = psi_true_loss = 0.0
            loss_list = []
            num_data = 0
            for _i, batch in enumerate(train_dataloader):
                idx, rhs, bdry, soln = batch
                rhs = jnp.asarray(rhs).reshape(-1, 1, self.nR, self.nZ)
                bdry = jnp.asarray(bdry)
                soln = jnp.asarray(soln).reshape(-1, 1, self.nR, self.nZ)
                self.model, opt_state_resi, opt_state_bdry, (
                    loss_resi, loss_bdry
                    ) = self.make_step(
                        self.model, opt_state_resi, opt_state_bdry,
                        rhs, bdry, optim_resi, optim_bdry
                        )
                loss_list.append([loss_resi, loss_bdry])
                if (_i + 1) % 200 == 0:
                    print(f'Step {_i+1} | Loss: resi {loss_resi:.6e} + '\
                        f'bdry {loss_bdry:.6e} = {loss_resi+loss_bdry:.6e}'
                        )
                num_data += rhs.shape[0]

            loss_list = np.array(loss_list)
            loss = np.sum(loss_list, axis=0) / num_data
            print(
                f"Epoch {epoch} train | " \
                f"Loss: resi {jnp.sqrt(loss[0]):.6e} + "\
                f"bdry {jnp.sqrt(loss[1]):.6e} = " \
                f"{loss[0]+loss[1]:.6e}"
                )
        
            test_abs_loss = 0.0
            test_rel_loss = 0.0
            num_test = 0 
            for _i, batch in enumerate(test_dataloader):

                idx, rhs, bdry, soln = batch
                rhs = jnp.asarray(rhs).reshape(-1, 1, self.nR, self.nZ)
                bdry = jnp.asarray(bdry)[:, :, None]
                soln = jnp.asarray(soln)[:, :, None]
                predict = vmap(
                    vmap(self.model, in_axes=(0, 0, None, None, None)),
                    in_axes=(None, None, 0, None, 0)
                )(
                    self.resi[:, 0:1], self.resi[:, 1:2], rhs,
                    self.bdry, bdry
                    )
                test_abs_loss += jnp.sum(jnp.mean(
                    (predict - soln) ** 2,
                    axis=(1, 2)
                ))
                test_rel_loss += jnp.sum(jnp.mean(
                    (predict - soln) ** 2 / (soln ** 2 + 1e-6),
                    axis=(1, 2)
                ))
                num_test += rhs.shape[0]
                if _i == 0:   
                    self.image(epoch, predict[0], soln[0], save_path)

            test_abs_loss /= num_test 
            test_rel_loss /= num_test
            print(
                f'Epoch {epoch} test | abs : {np.sqrt(test_abs_loss)} ' \
                f'rel : {np.sqrt(test_rel_loss)}'
                )
            self.save(epoch, save_path)
                
    def zero_bdry(
        self, R: JaxArray, Z: JaxArray, rhs: JaxArray
        ) -> Tuple[JaxArray]:
        nonzero_idx = jnp.isclose(
            jnp.abs(rhs), jnp.zeros_like(rhs), atol=1e-6
            ).squeeze(1)
        R_grid, Z_grid = jnp.meshgrid(
            jnp.linspace(self.Rmin, self.Rmax, self.nR),
            jnp.linspace(self.Zmin, self.Zmax, self.nZ),
            indexing='ij'
        )
        R_center = (R_grid * nonzero_idx).sum(axis=1).sum(axis=1) / \
            nonzero_idx.sum(axis=1).sum(axis=1)
        Z_center = (Z_grid * nonzero_idx).sum(axis=1).sum(axis=1) / \
            nonzero_idx.sum(axis=1).sum(axis=1)
        R_center = R_center[:, None]
        Z_center = Z_center[:, None]
        pR = R_center - self.Rmin
        qR = self.Rmax - R_center
        pZ = Z_center - self.Zmin
        qZ = self.Zmax - Z_center
        R = R[None, :]
        Z = Z[None, :]
        b = jnp.where(
            ((R - self.Rmin) < 0.02) | ((self.Rmax - R) < 0.02) | \
            ((Z - self.Zmin) < 0.02) | ((self.Zmax - Z) < 0.02),
            jnp.array(0.0, dtype=jnp.float64),
            (R - self.Rmin) ** pR * (self.Rmax - R) ** qR * \
            (Z - self.Zmin) ** pZ * (self.Zmax - Z) ** qZ
        )
        db_dR = jnp.where(
            ((R - self.Rmin) < 0.02) | ((self.Rmax - R) < 0.02) | \
            ((Z - self.Zmin) < 0.02) | ((self.Zmax - Z) < 0.02),
            jnp.array(0.0, dtype=jnp.float64),
            ((pR * (R - self.Rmin) ** (pR - 1) * (self.Rmax - R) ** qR) - \
            (qR * (R - self.Rmin) ** pR * (self.Rmax - R) ** (qR - 1))) * \
            (Z - self.Zmin) ** pZ * (self.Zmax - Z) ** qZ, 
            )
        db_dZ = jnp.where(
            ((R - self.Rmin) < 0.02) | ((self.Rmax - R) < 0.02) | \
            ((Z - self.Zmin) < 0.02) | ((self.Zmax - Z) < 0.02),
            jnp.array(0.0, dtype=jnp.float64),
            ((pZ * (Z - self.Zmin) ** (pZ - 1) * (self.Zmax - Z) ** qZ) - \
            (qZ * (Z - self.Zmin) ** pZ * (self.Zmax - Z) ** (qZ - 1))) * \
            (R - self.Rmin) ** pR * (self.Rmax - R) ** qR,
            )
        d2b_dR2 = jnp.where(
            ((R - self.Rmin) < 0.02) | ((self.Rmax - R) < 0.02) | \
            ((Z - self.Zmin) < 0.02) | ((self.Zmax - Z) < 0.02),
            jnp.array(0.0, dtype=jnp.float64),
            ((pR * (pR - 1) * (R - self.Rmin) ** (pR - 2) * (self.Rmax - R) ** qR) + \
            (qR * (qR - 1) * (R - self.Rmin) ** pR * (self.Rmax - R) ** (qR - 2)) - \
            (2 * pR * qR * (R - self.Rmin) ** (pR - 1) * (self.Rmax - R) ** (qR - 1))) * \
            (Z - self.Zmin) ** pZ * (self.Zmax - Z) ** qZ,
            )
        d2b_dZ2 = jnp.where(
            ((R - self.Rmin) < 0.02) | ((self.Rmax - R) < 0.02) | \
            ((Z - self.Zmin) < 0.02) | ((self.Zmax - Z) < 0.02),
            jnp.array(0.0, dtype=jnp.float64),
            ((pZ * (pZ - 1) * (Z - self.Zmin) ** (pZ - 2) * (self.Zmax - Z) ** qZ) + \
            (qZ * (qZ - 1) * (Z - self.Zmin) ** pZ * (self.Zmax - Z) ** (qZ - 2)) - \
            (2 * pZ * qZ * (Z - self.Zmin) ** (pZ - 1) * (self.Zmax - Z) ** (qZ - 1))) * \
            (R - self.Rmin) ** pR * (self.Rmax - R) ** qR,
            )
        b_max = pR ** pR * qR ** qR * pZ ** pZ * qZ ** qZ
        return b / b_max, db_dR / b_max, db_dZ / b_max, d2b_dR2 / b_max, d2b_dZ2 / b_max
                
    def loss_resi(
        self, params: JaxArray, static, rhs: Float64[JaxArray, "batch nx*ny"]
        ) -> Float64[JaxArray, ""]:
        
        model = eqx.combine(params, static)

        scaling = rhs.max(axis=(2, 3), keepdims=True) - \
            rhs.min(axis=(2, 3), keepdims=True)
        rhs = rhs / scaling

        R = self.resi[:, 0:1] # (nx*ny, )
        Z = self.resi[:, 1:2] # (nx*ny, )
        y = vmap(vmap(
            model, in_axes=(0, 0, None)
            ), in_axes=(None, None, 0))(R, Z, rhs)
        y = y.squeeze(2) # (B, nx*ny)
        
        grad_point_single_R = jacfwd(model.trunk, 0)
        grad_point_single_Z = jacfwd(model.trunk, 1)
        grad_point_single_RR = jacfwd(grad_point_single_R, 0)
        grad_point_single_ZZ = jacfwd(grad_point_single_Z, 1)
        grad_point_single_h = jacrev(model.output_mlp, 0)
        hessian_point_single_h = hessian(model.output_mlp, 0)
        
        output_branch = vmap(model.branch)(rhs)
        output_trunk = vmap(model.trunk)(R, Z)
        dy_dh = vmap(
            vmap(grad_point_single_h, in_axes=(0, None)),
            in_axes=(None, 0))(output_trunk, output_branch) 
        # (B, nx*ny, 1, 15)
        d2y_dh2 = vmap(
            vmap(hessian_point_single_h, in_axes=(0, None)),
            in_axes=(None, 0))(output_trunk, output_branch) 
        # (B, nx*ny, 1, 15, 15)
        dh_dr = vmap(grad_point_single_R)(R, Z)
        # (nx*ny, 15, 1)
        dh_dz = vmap(grad_point_single_Z)(R, Z)
        # (nx*ny, 15, 1)
        dh2_dr2 = vmap(grad_point_single_RR)(R, Z)
        # (nx*ny, 15, 1, 1)
        dh2_dz2 = vmap(grad_point_single_ZZ)(R, Z)
        # (nx*ny, 15, 1, 1)
        dy_dr = jnp.einsum('bxij, xji -> bx', dy_dh, dh_dr)
        # (B, nx*ny)
        dy_dz = jnp.einsum('bxij, xji -> bx', dy_dh, dh_dz)
        # (B, nx*ny)
        d2y_dr2 = jnp.einsum('bxkij, xik, xjk -> bx', d2y_dh2, dh_dr, dh_dr) + \
            jnp.einsum('bxij, xjik -> bx', dy_dh, dh2_dr2)
        # (B, nx*ny)
        d2y_dz2 = jnp.einsum('bxkij, xik, xjk -> bx', d2y_dh2, dh_dz, dh_dz) + \
            jnp.einsum('bxij, xjik -> bx', dy_dh, dh2_dz2)
        # (B, nx*ny)
        
        b, db_dR, db_dZ, d2b_dR2, d2b_dZ2 = self.zero_bdry(
            R.squeeze(1), Z.squeeze(1), rhs
            )
        # (B, nx*ny)
        dpsi_dr = dy_dr * b + y * db_dR
        d2psi_dr2 = d2y_dr2 * b + 2 * dy_dr * db_dR + y * d2b_dR2
        d2psi_dz2 = d2y_dz2 * b + 2 * dy_dz * db_dZ + y * d2b_dZ2
        psi_resi_loss = jnp.sum(jnp.mean((
            d2psi_dr2 + d2psi_dz2 - dpsi_dr / R[None, :, 0] - \
                rhs.reshape(rhs.shape[0], -1)
        ) ** 2, axis=1))

        return psi_resi_loss
    
    def loss_bdry(
        self, params: JaxArray, static, bdry_value: Float64[JaxArray, "2*nx+2*ny 1"]
        ) -> float:
        
        model = eqx.combine(params, static)

        scaling_bdry = bdry_value.max(axis=1, keepdims=True) - \
            bdry_value.min(axis=1, keepdims=True)
        bdry_value = bdry_value / scaling_bdry
        bdry_value = bdry_value[:, :, None]

        grad_point_single_R = jacrev(model, argnums=0)
        grad_point_single_Z = jacrev(model, argnums=1)
        grad_point_single_RR = jacrev(grad_point_single_R, argnums=0)
        grad_point_single_ZZ = jacrev(grad_point_single_Z, argnums=1)
        dpsi_dr = vmap(
            vmap(grad_point_single_R, in_axes=(0, 0, None, None)),
            in_axes=(None, None, None, 0)
            )(
                self.R_bdry_resi, self.Z_bdry_resi,
                self.bdry, bdry_value
                )[:, :, 0, 0, 0]
        d2psi_dr2 = vmap(
            vmap(grad_point_single_RR, in_axes=(0, 0, None, None)),
            in_axes=(None, None, None, 0)
            )(
                self.R_bdry_resi, self.Z_bdry_resi,
                self.bdry, bdry_value
                )[:, :, 0, 0, 0, 0]
        d2psi_dz2 = vmap(
            vmap(grad_point_single_ZZ, in_axes=(0, 0, None, None)),
            in_axes=(None, None, None, 0)
            )(
                self.R_bdry_resi, self.Z_bdry_resi,
                self.bdry, bdry_value
                )[:, :, 0, 0, 0, 0]
        psi_resi_loss = jnp.sum(jnp.mean((
            d2psi_dr2 - dpsi_dr / self.R_bdry_resi[None, :, 0] + d2psi_dz2
            ) ** 2, axis=1))

        psi_bdry = vmap(
            vmap(model, in_axes=(0, 0, None, None)),
            in_axes=(None, None, None, 0)
            )(self.bdry[:, 0:1], self.bdry[:, 1:2], self.bdry, bdry_value)
        psi_bdry = psi_bdry[:, :, :, 0]
        psi_bdry_loss = jnp.sum(jnp.mean((psi_bdry - bdry_value) ** 2, axis=(1, 2)))
        return psi_resi_loss + 30 * psi_bdry_loss, (psi_resi_loss, psi_bdry_loss)
        # return psi_resi_loss, (psi_resi_loss, psi_bdry_loss)

    @eqx.filter_jit
    def make_step(
        self, 
        model: Integratednet_jax,
        opt_state_resi: optax.OptState,
        opt_state_bdry: optax.OptState,
        rhs: Float64[JaxArray, "B 1 nx ny"],
        bdry: Float64[JaxArray, "B 2*nx+2*ny 1"],
        optim_resi: optax.GradientTransformation,
        optim_bdry: optax.GradientTransformation,
        ) -> Tuple[
            Integratednet_jax, optax.OptState, optax.OptState, Tuple[float, float]
            ]:

        params_resi, static_resi = eqx.partition(
            model.resi_net, eqx.is_array
        )
        def value_fn_resi(param: JaxArray) -> float:
            return self.loss_resi(param, static_resi, rhs) 
        value_resi, grads_resi = eqx.filter_value_and_grad(
            value_fn_resi
            )(params_resi)
        grads_resi = jax.grad(value_fn_resi)(params_resi)
        updates_resi, opt_state_resi = optim_resi.update(
            grads_resi, opt_state_resi, params_resi, 
        )
        new_resi_net = eqx.apply_updates(
            model.resi_net, updates_resi, 
        )
        model = eqx.tree_at(
            lambda model: model.resi_net, model, 
            new_resi_net
        )
        
        params_bdry, static_bdry = eqx.partition(
            model.bdry_net, eqx.is_array
        )
        def value_fn_bdry(param: JaxArray) -> float:
            value_bdry, _ = \
                self.loss_bdry(param, static_bdry, bdry) 
            return value_bdry
        value_bdry, grads_bdry = eqx.filter_value_and_grad(
            value_fn_bdry
        )(params_bdry)
        _, (loss_bdry_resi, loss_bdry_bdry) = self.loss_bdry(
            params_bdry, static_bdry, bdry
            )

        updates_bdry, opt_state_bdry = optim_bdry.update(
            grads_bdry, opt_state_bdry, params_bdry, 
        )
        new_bdry_net = eqx.apply_updates(
            model.bdry_net, updates_bdry, 
        )
        model = eqx.tree_at(
            lambda model: model.bdry_net, model, 
            new_bdry_net
        )
        # jax.debug.print("resi: {x}", x=value_resi)
        # jax.debug.print("bdry_resi: {x}", x=loss_bdry_resi)
        # jax.debug.print("bdry_bdry: {y}", y=loss_bdry_bdry)
        return model, opt_state_resi, opt_state_bdry, (value_resi, value_bdry)
            
    
    def load(self, load_name: Dict[int, str]) -> None:
        load_path = os.path.join(
            'logs', load_name['name'], 'model', f'model_{load_name["epoch"]}.eqx'
            )
        load_idx_path = os.path.join(
            'logs', load_name['name']
        )
        dataset_idx = torch.load(os.path.join(
            load_idx_path, 'split_idx.pt'
        ), weights_only=True)
        train_dataset = Subset(self.dataset, dataset_idx['train'])
        test_dataset = Subset(self.dataset, dataset_idx['test'])
        with open(load_path, 'rb') as f:
            return eqx.tree_deserialise_leaves(
                f, self.model
                ), train_dataset, test_dataset
            
            
    def save(self, epoch: int, save_path: str) -> None:
        save_path = os.path.join(save_path, 'model')
        os.makedirs(
            save_path, exist_ok=True
        )
        save_path = os.path.join(
            save_path, f'model_{epoch}.eqx'
        )
        with open(save_path, 'wb') as f:
            eqx.tree_serialise_leaves(f, self.model)

    def image(
        self, epoch: int, predict: JaxArray, soln: JaxArray, save_path: str
    ) -> None:
        R = jnp.linspace(self.Rmin, self.Rmax, self.nR)
        Z = jnp.linspace(self.Zmin, self.Zmax, self.nZ)
        R, Z = jnp.meshgrid(R, Z, indexing='ij')
        
        predict = predict.reshape(self.nR, self.nZ)
        soln = soln.reshape(self.nR, self.nZ)
        extent = [self.Rmin, self.Rmax, self.Zmin, self.Zmax]
        
        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        cmax, cmin = soln.max(), soln.min()
        cdelta = 0.95 * cmax / 15 
        levels = jnp.arange(0.0, 0.95 * cmax, cdelta)
        cnorm = colors.Normalize(vmin=cmin, vmax=cmax)
        
        c0 = ax[0].imshow(
            predict.T, extent=extent, origin='lower', norm=cnorm, 
            aspect='equal'
            )
        cs0 = ax[0].contour(
            R, Z, predict, levels=levels, colors='white', linestyles='--'
        )
        ax[0].clabel(cs0, inline=True, fontsize=8, fmt='%.2f')
        cbar0 = fig.colorbar(c0, ax=ax[0])
        
        c1 = ax[1].imshow(
            soln.T, extent=extent, origin='lower', norm=cnorm,
            aspect='equal'
            )
        cs1 = ax[1].contour(
            R, Z, soln, levels=levels, colors='white', linestyles='--'
        )
        ax[1].clabel(cs1, inline=True, fontsize=8, fmt='%.2f')
        cbar1 = fig.colorbar(c1, ax=ax[1])
        
        c2 = ax[2].imshow(
            jnp.abs((predict - soln) / (soln + 1e-6)).T, extent=extent,
            origin='lower', aspect='equal',
            norm=colors.LogNorm(vmin=1e-6, vmax=1.0)
            )
        cbar2 = fig.colorbar(c2, ax=ax[2])
        save_path = os.path.join(save_path, 'images')
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(save_path, f'epoch_{epoch}.png'),
            bbox_inches='tight'
            )
        plt.close()
            
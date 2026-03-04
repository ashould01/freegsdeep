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
from freegsdeep.dataset import *
from freegsdeep.model import *
from jaxtyping import Float64
from freegsdeep.typing import *
from freegsdeep.utils import ellipk, ellipe, break_if_nan
from freegsnke import build_machine, equilibrium_update, limiter_func

class Trainer_f:
    
    def __init__(
        self, Rmin: float, Rmax: float, Zmin: float, Zmax: float,
        nR: int, nZ: int, batch_size: int, adam_epoch: int, lbfgs_epoch: int,
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
        self.adam_epoch = adam_epoch
        self.lbfgs_epoch = lbfgs_epoch

        self.dataset = GSrhsdatasetMASTU_f(
            Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax,
            nR=nR, nZ=nZ, num=len_data, max_iter=150,
            load_path=data_load_path, save_path=data_save_path
        )

        self.resi = jnp.asarray(self.dataset.residual)
        self.bdry = jnp.asarray(self.dataset.boundary)

    def train(
        self,
        save_name: str,
        load_name: Optional[Dict[int, str]] = None
        ) -> None:

        model = Integratednet_jax(
            Rmin=self.Rmin, Rmax=self.Rmax, Zmin=self.Zmin, Zmax=self.Zmax,
            nx=self.nR, ny=self.nZ, hidden_dim=15, key=jax.random.PRNGKey(0)
            )
        save_path, train_dataloader, test_dataloader, model, load_epoch = \
            self.preprocess(model, save_name, load_name)

        print("Length for the dataloader :", len(train_dataloader))

        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        n_resi = 400
        self.R_bdry_resi = jax.random.uniform(
            key=key1, shape=(n_resi, 1), minval=self.Rmin, maxval=self.Rmax
            )
        self.Z_bdry_resi = jax.random.uniform(
            key=key2, shape=(n_resi, 1), minval=self.Zmin, maxval=self.Zmax
            )
        print("Phase 1 : Adam optimization")
        optim_resi = optax.adam(learning_rate=1e-3)
        optim_bdry = optax.adam(learning_rate=1e-3)
        params_resi = eqx.filter(model.resi_net, eqx.is_array)
        params_bdry = eqx.filter(model.bdry_net, eqx.is_array)
        opt_state_resi = optim_resi.init(params_resi)
        opt_state_bdry = optim_bdry.init(params_bdry)
        for epoch in range(load_epoch, self.adam_epoch):
            loss = 0.0
            loss_list = []
            num_data = 0
            for _i, batch in enumerate(train_dataloader):
                idx, rhs, bdry, soln = batch
                rhs = jnp.asarray(rhs).reshape(-1, 1, self.nR, self.nZ)
                bdry = jnp.asarray(bdry)
                model, opt_state_resi, opt_state_bdry, (
                    loss_resi, loss_bdry
                    ) = self.make_step(
                        model, opt_state_resi, opt_state_bdry,
                        rhs, bdry, optim_resi, optim_bdry,
                        lbfgs=False
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
                f"{jnp.sqrt(loss[0])+jnp.sqrt(loss[1]):.6e}"
                )
        
            test_abs_loss = 0.0
            test_rel_loss = 0.0
            num_test = 0 
            for _i, batch in enumerate(test_dataloader):

                idx, rhs, bdry, soln = batch
                rhs = jnp.asarray(rhs).reshape(-1, 1, self.nR, self.nZ)
                bdry = jnp.asarray(bdry)[:, :, None]
                soln = jnp.asarray(soln)[:, :, None]
                test_abs_loss_batch, test_rel_loss_batch = \
                    self.compute_test_loss(
                        model, rhs, bdry, soln
                )
                test_abs_loss += test_abs_loss_batch
                test_rel_loss += test_rel_loss_batch
                num_test += rhs.shape[0]

            test_abs_loss /= num_test 
            test_rel_loss /= num_test
            print(
                f'Epoch {epoch} test | abs : {np.sqrt(test_abs_loss)} ' \
                f'rel : {np.sqrt(test_rel_loss)}'
                )
            self.save(model, epoch, save_path)

        print("Phase 2 : L-BFGS optimization")
        linesearch_fn = optax.scale_by_zoom_linesearch(max_linesearch_steps=20)
        optim_resi = optax.lbfgs(
            memory_size=100, 
            linesearch=linesearch_fn
            )
        optim_bdry = optax.lbfgs(
            memory_size=100, 
            linesearch=linesearch_fn
            )
        opt_state_resi = optim_resi.init(params_resi)
        opt_state_bdry = optim_bdry.init(params_bdry)
        for epoch in range(load_epoch, self.lbfgs_epoch):
            loss = 0.0
            loss_list = []
            num_data = 0
            for _i, batch in enumerate(train_dataloader):
                idx, rhs, bdry, soln = batch
                rhs = jnp.asarray(rhs).reshape(-1, 1, self.nR, self.nZ)
                bdry = jnp.asarray(bdry)
                soln = jnp.asarray(soln).reshape(-1, 1, self.nR, self.nZ)
                model, opt_state_resi, opt_state_bdry, (
                    loss_resi, loss_bdry
                    ) = self.make_step(
                        model, opt_state_resi, opt_state_bdry,
                        rhs, bdry, optim_resi, optim_bdry, lbfgs=True
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
                f"{jnp.sqrt(loss[0])+jnp.sqrt(loss[1]):.6e}"
                )
        
            test_abs_loss = 0.0
            test_rel_loss = 0.0
            num_test = 0 
            for _i, batch in enumerate(test_dataloader):

                idx, rhs, bdry, soln = batch
                rhs = jnp.asarray(rhs).reshape(-1, 1, self.nR, self.nZ)
                bdry = jnp.asarray(bdry)[:, :, None]
                soln = jnp.asarray(soln)[:, :, None]
                test_abs_loss_batch, test_rel_loss_batch = \
                    self.compute_test_loss(
                        model, rhs, bdry, soln
                )
                test_abs_loss += test_abs_loss_batch
                test_rel_loss += test_rel_loss_batch
                num_test += rhs.shape[0]

            test_abs_loss /= num_test 
            test_rel_loss /= num_test
            print(
                f'Epoch {epoch} test | abs : {jnp.sqrt(test_abs_loss)} ' \
                f'rel : {jnp.sqrt(test_rel_loss)}'
                )
            self.save(model, epoch, save_path)

    def preprocess(
        self, model: eqx.Module, save_name: str,
        load_name: Optional[Dict[int, str]] = None
        ) -> Tuple[str, DataLoader, DataLoader, int]:
        save_path = os.path.join('logs', save_name)
        os.makedirs(save_path, exist_ok=True)
        data_size = len(self.dataset)
        train_size = int(0.9 * data_size)
        test_size = data_size - train_size

        if load_name:
            load_epoch = load_name['epoch']
            model, train_dataset, test_dataset = self.load(model, load_name)
        else:
            load_epoch = 0
            train_dataset, test_dataset = random_split(
                self.dataset, [train_size, test_size]
                )
            del_idx = torch.where(
                self.dataset.idx_f[1:] != self.dataset.idx_f[:-1]
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
        return save_path, train_dataloader, test_dataloader, model, load_epoch
    
    @eqx.filter_jit
    def compute_test_loss(
        self, 
        model: Integratednet_jax,
        rhs: Float64[JaxArray, "B 1 nx ny"],
        bdry: Float64[JaxArray, "B 2*nx+2*ny 1"],
        soln: Float64[JaxArray, "B 1 nx ny"],
        ) -> Tuple[float, float]:
                
        predict = vmap(
            vmap(model, in_axes=(0, 0, None, None, None)),
            in_axes=(None, None, 0, None, 0)
            )(
                self.resi[:, 0:1], self.resi[:, 1:2], rhs,
                self.bdry, bdry
            )
        
        test_abs_loss = jnp.sum(jnp.mean(
            (predict - soln) ** 2,
            axis=(1, 2)
        ))
        test_rel_loss = jnp.sum(jnp.mean(
            (predict - soln) ** 2 / (soln ** 2 + 1e-6),
            axis=(1, 2)
        ))
        
        return test_abs_loss, test_rel_loss

    def transformation(self, R: JaxArray, Z: JaxArray) -> Tuple[JaxArray, JaxArray]:
        x = (R - self.Rmin) / (self.Rmax - self.Rmin)
        y = (Z - self.Zmin) / (self.Zmax - self.Zmin)
        return x, y 

    def loss_resi(
        self, params: JaxArray, static, rhs: Float64[JaxArray, "batch nx*ny"],
        ) -> Float64[JaxArray, ""]:
        
        model = eqx.combine(params, static)

        scaling = rhs.max(axis=(2, 3), keepdims=True) - \
            rhs.min(axis=(2, 3), keepdims=True)
        rhs = rhs / scaling

        R = self.resi[:, 0:1] # (nx*ny, )
        Z = self.resi[:, 1:2] # (nx*ny, )
        x, y = self.transformation(R, Z)

        grad_x = jacfwd(model.trunk, 0)
        grad_y = jacfwd(model.trunk, 1)
        grad_xx = jacfwd(grad_x, 0)
        grad_yy = jacfwd(grad_y, 1)
        grad_h = jacrev(model.output_mlp, 0)
        hessian_h = hessian(model.output_mlp, 0)
        
        output_branch = vmap(model.branch)(rhs)
        output_trunk = vmap(model.trunk)(x, y)
        do_dh = vmap(
            vmap(grad_h, in_axes=(0, None)),
            in_axes=(None, 0))(output_trunk, output_branch) 
        # (B, nx*ny, 1, 15)
        d2o_dh2 = vmap(
            vmap(hessian_h, in_axes=(0, None)),
            in_axes=(None, 0))(output_trunk, output_branch) 
        # (B, nx*ny, 1, 15, 15)
        dh_dx = vmap(grad_x)(x, y)
        # (nx*ny, 15, 1)
        dh_dy = vmap(grad_y)(x, y)
        # (nx*ny, 15, 1)
        dh2_dx2 = vmap(grad_xx)(x, y)
        # (nx*ny, 15, 1, 1)
        dh2_dy2 = vmap(grad_yy)(x, y)
        # (nx*ny, 15, 1, 1)
        do_dx = jnp.einsum('bxij, xji -> bx', do_dh, dh_dx)
        # (B, nx*ny)
        d2o_dx2 = jnp.einsum('bxkij, xik, xjk -> bx', d2o_dh2, dh_dx, dh_dx) + \
            jnp.einsum('bxij, xjik -> bx', do_dh, dh2_dx2)
        # (B, nx*ny)
        d2o_dy2 = jnp.einsum('bxkij, xik, xjk -> bx', d2o_dh2, dh_dy, dh_dy) + \
            jnp.einsum('bxij, xjik -> bx', do_dh, dh2_dy2)
        # (B, nx*ny)

        dx_dr = 1 / (self.Rmax - self.Rmin)
        dy_dz = 1 / (self.Zmax - self.Zmin)

        zero = jnp.zeros_like(x, dtype=jnp.float64)
        one = jnp.ones_like(x, dtype=jnp.float64)
        output_trunk_0y = vmap(model.trunk)(zero, y)
        output_trunk_1y = vmap(model.trunk)(one, y)
        output_trunk_x0 = vmap(model.trunk)(x, zero)
        output_trunk_x1 = vmap(model.trunk)(x, one)

        dl_dh_0y = vmap(
            vmap(grad_h, in_axes=(0, None)),
            in_axes=(None, 0))(output_trunk_0y, output_branch)
        dl_dh_1y = vmap(
            vmap(grad_h, in_axes=(0, None)),
            in_axes=(None, 0))(output_trunk_1y, output_branch)
        dl_dh_x0 = vmap(
            vmap(grad_h, in_axes=(0, None)),
            in_axes=(None, 0))(output_trunk_x0, output_branch)
        dl_dh_x1 = vmap(
            vmap(grad_h, in_axes=(0, None)),
            in_axes=(None, 0))(output_trunk_x1, output_branch)

        dh_dx_x0 = vmap(grad_x)(x, zero)
        dh_dx_x1 = vmap(grad_x)(x, one)
        dh_dy_0y = vmap(grad_y)(zero, y)
        dh_dy_1y = vmap(grad_y)(one, y)

        dl_dx_x0 = jnp.einsum('bxij, xji -> bx', dl_dh_x0, dh_dx_x0)
        dl_dx_x1 = jnp.einsum('bxij, xji -> bx', dl_dh_x1, dh_dx_x1)

        d2l_dh2_0y = vmap(
            vmap(hessian_h, in_axes=(0, None)), in_axes=(None, 0)
        )(output_trunk_0y, output_branch)
        d2l_dh2_1y = vmap(
            vmap(hessian_h, in_axes=(0, None)), in_axes=(None, 0)
        )(output_trunk_1y, output_branch)
        d2l_dh2_x0 = vmap(
            vmap(hessian_h, in_axes=(0, None)), in_axes=(None, 0)
        )(output_trunk_x0, output_branch)
        d2l_dh2_x1 = vmap(
            vmap(hessian_h, in_axes=(0, None)), in_axes=(None, 0)
        )(output_trunk_x1, output_branch)

        dh2_dx2_x0 = vmap(grad_xx)(x, zero)
        dh2_dx2_x1 = vmap(grad_xx)(x, one)
        dh2_dy2_0y = vmap(grad_yy)(zero, y)
        dh2_dy2_1y = vmap(grad_yy)(one, y)

        d2l_dx2_x0 = jnp.einsum(
            'bxkij, xik, xjk -> bx', d2l_dh2_x0, dh_dx_x0, dh_dx_x0
            ) + jnp.einsum(
                'bxij, xjik -> bx', dl_dh_x0, dh2_dx2_x0
            )
        d2l_dx2_x1 = jnp.einsum(
            'bxkij, xik, xjk -> bx', d2l_dh2_x1, dh_dx_x1, dh_dx_x1
            ) + jnp.einsum(
                'bxij, xjik -> bx', dl_dh_x1, dh2_dx2_x1
            )
        d2l_dy2_0y = jnp.einsum(
            'bxkij, xik, xjk -> bx', d2l_dh2_0y, dh_dy_0y, dh_dy_0y
            ) + jnp.einsum(
                'bxij, xjik -> bx', dl_dh_0y, dh2_dy2_0y
            )
        d2l_dy2_1y = jnp.einsum(
            'bxkij, xik, xjk -> bx', d2l_dh2_1y, dh_dy_1y, dh_dy_1y
            ) + jnp.einsum(
                'bxij, xjik -> bx', dl_dh_1y, dh2_dy2_1y
            )
        
        o_f = lambda x, y, rhs: vmap(vmap(
            model, in_axes=(0, 0, None)
            ), in_axes=(None, None, 0))(x, y, rhs).squeeze(2)

        x_t = jnp.permute_dims(x, (1, 0))
        y_t = jnp.permute_dims(y, (1, 0))

        dl_dx = o_f(one, y, rhs) - o_f(zero, y, rhs) + (1 - y_t) * (
            o_f(zero, zero, rhs) - o_f(one, zero, rhs)
        ) + y_t * (o_f(one, one, rhs) - o_f(zero, one, rhs)) + \
            (1 - y_t) * dl_dx_x0 + y_t * dl_dx_x1
        d2l_dx2 = (1 - y_t) * d2l_dx2_x0 + y_t * d2l_dx2_x1
        d2l_dy2 = (1 - x_t) * d2l_dy2_0y + x_t * d2l_dy2_1y

        dpsi_dR = (do_dx - dl_dx) * dx_dr
        d2psi_dR2 = (d2o_dx2 - d2l_dx2) * dx_dr ** 2
        d2psi_dZ2 = (d2o_dy2 - d2l_dy2) * dy_dz ** 2

        psi_loss = jnp.sum(jnp.mean((
            d2psi_dR2 + d2psi_dZ2 - dpsi_dR / R[None, :, 0] - \
                rhs.reshape(rhs.shape[0], -1)
        ) ** 2, axis=1))

        return psi_loss

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
        lbfgs: bool,
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
        if lbfgs:
            updates_resi, opt_state_resi = optim_resi.update(
                grads_resi, opt_state_resi, params_resi, 
                value=value_resi, grad=grads_resi, value_fn=value_fn_resi
            )
        else:
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
        if lbfgs:
            updates_bdry, opt_state_bdry = optim_bdry.update(
                grads_bdry, opt_state_bdry, params_bdry, 
                value=value_bdry, grad=grads_bdry, value_fn=value_fn_bdry
            )
        else:
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
        return model, opt_state_resi, opt_state_bdry, (value_resi, value_bdry)
    
    def load(self, model: eqx.Module, load_name: Dict[int, str]) -> None:
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
                f, model
                ), train_dataset, test_dataset
            
            
    def save(self, model: eqx.Module, epoch: int, save_path: str) -> None:
        save_path = os.path.join(save_path, 'model')
        os.makedirs(
            save_path, exist_ok=True
        )
        save_path = os.path.join(
            save_path, f'model_{epoch}.eqx'
        )
        with open(save_path, 'wb') as f:
            eqx.tree_serialise_leaves(f, model)

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

class Trainer_g:
    
    def __init__(
        self, Rmin: float, Rmax: float, Zmin: float, Zmax: float,
        nR: int, nZ: int, batch_size: int, epoch: int, len_data: Optional[int],
        data_load_path: Optional[str], data_save_path: Optional[str],
        use_nk_loss: bool = True, use_g: bool = True,
        linear_solver_load_path_dict: Optional[Dict] = None,
        xplim_load_path_dict: Optional[Dict] = None,
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
        self.use_nk_loss = use_nk_loss
        self.use_g = use_g

        self.dataset = GSrhsdatasetMASTU_g(
            Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax,
            nR=nR, nZ=nZ, num=len_data, max_iter=150,
            load_path=data_load_path, save_path=data_save_path
        )
        self.resi = jnp.asarray(self.dataset.residual)
        self.bdry = jnp.asarray(self.dataset.boundary)
        # R_1D = jnp.linspace(Rmin, Rmax, nR)
        # Z_1D = jnp.linspace(Zmin, Zmax, nZ)
        # bndry_indices = np.concatenate(
        #     [
        #         [(x, 0) for x in range(nR)],
        #         [(x, nZ - 1) for x in range(nR)],
        #         [(0, y) for y in np.arange(1, nZ - 1)],
        #         [(nR - 1, y) for y in np.arange(1, nZ - 1)],
        #     ]
        # )
        # R_1D_green = jnp.asarray([R_1D[i] for i, j in bndry_indices])[:, None, None]
        # Z_1D_green = jnp.asarray([Z_1D[j] for i, j in bndry_indices])[:, None, None]
        # R_green = self.resi[None, :, 0].reshape(1, nR, nZ)
        # Z_green = self.resi[None, :, 1].reshape(1, nR, nZ)

        # self.dR = (self.Rmax - self.Rmin) / (self.nR - 1)
        # self.dZ = (self.Zmax - self.Zmin) / (self.nZ - 1)
        # self.beta = jax.scipy.special.beta(1.0 / 1.8, 1.0 + 1.2)
        # self.mu0 = 4e-7 * jnp.pi
        # self.rhs_before_jtor = - self.mu0 * self.resi[:, 0].reshape(1, 1, nR, nZ)
        # k2 = 4.0 * R_green * R_1D_green / (
        #     (R_green + R_1D_green) ** 2 + (Z_green - Z_1D_green) ** 2
        #     )
        # k2 = jnp.clip(k2, 1e-10, 1.0 - 1e-10)
        # greenfunc = self.mu0 / (2.0 * jnp.pi) * jnp.sqrt(R_green * R_1D_green) * (
        #     (2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2) 
        # ) / jnp.sqrt(k2)
        # zeros = np.ones_like(greenfunc)
        # zeros[
        #     np.arange(len(bndry_indices)), bndry_indices[:, 0], bndry_indices[:, 1]
        # ] = 0
        # self.greenfunc = greenfunc * jnp.asarray(zeros) * self.dR * self.dZ
        # tokamak = build_machine.tokamak(
        #     active_coils_path=f"freegsnke/machine_configs/MAST-U/MAST-U_like_active_coils.pickle",
        #     passive_coils_path=f"freegsnke/machine_configs/MAST-U/MAST-U_like_passive_coils.pickle",
        #     limiter_path=f"freegsnke/machine_configs/MAST-U/MAST-U_like_limiter.pickle",
        #     wall_path=f"freegsnke/machine_configs/MAST-U/MAST-U_like_wall.pickle",
        # )
        # eq = equilibrium_update.Equilibrium(
        #     tokamak=tokamak,      # provide tokamak object
        #     Rmin=0.1, Rmax=2.0,   # radial range
        #     Zmin=-2.2, Zmax=2.2,  # vertical range
        #     nx=65,                # number of grid points in the radial direction (needs to be of the form (2**n + 1) with n being an integer)
        #     ny=129,               # number of grid points in the vertical direction (needs to be of the form (2**n + 1) with n being an integer)
        #     # psi=plasma_psi
        # )  
        # limiter_handler = limiter_func.Limiter_handler(eq, tokamak.limiter)
        # limiter_handler.build_mask_inside_limiter()
        # self.mask = jnp.asarray(limiter_handler.mask_inside_limiter)

        # self.linear_solver = Integratednet_jax(
        #     Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax,
        #     nx=nR, ny=nZ, hidden_dim=15, key=jax.random.PRNGKey(0)
        # )
        # linear_solver_load_path = os.path.join(
        #     'logs', linear_solver_load_path_dict['name'],
        #     'model', f'model_{linear_solver_load_path_dict["epoch"]}.eqx'
        # )
        # with open(linear_solver_load_path, 'rb') as f:
        #     self.linear_solver = eqx.tree_deserialise_leaves(
        #         f, self.linear_solver
        #         )
        # self.vmap_linear_solver = vmap(vmap(
        #     self.linear_solver, in_axes=(0, 0, None, None, None),
        #     ), in_axes=(None, None, 0, None, 0))
        # self.xplimnet = XPlimnet(nx=nR, ny=nZ, key=jax.random.PRNGKey(0))
        # xplim_load_path = os.path.join(
        #     'logs', xplim_load_path_dict['name'],
        #     'model', f'model_{xplim_load_path_dict["epoch"]}.eqx'
        # )
        # with open(xplim_load_path, 'rb') as f:
        #     self.xplimnet = eqx.tree_deserialise_leaves(f, self.xplimnet)
        # self.vmap_xplimnet = vmap(self.xplimnet)

    def init_linear_weight(self, model: NKDeepONet):
        def trunc_init(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
            out, in_ = weight.shape
            return jax.random.truncated_normal(
                key, shape=(out, in_), lower=-10e-4, upper=10e-4
                )

        is_linear = lambda x: isinstance(x, eqx.nn.Linear)
        get_weights = lambda m: [
            x.weight for x in jax.tree_util.tree_leaves(
                m.branch_tokamak, is_leaf=is_linear
                ) if is_linear(x)
            ]
        weights = get_weights(model)
        new_weights = [
            trunc_init(weight, subkey) for weight, subkey in zip(
                weights, jax.random.split(jax.random.PRNGKey(0), len(weights))
                )]
        new_model = eqx.tree_at(get_weights, model, new_weights)
        return new_model

    def train(
        self,
        save_name: str,
        load_name: Optional[Dict[int, str]] = None,
        ) -> None:
        from jax.sharding import Mesh, PartitionSpec as P
        mesh = Mesh(jax.devices(), ('batch', ))
        sharding = jax.sharding.NamedSharding(mesh, P('batch'))
        sharding_model = jax.sharding.NamedSharding(mesh, P())

        if self.use_nk_loss:
            self.output_dim = 9 
        else:
            self.output_dim = 1

        # model = NKDeepONet(
        #     nx=self.nR, ny=self.nZ, hidden_dim=15, 
        #     tokamak_input_len=3, output_dim=self.output_dim, key=jax.random.PRNGKey(0)
        #     )
        # model = self.init_linear_weight(model)
        model = FNO2D(
            input_dim=3, output_dim=self.output_dim, channels_last_proj=128, num_constraints=3,
            modes1=16, modes2=16, width=32, depth=4
            )
        save_path, train_dataloader, test_dataloader, load_epoch = \
            self.preprocess(save_name, load_name)
            
        print("Length for the dataloader :", len(train_dataloader))

        grad_transf = optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_schedule(optax.linear_schedule(
                init_value=1e-3, end_value=1e-5,
                transition_steps=int(self.epoch * 0.8),
                transition_begin=int(self.epoch * 0.15),
                )),
            optax.scale(-1.0)
        )
        params = eqx.filter(model, eqx.is_array)
        opt_state = grad_transf.init(params)
        model, opt_state = eqx.filter_shard((model, opt_state), sharding_model)
        for epoch in range(load_epoch, self.epoch):
            loss = test_loss = 0.0
            num_data = num_data_test = 0
            for _i, batch in enumerate(train_dataloader):
                # idx, plasma_psi, tokamak_psi, constraint, update = batch
                idx, plasma_psi, tokamak_psi, constraint, _, G = batch
                plasma_psi = jnp.asarray(
                    plasma_psi.reshape(-1, 1, self.nR, self.nZ)
                )
                tokamak_psi = jnp.asarray(
                    tokamak_psi.reshape(-1, 1, self.nR, self.nZ)
                )
                constraint = jnp.asarray(constraint)
                # update = jnp.asarray(update)
                G = jnp.asarray(G)

                (model, opt_state), loss_batch = self.make_step(
                    model, opt_state, plasma_psi,
                    tokamak_psi, constraint, G, grad_transf,
                    sharding, sharding_model
                    )
                # (model, opt_state), loss_batch = self.make_step(
                #     model, opt_state, plasma_psi,
                #     tokamak_psi, constraint, G, grad_transf,
                #     sharding, sharding_model
                #     )
                if (_i + 1) % 20 == 0:
                    print(f'Step {_i+1} | Loss: {loss_batch:.6e}')
                loss += loss_batch
                num_data += len(idx)

            loss /= num_data
            print(
                f"Epoch {epoch} train | " \
                f"Loss: {jnp.sqrt(loss):.6e}"
                )
            if (epoch + 1) % 10 == 0:
                self.save(model, epoch, save_path)

            for _i, batch in enumerate(test_dataloader):
                idx, plasma_psi, tokamak_psi, constraint, Q, _ = batch
                plasma_psi = jnp.asarray(
                    plasma_psi.reshape(-1, 1, self.nR, self.nZ)
                )
                tokamak_psi = jnp.asarray(
                    tokamak_psi.reshape(-1, 1, self.nR, self.nZ)
                )
                constraint = jnp.asarray(constraint)
                Q = jnp.asarray(Q)
                # test_loss_batch = self.loss(
                #     params, static, plasma_psi, tokamak_psi, constraint
                #     )
                test_loss_batch = self.loss_data(
                    model, plasma_psi, tokamak_psi, constraint, Q,
                    sharding, sharding_model
                )
                test_loss += test_loss_batch
                num_data_test += len(idx)
            test_loss /= num_data_test
            print(
                f"Epoch {epoch} test | " \
                f"Loss: {jnp.sqrt(test_loss):.6e}"
                )

    def preprocess(
        self, save_name: str, load_name: Optional[Dict[int, str]] = None
        ) -> Tuple[str, DataLoader, DataLoader, int]:
        save_path = os.path.join('logs', save_name)
        os.makedirs(save_path, exist_ok=True)
        data_size = len(self.dataset)
        train_size = int(0.9 * data_size)
        test_size = data_size - train_size

        if load_name:
            load_epoch = load_name['epoch']
            model, train_dataset, test_dataset = self.load(model, load_name)
        else:
            load_epoch = 0
            train_dataset, test_dataset = random_split(
                self.dataset, [train_size, test_size]
                )
            del_idx = torch.where(
                self.dataset.idx_f[1:] != self.dataset.idx_f[:-1]
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
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
        )
        return save_path, train_dataloader, test_dataloader, load_epoch
    
    def jtor(
        self, plasma_psi: Float64[JaxArray, "batch 1 nx ny"],
        tokamak_psi: Float64[JaxArray, "batch 1 nx ny"],
        constraints: Float64[JaxArray, "batch n_params"]
        ) -> Float64[JaxArray, "batch nx*ny"]:
        R = self.resi[None, :, 0]

        psi = plasma_psi + tokamak_psi
        scale = psi.max(axis=(2, 3), keepdims=True) - psi.min(axis=(2, 3), keepdims=True)
        psi_scale = (psi - psi.min(axis=(2, 3), keepdims=True)) / \
            (psi.max(axis=(2, 3), keepdims=True) - psi.min(axis=(2, 3), keepdims=True))

        psi_recon = self.vmap_xplimnet(psi_scale)[0] * (scale.squeeze(3).squeeze(2)) + \
            psi.min(axis=(2, 3))
        psi_axis, psi_bndry = psi_recon[:, 0:1], psi_recon[:, 1:2]
        paxis = constraints[:, 0:1]
        Ip = constraints[:, 1:2]
        fvac = constraints[:, 2:3]

        plasma_psi = plasma_psi.reshape(-1, self.nR * self.nZ)
        tokamak_psi = tokamak_psi.reshape(-1, self.nR * self.nZ)
        psi_norm = (plasma_psi + tokamak_psi - psi_axis) / (
            psi_bndry - psi_axis + 1e-6
            )
        psi_norm = jnp.clip(psi_norm, 0.0, 1.0)
        mask = jax.nn.sigmoid((0.95 - psi_norm) / 0.025) * \
            jax.nn.sigmoid((self.mask.reshape(1, -1) - 0.5) / 0.05)
        # mask = jnp.where(psi_norm > 0.995, 0.0, 1.0) * self.mask.reshape(1, -1)
        jtor_shape = ((1.0 - psi_norm ** 1.8) ** 1.2) * mask
        shapeintegral = self.beta / 1.8 * (
            psi_bndry - psi_axis
        )
        IR = jnp.sum(jtor_shape * R) * self.dR * self.dZ
        I_R = jnp.sum(jtor_shape / R) * self.dR * self.dZ
        LBeta0 = -paxis / shapeintegral
        L = Ip / I_R - LBeta0 * (IR / I_R - 1)
        Beta0 = LBeta0 / L

        jtor = L * (
            Beta0 * R + (1 - Beta0) / R
            ) * jtor_shape
        return jtor
    
    @eqx.filter_jit
    def loss(
        self,
        params: JaxArray, static,
        plasma_psi: Float64[JaxArray, "batch 1 nx ny"],
        tokamak_psi: Float64[JaxArray, "batch 1 nx ny"],
        constraints: Float64[JaxArray, "batch n_params"]
        ) -> Float64[JaxArray, ""]:
        
        # (1) Boundary condition update after update toroidal plasma current 
        model = eqx.combine(params, static)
        psi = jnp.concatenate([plasma_psi, tokamak_psi], axis=1)
        
        vmap_model = vmap(
            vmap(model, in_axes=(0, 0, None, None)),
            in_axes=(None, None, 0, 0)
            )

        R = self.resi[:, 0:1] # (nx*ny, )
        Z = self.resi[:, 1:2] # (nx*ny, )

        update = vmap_model(R, Z, psi, constraints)
        norm = jnp.linalg.norm(update, axis=(1, 2), keepdims=True) + 1e-6
        norm_update = update / norm
        resi_jtor = self.jtor(
            plasma_psi.reshape(-1, 1, self.nR, self.nZ),
            tokamak_psi.reshape(-1, 1, self.nR, self.nZ), 
            constraints
        ).reshape(-1, self.nR, self.nZ)
        resi_psi_bnd = jnp.tensordot(
            resi_jtor, self.greenfunc, axes=([1, 2], [1, 2])
        )[:, :, None]
        resi_rhs = self.rhs_before_jtor * resi_jtor.reshape(-1, 1, self.nR, self.nZ)
        resi_psi_sol = self.vmap_linear_solver(
            self.resi[:, 0:1], self.resi[:, 1:2],
            resi_rhs,
            self.bdry,
            resi_psi_bnd
            )
        scale = jnp.linalg.norm(
            plasma_psi.reshape(-1, self.nR * self.nZ) - resi_psi_sol.reshape(-1, self.nR * self.nZ),
            axis=1, keepdims=True
        ) * 0.1

        update = norm_update * scale[:, None, :]
    
        jtor = self.jtor(
            plasma_psi.reshape(-1, 1, self.nR, self.nZ) + \
            update.reshape(-1, 1, self.nR, self.nZ),
            tokamak_psi,
            constraints
            ).reshape(-1, self.nR, self.nZ)
        psi_bnd = jnp.tensordot(
            jtor, self.greenfunc, axes=([1, 2], [1, 2])
        )[:, :, None]

        rhs = self.rhs_before_jtor * jtor.reshape(-1, 1, self.nR, self.nZ)

        # (2) Solve Grad-Shafranov equation w/ linear solver
        psi_sol = self.vmap_linear_solver(
            self.resi[:, 0:1], self.resi[:, 1:2],
            rhs,
            self.bdry,
            psi_bnd
            )

        loss = jnp.linalg.norm(jnp.where(
            rhs.max(axis=(2, 3)) - rhs.min(axis=(2, 3)) > 1e-6,
            plasma_psi.reshape(-1, self.nR * self.nZ) + \
            update.reshape(-1, self.nR * self.nZ) - \
            psi_sol.reshape(-1, self.nR * self.nZ),
            0.0
            ), axis=1).sum()
        return loss

    def loss_nk(
        self,
        params: JaxArray, static,
        plasma_psi: Float64[JaxArray, "batch 1 nx ny"],
        tokamak_psi: Float64[JaxArray, "batch 1 nx ny"],
        constraints: Float64[JaxArray, "batch n_params"],
        G: Float64[JaxArray, "batch nx*ny output_dim"]
        ) -> Float64[JaxArray, ""]:

        model = eqx.combine(params, static)
        psi = jnp.concatenate([plasma_psi, tokamak_psi], axis=1)
        
        # vmap_model = vmap(
        #     vmap(model, in_axes=(0, 0, None, None)),
        #     in_axes=(None, None, 0, 0)
        #     )
        # R = self.resi[:, 0:1] # (nx*ny, )
        # Z = self.resi[:, 1:2] # (nx*ny, )
        # output = vmap_model(R, Z, psi, constraints)

        vmap_model = vmap(model)
        output = vmap_model(psi, constraints)
        output = jnp.permute_dims(output, (0, 2, 3, 1)).reshape(
            -1, self.nR * self.nZ, self.output_dim
            )

        Q, _ = jnp.linalg.qr(output, mode='reduced')

        if self.use_g:
            n_G = jnp.linalg.norm(G, axis=1, keepdims=True)
            Gn = G / n_G
            invar_loss = Gn - jnp.matmul(jnp.matmul(
                Q,
                jnp.permute_dims(Q, (0, 2, 1))
                ), Gn)
        
        else:
            # (1) Boundary condition update after update toroidal plasma current 

            resi_jtor = self.jtor(
                plasma_psi.reshape(-1, 1, self.nR, self.nZ),
                tokamak_psi.reshape(-1, 1, self.nR, self.nZ), 
                constraints
            ).reshape(-1, self.nR, self.nZ)
            resi_psi_bnd = jnp.tensordot(
                resi_jtor, self.greenfunc, axes=([1, 2], [1, 2])
            )[:, :, None]
            resi_rhs = self.rhs_before_jtor * resi_jtor.reshape(-1, 1, self.nR, self.nZ)
            resi_psi_sol = self.vmap_linear_solver(
                self.resi[:, 0:1], self.resi[:, 1:2],
                resi_rhs,
                self.bdry,
                resi_psi_bnd
                ).squeeze(2)
            nR0 = jnp.linalg.norm(
                plasma_psi.reshape(-1, self.nR * self.nZ) - resi_psi_sol.reshape(-1, self.nR * self.nZ),
                axis=1
            )
            adjusted_step_size = jnp.repeat(
                nR0[:, None, None, None], self.output_dim, axis=0
                ) * 2.5
            adjusted_step_size *= jnp.repeat(
                (1 + jnp.arange(0.0, self.output_dim)[:, None, None, None]) ** -1, 
                self.batch_size, axis=0
            )
            G = adjusted_step_size * Q

            jtor = self.jtor(
                jnp.repeat(plasma_psi.reshape(-1, 1, self.nR, self.nZ), self.output_dim, axis=0) + \
                G,
                jnp.repeat(tokamak_psi, self.output_dim, axis=0),
                jnp.repeat(constraints, self.output_dim, axis=0)
                ).reshape(-1, self.nR, self.nZ)
            psi_bnd = jnp.tensordot(
                jtor, self.greenfunc, axes=([1, 2], [1, 2])
            )[:, :, None]

            rhs = self.rhs_before_jtor * jtor.reshape(-1, 1, self.nR, self.nZ)

            # (2) Solve Grad-Shafranov equation w/ linear solver
            psi_sol = self.vmap_linear_solver(
                self.resi[:, 0:1], self.resi[:, 1:2],
                rhs,
                self.bdry,
                psi_bnd
                )
            psi_sol = jnp.permute_dims(
                psi_sol.reshape(self.output_dim, -1, self.nR * self.nZ), (1, 2, 0)
                )
            Q = jnp.permute_dims(
                Q.reshape(self.output_dim, -1, self.nR * self.nZ), (1, 2, 0)
            )
            jacvec = psi_sol - resi_psi_sol[:, :, None]
            invar_loss = jacvec - jnp.matmul(jnp.matmul(
                Q,
                jnp.permute_dims(Q, (0, 2, 1))
                ), jacvec)
        # loss = jnp.sum(
        #     jnp.linalg.norm(invar_loss, axis=(1, 2), ord='fro') ** 2  / \
        #     (jnp.linalg.norm(jacvec, axis=(1, 2), ord='fro') ** 2 + 1e-6)
        #     )
        loss = jnp.sum(
            jnp.linalg.norm(invar_loss, axis=(1, 2), ord='fro') ** 2
        )
        # jax.debug.print("nan_psi_sol | {x}", x=jnp.isnan(psi_sol).any())
        # jax.debug.print("loss | {x}", x=loss)
        # break_if_nan(psi_sol)
        # jax.debug.breakpoint()
        return loss
    
    @eqx.filter_jit(donate="all-except-first")
    def loss_data(
        self, model: NKDeepONet,
        plasma_psi: Float64[JaxArray, "batch nx*ny"],
        tokamak_psi: Float64[JaxArray, "batch nx*ny"],
        constraints: Float64[JaxArray, "batch n_params"],
        Q: Float64[JaxArray, "batch nx*ny output_dim"],
        sharding: jax.sharding.NamedSharding,
        sharding_model: jax.sharding.NamedSharding
        ) -> Float64[JaxArray, ""]:
        
        psi = jnp.concatenate([plasma_psi, tokamak_psi], axis=1)

        vmap_model = vmap(model)
        output = vmap_model(psi, constraints)
        output = jnp.permute_dims(output, (0, 2, 3, 1)).reshape(
            -1, self.nR * self.nZ, self.output_dim
            )

        Q_predict, _ = jnp.linalg.qr(output, mode='reduced')

        loss = jnp.linalg.norm(Q - Q_predict, axis=(1, 2), ord='fro').sum()

        return loss

    @eqx.filter_jit(donate="all")
    def make_step(
        self, 
        model: NKDeepONet,
        opt_state: optax.OptState,
        plasma_psi: Float64[JaxArray, "B 1 nx ny"],
        tokamak_psi: Float64[JaxArray, "B 1 nx ny"],
        constraints: Float64[JaxArray, "B n_params"],
        G: Float64[JaxArray, "B nx*ny"],
        grad_transf: optax.GradientTransformation,
        sharding: jax.sharding.NamedSharding,
        sharding_model: jax.sharding.NamedSharding
        ) -> Tuple[NKDeepONet, optax.OptState, float]:

        model, opt_state = eqx.filter_shard((model, opt_state), sharding_model)
        plasma_psi, tokamak_psi, constraints, G = eqx.filter_shard(
            (plasma_psi, tokamak_psi, constraints, G), sharding
        )

        params, static = eqx.partition(model, eqx.is_array)
        def value_fn(param: JaxArray) -> float:
            if self.use_nk_loss:
                return self.loss_nk(
                    param, static, plasma_psi, tokamak_psi, constraints, G
                )
            else:
                return self.loss(
                    param, static, plasma_psi, tokamak_psi, constraints
                    )
            # return self.loss_data(
            #     param, static, plasma_psi, tokamak_psi, constraints, updates_psi
            #     )
        
        value, grads = eqx.filter_value_and_grad(value_fn)(params)
        updates, opt_state = grad_transf.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)

        return eqx.filter_shard((model, opt_state), sharding_model), value
    
    def load(self, model: eqx.Module, load_name: Dict[int, str]) -> None:
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
                f, model
                ), train_dataset, test_dataset
            
            
    def save(self, model: eqx.Module, epoch: int, save_path: str) -> None:
        save_path = os.path.join(save_path, 'model')
        os.makedirs(
            save_path, exist_ok=True
        )
        save_path = os.path.join(
            save_path, f'model_{epoch}.eqx'
        )
        with open(save_path, 'wb') as f:
            eqx.tree_serialise_leaves(f, model)

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

class Trainer_separatrix:
    
    def __init__(
        self, Rmin: float, Rmax: float, Zmin: float, Zmax: float,
        nR: int, nZ: int, batch_size: int, epoch: int, 
        len_data: Optional[int] = None, 
        data_load_path: Optional[str] = None,
        data_save_path: Optional[str] = None,
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

        self.dataset = GSrhsdatasetMASTU_separatrix(
            Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax,
            nR=nR, nZ=nZ, num=len_data, max_iter=150,
            load_path=data_load_path, save_path=data_save_path
        )

        tokamak = build_machine.tokamak(
            active_coils_path=f"freegsnke/machine_configs/MAST-U/MAST-U_like_active_coils.pickle",
            passive_coils_path=f"freegsnke/machine_configs/MAST-U/MAST-U_like_passive_coils.pickle",
            limiter_path=f"freegsnke/machine_configs/MAST-U/MAST-U_like_limiter.pickle",
            wall_path=f"freegsnke/machine_configs/MAST-U/MAST-U_like_wall.pickle",
        )
        eq = equilibrium_update.Equilibrium(
            tokamak=tokamak,      # provide tokamak object
            Rmin=0.1, Rmax=2.0,   # radial range
            Zmin=-2.2, Zmax=2.2,  # vertical range
            nx=65,                # number of grid points in the radial direction (needs to be of the form (2**n + 1) with n being an integer)
            ny=129,               # number of grid points in the vertical direction (needs to be of the form (2**n + 1) with n being an integer)
            # psi=plasma_psi
        )  
        limiter_handler = limiter_func.Limiter_handler(eq, tokamak.limiter)
        limiter_handler.build_mask_inside_limiter()
        self.mask = jnp.asarray(limiter_handler.mask_inside_limiter)

    def train(
        self,
        save_name: str,
        load_name: Optional[Dict[int, str]] = None
        ) -> None:

        model = XPlimnet(
            nx=self.nR, ny=self.nZ, key=jax.random.PRNGKey(0)
            )
        save_path, train_dataloader, test_dataloader, load_epoch = \
            self.preprocess(save_name, load_name)

        print("Length for the dataloader :", len(train_dataloader))

        params = eqx.filter(model, eqx.is_array)
        
        grad_transf = optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_schedule(optax.linear_schedule(
                init_value=1e-3, end_value=1e-5,
                transition_steps=int(self.epoch * 0.8),
                transition_begin=int(self.epoch * 0.15),
                )),
            optax.scale(-1.0)
        )
        # grad_transf = optax.adam(optax.scale_by_schedule(optax.linear_schedule(
        #     init_value=1e-3, end_value=1e-5,
        #     transition_steps=int(self.epoch * 0.8),
        #     transition_begin=int(self.epoch * 0.15),
        #     )))
        opt_state = grad_transf.init(params)

        loss_track = []
        loss_track_test = []
        for epoch in range(load_epoch, self.epoch):
            loss_recon = 0.0
            loss_class = 0.0
            test_loss_recon_axis = 0.0
            test_loss_recon_bdry = 0.0
            num_data = num_data_test = 0
            test_tp = test_tn = test_fp = test_fn = 0
            for _i, batch in enumerate(train_dataloader):
                idx, plasma_psi, tokamak_psi, \
                    psi_axis, psi_bndry, flag_limiter = batch
                plasma_psi = jnp.asarray(plasma_psi).reshape(-1, 1, self.nR, self.nZ)
                tokamak_psi = jnp.asarray(tokamak_psi).reshape(-1, 1, self.nR, self.nZ)
                psi = plasma_psi + tokamak_psi

                psi_axis = jnp.asarray(psi_axis)[:, None]
                psi_bndry = jnp.asarray(psi_bndry)[:, None]
                psi_recon = jnp.concatenate(
                    [psi_axis, psi_bndry], axis=1
                )
                flag_limiter = jnp.asarray(flag_limiter)

                model, opt_state, loss_recon_batch, loss_class_batch = self.make_step(
                    model, opt_state, psi,
                    psi_recon, flag_limiter, grad_transf
                    )
                if (_i + 1) % 200 == 0:
                    print(
                        f'Step {_i+1} | Loss_recon: {loss_recon_batch:.6e} ' \
                        f'| Loss_class: {loss_class_batch:.6e}')
                loss_recon += loss_recon_batch
                loss_class += loss_class_batch
                num_data += len(idx)

            loss_recon /= num_data
            loss_class /= num_data
            print(
                f"Epoch {epoch} train | " \
                f"Loss_recon: {jnp.sqrt(loss_recon):.6e} | " \
                f"Loss_class: {loss_class:.6e}"
                )
            loss_track.append(jnp.sqrt(loss_recon))

            for _i, batch in enumerate(test_dataloader):
                idx, plasma_psi, tokamak_psi, \
                    psi_axis, psi_bndry, flag_limiter = batch
                plasma_psi = jnp.asarray(plasma_psi).reshape(-1, 1, self.nR, self.nZ)
                tokamak_psi = jnp.asarray(tokamak_psi).reshape(-1, 1, self.nR, self.nZ)
                psi = plasma_psi + tokamak_psi

                psi_axis = jnp.asarray(psi_axis)[:, None]
                psi_bndry = jnp.asarray(psi_bndry)[:, None]
                psi_recon = jnp.concatenate(
                    [psi_axis, psi_bndry], axis=1
                )
                flag_limiter = jnp.asarray(flag_limiter)
                params, static = eqx.partition(model, eqx.is_array)
                test_loss_batch_recon_axis, test_loss_batch_recon_bdry, \
                test_class_tp, test_class_tn, \
                    test_class_fp, test_class_fn  = self.loss_test(
                    params, static, psi, psi_recon, flag_limiter
                )
                test_tp += test_class_tp
                test_tn += test_class_tn
                test_fp += test_class_fp
                test_fn += test_class_fn
                test_loss_recon_axis += test_loss_batch_recon_axis
                test_loss_recon_bdry += test_loss_batch_recon_bdry
                num_data_test += len(idx)
            test_loss_recon_axis /= num_data_test
            test_loss_recon_bdry /= num_data_test
            test_loss_class_f1 = 2 * test_tp / (
                2 * test_tp + test_fp + test_fn + 1e-8
                )
            print(
                f"Epoch {epoch} test | " \
                f"Loss reconstruction axis: {jnp.sqrt(test_loss_recon_axis):.6e} | " \
                f"Loss reconstruction boundary: {jnp.sqrt(test_loss_recon_bdry):.6e} | " \
                f"F1 score classification: {test_loss_class_f1:.2e}"
                )
            loss_track_test.append([jnp.sqrt(test_loss_recon_axis), jnp.sqrt(test_loss_recon_bdry)])
            if (epoch + 1) % 50 == 0:
                self.save(model, epoch, save_path, loss_track, loss_track_test)

    def preprocess(
        self, save_name: str, load_name: Optional[Dict[int, str]] = None
        ) -> Tuple[str, DataLoader, DataLoader, int]:
        save_path = os.path.join('logs', save_name)
        os.makedirs(save_path, exist_ok=True)
        data_size = len(self.dataset)
        train_size = int(0.9 * data_size)
        test_size = data_size - train_size

        if load_name:
            load_epoch = load_name['epoch']
            model, train_dataset, test_dataset = self.load(model, load_name)
        else:
            load_epoch = 0
            train_dataset, test_dataset = random_split(
                self.dataset, [train_size, test_size]
                )
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
        return save_path, train_dataloader, test_dataloader, load_epoch
    
    @eqx.filter_jit
    def loss(
        self, params: JaxArray, static,
        psi: Float64[JaxArray, "batch nx*ny"],
        psi_recon: Float64[JaxArray, "batch nx*ny"],
        flag_limiter: Float64[JaxArray, "batch 1"],
        ) -> Float64[JaxArray, "batch 2"]:
        
        model = eqx.combine(params, static)
        vmap_model = vmap(model)
        out = vmap_model(psi)
        reconstruction, classification = out[0], out[1].squeeze(1)
        labels = flag_limiter.astype(jnp.float64)
        log_p = jnp.log(jnp.clip(classification, a_min=1e-6))
        log_not_p = jnp.log(
            jnp.clip(1.0 - classification, a_min=1e-6)
            )
        loss_classification = jnp.sum(-labels * log_p - (1.0 - labels) * log_not_p)
        loss_reconstruction = jnp.sum(
            (reconstruction - psi_recon) ** 2
        )

        return loss_classification, loss_reconstruction
    
    @eqx.filter_jit
    def loss_test(
        self, params: JaxArray, static,
        psi: Float64[JaxArray, "batch nx*ny"],
        psi_recon: Float64[JaxArray, "batch nx*ny"],
        flag_limiter: Float64[JaxArray, "batch 1"],
        ) -> Tuple[Float64[JaxArray, ""], int, int, int, int]:
        
        model = eqx.combine(params, static)
        vmap_model = vmap(model)
        vals = psi * self.mask[None, None, :, :]
        scaling = jnp.max(vals, axis=(2, 3), keepdims=True) - \
            jnp.min(vals, axis=(2, 3), keepdims=True)
        psi_scaling = (vals - jnp.min(vals, axis=(2, 3), keepdims=True)) / scaling 
        psi_scaling *= self.mask[None, None, :, :]
        psi_recon_scaling = (psi_recon - jnp.min(vals, axis=(2, 3))) / scaling.squeeze(3).squeeze(2)
        out = vmap_model(psi_scaling)
        reconstruction = out[0]
        classification = out[1].squeeze(1)
        predict_labels = classification >= 0.5
        labels = flag_limiter
        tp = jnp.sum((labels == 1) & (predict_labels == 1))
        tn = jnp.sum((labels == 1) & (predict_labels == 0))
        fp = jnp.sum((labels == 0) & (predict_labels == 0))
        fn = jnp.sum((labels == 0) & (predict_labels == 1))

        loss_recon = jnp.sum(((psi_recon - reconstruction)) ** 2, axis=0)
        loss_recon_axis, loss_recon_bdry = loss_recon[0], loss_recon[1]
        return loss_recon_axis, loss_recon_bdry, tp, tn, fp, fn

    @eqx.filter_jit
    def make_step(
        self, 
        model: NKDeepONet,
        opt_state: optax.OptState,
        psi: Float64[JaxArray, "B 1 nx ny"],
        psi_recon: Float64[JaxArray, "B 1"],
        flag_limiter: Float64[JaxArray, "B 1"],
        grad_transf: optax.GradientTransformation,
        ) -> Tuple[NKDeepONet, optax.OptState, float]:

        params, static = eqx.partition(model, eqx.is_array)
        vals = psi * self.mask[None, None, :, :]
        scaling = jnp.max(vals, axis=(2, 3), keepdims=True) - \
            jnp.min(vals, axis=(2, 3), keepdims=True)
        psi_scaling = (vals - jnp.min(vals, axis=(2, 3), keepdims=True)) / scaling 
        psi_scaling *= self.mask[None, None, :, :]
        psi_recon_scaling = (psi_recon - jnp.min(vals, axis=(2, 3))) / scaling.squeeze(3).squeeze(2)
        def value_fn(param: JaxArray) -> float:
            loss_recon, loss_class = self.loss(
                param, static, psi_scaling, psi_recon_scaling, flag_limiter
                )
            return loss_recon + loss_class, (loss_recon, loss_class)
        (_, aux), grads = eqx.filter_value_and_grad(
            value_fn, has_aux=True
            )(params)
        updates, opt_state = grad_transf.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, aux[0], aux[1]
    
    def load(self, model: eqx.Module, load_name: Dict[int, str]) -> None:
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
                f, model
                ), train_dataset, test_dataset
            
            
    def save(
        self, model: eqx.Module, epoch: int, save_path: str,
        loss_track: list, loss_track_test: list
        ) -> None:
        save_path_model = os.path.join(save_path, 'model')
        os.makedirs(
            save_path_model, exist_ok=True
        )
        save_path_model = os.path.join(
            save_path_model, f'model_{epoch}.eqx'
        )
        with open(save_path_model, 'wb') as f:
            eqx.tree_serialise_leaves(f, model)
        
        save_path_loss = os.path.join(save_path, 'loss')
        os.makedirs(
            save_path_loss, exist_ok=True
        )
        loss_track = jnp.asarray(loss_track)
        loss_track_test = jnp.asarray(loss_track_test)
        np.savez(
            os.path.join(save_path_loss, "loss.npz"),
            train=jax.device_get(loss_track), test=jax.device_get(loss_track_test)
            )


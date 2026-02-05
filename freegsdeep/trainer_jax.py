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
        data_load_path: Optional[str], data_save_path: Optional[str]
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

        self.dataset = GSrhsdatasetMASTU_g(
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

        model = NKDeepONet(
            nx=self.nR, ny=self.nZ, hidden_dim=15, 
            tokamak_input_len=3, key=jax.random.PRNGKey(0)
            )
        save_path, train_dataloader, test_dataloader, load_epoch = \
            self.preprocess(save_name, load_name)

        print("Length for the dataloader :", len(train_dataloader))

        optim = optax.adam(learning_rate=1e-3)
        params = eqx.filter(model, eqx.is_array)
        opt_state = optim.init(params)
        for epoch in range(load_epoch, self.epoch):
            loss = test_loss = 0.0
            num_data = num_data_test = 0
            for _i, batch in enumerate(train_dataloader):
                idx, plasma_psi, tokamak_psi, constraint, update = batch
                plasma_psi = jnp.asarray(
                    plasma_psi.reshape(-1, 1, self.nR, self.nZ)
                )
                tokamak_psi = jnp.asarray(
                    tokamak_psi.reshape(-1, 1, self.nR, self.nZ)
                )
                constraint = jnp.asarray(constraint)
                update = jnp.asarray(update)

                model, opt_state, loss_batch = self.make_step(
                    model, opt_state, plasma_psi,
                    tokamak_psi, constraint, update, optim
                    )
                if (_i + 1) % 200 == 0:
                    print(f'Step {_i+1} | Loss: {loss_batch:.6e}')
                loss += loss_batch
                num_data += len(idx)

            loss /= num_data
            print(
                f"Epoch {epoch} train | " \
                f"Loss: {jnp.sqrt(loss):.6e}"
                )
            self.save(model, epoch, save_path)

            for _i, batch in enumerate(test_dataloader):
                idx, plasma_psi, tokamak_psi, constraint, update = batch
                plasma_psi = jnp.asarray(
                    plasma_psi.reshape(-1, 1, self.nR, self.nZ)
                )
                tokamak_psi = jnp.asarray(
                    tokamak_psi.reshape(-1, 1, self.nR, self.nZ)
                )
                constraint = jnp.asarray(constraint)
                update = jnp.asarray(update)
                params, static = eqx.partition(model, eqx.is_array)
                # test_loss_batch = self.loss(
                #     params, static, plasma_psi, tokamak_psi, constraint
                #     )
                test_loss_batch = self.loss_data(
                    params, static, plasma_psi, tokamak_psi, constraint, update
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
            train_dataset, batch_size=self.batch_size, shuffle=True, 
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size * 5, shuffle=False,
        )
        return save_path, train_dataloader, test_dataloader, load_epoch
    
    def linear_solver(
        self, plasma_psi: Float64[JaxArray, "batch nx*ny"],
        tokamak_psi: Float64[JaxArray, "batch nx*ny"],
        constraints: Float64[JaxArray, "batch n_params"]
        ) -> Float64[JaxArray, "batch nx*ny"]:
        # (1) Boundary condition update after update toroidal plasma current 

        # (2) Solve Grad-Shafranov equation w/ linear solver
        pass
    

    def loss(
        self, params: JaxArray, static,
        plasma_psi: Float64[JaxArray, "batch nx*ny"],
        tokamak_psi: Float64[JaxArray, "batch nx*ny"],
        constraints: Float64[JaxArray, "batch n_params"]
        ) -> Float64[JaxArray, ""]:
        
        model = eqx.combine(params, static)
        psi = jnp.concatenate([plasma_psi, tokamak_psi], axis=1)
        
        vmap_model = vmap(
            vmap(model, in_axes=(0, 0, None, None)),
            in_axes=(None, None, 0, 0)
            )

        R = self.resi[:, 0:1] # (nx*ny, )
        Z = self.resi[:, 1:2] # (nx*ny, )

        update = vmap_model(R, Z, psi, constraints)
    
        loss = jnp.linalg.norm(plasma_psi + update - self.linear_solver(
            plasma_psi + update, tokamak_psi, constraints
        ), axis=1).sum()

        return loss
    
    @eqx.filter_jit
    def loss_data(
        self, params: JaxArray, static,
        plasma_psi: Float64[JaxArray, "batch nx*ny"],
        tokamak_psi: Float64[JaxArray, "batch nx*ny"],
        constraints: Float64[JaxArray, "batch n_params"],
        update: Float64[JaxArray, "batch nx*ny"],
        ) -> Float64[JaxArray, ""]:
        
        model = eqx.combine(params, static)
        psi = jnp.concatenate([plasma_psi, tokamak_psi], axis=1)
        
        vmap_model = vmap(
            vmap(model, in_axes=(0, 0, None, None)),
            in_axes=(None, None, 0, 0)
            )

        R = self.resi[:, 0:1] # (nx*ny, )
        Z = self.resi[:, 1:2] # (nx*ny, )

        predict = vmap_model(R, Z, psi, constraints).squeeze(2)
        loss = jnp.linalg.norm(update - predict, axis=1).sum()

        return loss

    @eqx.filter_jit
    def make_step(
        self, 
        model: NKDeepONet,
        opt_state: optax.OptState,
        plasma_psi: Float64[JaxArray, "B 1 nx ny"],
        tokamak_psi: Float64[JaxArray, "B 1 nx ny"],
        constraints: Float64[JaxArray, "B n_params"],
        updates_psi: Float64[JaxArray, "B 1 nx ny"],
        optim: optax.GradientTransformation,
        ) -> Tuple[NKDeepONet, optax.OptState, float]:

        params, static = eqx.partition(model, eqx.is_array)
        def value_fn(param: JaxArray) -> float:
            # return self.loss(
            #     param, static, plasma_psi, tokamak_psi, constraints
            #     ) 
            return self.loss_data(
                param, static, plasma_psi, tokamak_psi, constraints, updates_psi
                )
        value, grads = eqx.filter_value_and_grad(value_fn)(params)
        updates, opt_state = optim.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, value
    
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
        nR: int, nZ: int, batch_size: int, epoch: int, len_data: Optional[int],
        data_load_path: Optional[str], data_save_path: Optional[str]
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
        
        # grad_transf = optax.chain(
        #     optax.scale_by_adam(),
        #     optax.scale_by_schedule(optax.linear_schedule(
        #         init_value=1e-3, end_value=1e-5,
        #         transition_steps=int(self.epoch * 0.8),
        #         transition_begin=int(self.epoch * 0.15),
        #         )),
        #     optax.scale(-1.0)
        # )
        grad_transf = optax.adam(learning_rate=1e-3)
        opt_state = grad_transf.init(params)

        for epoch in range(load_epoch, self.epoch):
            loss = test_loss_recon = 0.0
            num_data = num_data_test = 0
            test_tp = test_tn = test_fp = test_fn = 0
            for _i, batch in enumerate(train_dataloader):
                idx, psi, psi_bndry, flag_limiter = batch
                psi = jnp.asarray(psi).reshape(-1, 1, self.nR, self.nZ)
                psi_bndry = jnp.asarray(psi_bndry)
                flag_limiter = jnp.asarray(flag_limiter)

                model, opt_state, loss_batch = self.make_step(
                    model, opt_state, psi,
                    psi_bndry, flag_limiter, grad_transf
                    )
                if (_i + 1) % 200 == 0:
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
                idx, psi, psi_bndry, flag_limiter = batch
                psi = jnp.asarray(
                    psi.reshape(-1, 1, self.nR, self.nZ)
                )
                psi_bndry = jnp.asarray(psi_bndry)
                flag_limiter = jnp.asarray(flag_limiter)
                params, static = eqx.partition(model, eqx.is_array)
                test_loss_batch_recon, test_class_tp, test_class_tn, \
                    test_class_fp, test_class_fn  = self.loss_test(
                    params, static, psi, psi_bndry, flag_limiter
                )
                test_tp += test_class_tp
                test_tn += test_class_tn
                test_fp += test_class_fp
                test_fn += test_class_fn
                test_loss_recon += test_loss_batch_recon
                num_data_test += len(idx)
            test_loss_recon /= num_data_test
            test_loss_class_f1 = 2 * test_tp / (
                2 * test_tp + test_fp + test_fn + 1e-8
                )
            print(
                f"Epoch {epoch} test | " \
                f"Loss reconstruction: {jnp.sqrt(test_loss_recon):.6e} | " \
                f"F1 score classification: {test_loss_class_f1:.2e}"
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
        psi_bndry: Float64[JaxArray, "batch nx*ny"],
        flag_limiter: Float64[JaxArray, "batch 1"],
        ) -> Float64[JaxArray, "batch 2"]:
        
        model = eqx.combine(params, static)
        vmap_model = vmap(model)
        out = vmap_model(psi)
        reconstruction, classification = out[0].squeeze(1), out[1].squeeze(1)
        labels = flag_limiter.astype(jnp.float64)
        log_p = jnp.log(jnp.clip(classification, a_min=1e-6))
        log_not_p = jnp.log(
            jnp.clip(1.0 - classification, a_min=1e-6)
            )
        loss_classification = jnp.sum(-labels * log_p - (1.0 - labels) * log_not_p)
        loss_reconstruction = jnp.sum(
            (reconstruction - psi_bndry) ** 2
        )

        return loss_classification, loss_reconstruction
    
    @eqx.filter_jit
    def loss_test(
        self, params: JaxArray, static,
        psi: Float64[JaxArray, "batch nx*ny"],
        psi_bndry: Float64[JaxArray, "batch nx*ny"],
        flag_limiter: Float64[JaxArray, "batch 1"],
        ) -> Tuple[Float64[JaxArray, ""], int, int, int, int]:
        
        model = eqx.combine(params, static)
        vmap_model = vmap(model)
        out = vmap_model(psi)
        reconstruction, classification = out[0].squeeze(1), out[1].squeeze(1)
        predict_labels = classification >= 0.5
        labels = flag_limiter
        tp = jnp.sum((labels == 1) & (predict_labels == 1))
        tn = jnp.sum((labels == 1) & (predict_labels == 0))
        fp = jnp.sum((labels == 0) & (predict_labels == 0))
        fn = jnp.sum((labels == 0) & (predict_labels == 1))

        loss_recon = jnp.sum((psi_bndry - reconstruction) ** 2)

        return loss_recon, tp, tn, fp, fn

    @eqx.filter_jit
    def make_step(
        self, 
        model: NKDeepONet,
        opt_state: optax.OptState,
        psi: Float64[JaxArray, "B 1 nx ny"],
        psi_bndry: Float64[JaxArray, "B 1"],
        flag_limiter: Float64[JaxArray, "B 1"],
        grad_transf: optax.GradientTransformation,
        ) -> Tuple[NKDeepONet, optax.OptState, float]:

        params, static = eqx.partition(model, eqx.is_array)
        def value_fn(param: JaxArray) -> float:
            loss_recon, loss_class = self.loss(
                param, static, psi, psi_bndry, flag_limiter
                )
            return loss_recon + loss_class
        value, grads = eqx.filter_value_and_grad(value_fn)(params)
        updates, opt_state = grad_transf.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, value
    
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

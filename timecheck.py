import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
tex_bin = "/data1/home/ahn40200393/texlive/2025/bin/x86_64-linux"
os.environ["PATH"] = tex_bin + ":" + os.environ.get("PATH", "")
import torch
import numpy as np
import jax
import jax.numpy as jnp
from jax import config
from jax.numpy import ndarray as JaxArray
import equinox as eqx
config.update("jax_enable_x64", True)
from freegsdeep.model import DeepONet_resi, PINTO, DeepONet_bdry
from freegs.freegs.multigrid import createVcycle
from freegs.freegs.gradshafranov import GSsparse4thOrder
from datetime import datetime
from freegsnke import GSstaticsolver, nk_solver_H
import cProfile
import pstats
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

class Waveact_jax(eqx.Module):
    w1: JaxArray
    w2: JaxArray
    def __init__(self) -> None:
        self.w1 = jnp.ones((1, ), dtype=jnp.float64)
        self.w2 = jnp.ones((1, ), dtype=jnp.float64)
    def __call__(self, x: JaxArray) -> JaxArray:
        return self.w1 * jnp.sin(x) + self.w2 * jnp.cos(x)

class Trunk(eqx.Module):
    layer1: eqx.nn.Linear
    act1: callable
    layer2: eqx.nn.Linear
    def __init__(self, key) -> None:
        key1, key2 = jax.random.split(key, 2)
        self.layer1 = eqx.nn.Linear(2, 32, key=key1, dtype=jnp.float64)
        self.act1 = Waveact_jax()
        self.layer2 = eqx.nn.Linear(32, 10, key=key2, dtype=jnp.float64)
    def __call__(self, x: JaxArray) -> JaxArray:
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        return x.T

class Branch(eqx.Module):
    layer1: eqx.nn.Linear
    act1: callable
    layer2: eqx.nn.Linear
    act2: callable
    layer3: eqx.nn.Linear
    def __init__(self, nx: int, ny: int, key) -> None:
        key1, key2, key3 = jax.random.split(key, 3)
        self.layer1 = eqx.nn.Linear(nx * ny, (nx // 4) * (ny // 4), key=key1, dtype=jnp.float64)
        self.act1 = jax.nn.silu
        self.layer2 = eqx.nn.Linear((nx // 4) * (ny // 4), 32, key=key2, dtype=jnp.float64)
        self.act2 = jax.nn.silu
        self.layer3 = eqx.nn.Linear(32, 10, key=key3, dtype=jnp.float64)
    def __call__(self, x: JaxArray) -> JaxArray:
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        return x

class MLP_output(eqx.Module):
    layer1: eqx.nn.Linear
    act1: callable
    layer2: eqx.nn.Linear
    def __init__(self, key) -> None:
        key1, key2 = jax.random.split(key, 2)
        self.layer1 = eqx.nn.Linear(20, 32, key=key1, dtype=jnp.float64)
        self.act1 = Waveact_jax()
        self.layer2 = eqx.nn.Linear(32, 1, key=key2, dtype=jnp.float64)
    def __call__(self, x: JaxArray) -> JaxArray:
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        return x
    
class DeepONet_resi_jax(eqx.Module):
    nx: int = eqx.field(static=True)
    ny: int = eqx.field(static=True)
    trunk: Trunk
    branch_source: Branch
    output_mlp: MLP_output
    def __init__(self, nx: int, ny: int, key) -> None:
        key1, key2, key3 = jax.random.split(key, 3)
        self.nx, self.ny = nx, ny
        self.trunk = Trunk(key1)
        self.branch_source = Branch(nx, ny, key2)
        self.output_mlp = MLP_output(key3)
    
    def __call__(
        self, R: JaxArray, Z: JaxArray, rhs: JaxArray
        ) -> JaxArray:
        input_trunk = jnp.concatenate((R, Z), axis=0)
        output1 = self.trunk(input_trunk)
        output2 = self.branch_source(rhs)
        source_output = self.output_mlp(jnp.concatenate(
            [output1, output2], axis=0
        ))
        return source_output

class FiLM(eqx.Module):
    alpha: eqx.nn.Linear
    beta: eqx.nn.Linear
    def __init__(self, in_features: int, out_features: int, key) -> None:
        key1, key2 = jax.random.split(key, 2)
        self.alpha = eqx.nn.Linear(in_features, out_features, key=key1)
        self.beta = eqx.nn.Linear(in_features, out_features, key=key2)
    
    def __call__(self, x: JaxArray, cond: JaxArray) -> JaxArray:
        alpha = self.alpha(cond)
        beta = self.beta(cond)
        return alpha * x + beta
    
class DeepONet_jax(eqx.Module):
    nx: int = eqx.field(static=True)
    ny: int = eqx.field(static=True)
    linear1: eqx.nn.Linear
    act1: Waveact_jax
    film1: FiLM
    linear2: eqx.nn.Linear
    act2: Waveact_jax
    film2: FiLM
    linear3: eqx.nn.Linear
    film3: FiLM
    
    def __init__(self, nx: int, ny: int, key) -> None:
        self.nx, self.ny = nx, ny
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)
        self.linear1 = eqx.nn.Linear(2, 16, key=key1)
        self.act1 = Waveact_jax()
        self.film1 = FiLM(2 * nx + 2 * ny, 16, key=key2)
        self.linear2 = eqx.nn.Linear(16, 16, key=key3)
        self.act2 = Waveact_jax()
        self.film2 = FiLM(2 * nx + 2 * ny, 16, key=key4)
        self.linear3 = eqx.nn.Linear(16, 1, key=key5)
        self.film3 = FiLM(2 * nx + 2 * ny, 1, key=key6)
        
    def __call__(
        self, R: JaxArray, Z: JaxArray, bdry: JaxArray
        ) -> JaxArray:
        input_trunk = jnp.concatenate((R, Z))
        output = self.linear1(input_trunk)
        output = self.film1(output, bdry)
        output = self.act1(output)
        # output = self.linear2(output)
        # output = self.film2(output, bdry)
        # output = self.act2(output)
        output = self.linear3(output)
        output = self.film3(output, bdry)
        return output


class PINTO_jax(eqx.Module):
    pos_encoder: eqx.nn.Sequential
    key_encoder: eqx.nn.Sequential
    value_encoder: eqx.nn.Sequential
    MHA1: eqx.nn.MultiheadAttention
    mlp2: eqx.nn.Sequential
    MHA3: eqx.nn.MultiheadAttention
    mlp4: eqx.nn.Sequential
    decoder: eqx.nn.Sequential
    
    def __init__(self, key) -> None:
        key1, key2, key3, key4, key5, key6, key7, key8 = jax.random.split(key, 8)
        key11, key12 = jax.random.split(key1, 2)
        self.pos_encoder = eqx.nn.Sequential((
            eqx.nn.Linear(2, 32, key=key11),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(32, 10, key=key12)
        ))
        self.pos_encoder = eqx.filter_vmap(self.pos_encoder)
        key21, key22 = jax.random.split(key2, 2)
        self.key_encoder = eqx.nn.Sequential((
            eqx.nn.Linear(2, 32, key=key21),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(32, 10, key=key22)
        ))
        self.key_encoder = eqx.filter_vmap(self.key_encoder)
        key31, key32 = jax.random.split(key3, 2)
        self.value_encoder = eqx.nn.Sequential((
            eqx.nn.Linear(1, 32, key=key31),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(32, 10, key=key32)
        ))
        self.value_encoder = eqx.filter_vmap(self.value_encoder)
        self.MHA1 = eqx.nn.MultiheadAttention(num_heads=1, query_size=10, key=key4)
        key51, key52 = jax.random.split(key5, 2)
        self.mlp2 = eqx.nn.Sequential((
            eqx.nn.Linear(10, 32, key=key51),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(32, 10, key=key52)
        ))
        self.mlp2 = eqx.filter_vmap(self.mlp2)
        self.MHA3 = eqx.nn.MultiheadAttention(num_heads=1, query_size=10, key=key6)
        key71, key72 = jax.random.split(key7, 2)
        self.mlp4 = eqx.nn.Sequential((
            eqx.nn.Linear(10, 50, key=key71),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(50, 10, key=key72)
        ))
        self.mlp4 = eqx.filter_vmap(self.mlp4)
        key81, key82 = jax.random.split(key8, 2)
        self.decoder = eqx.nn.Sequential((
            eqx.nn.Linear(10, 32, key=key81),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(32, 1, key=key82)
        ))
        self.decoder = eqx.filter_vmap(self.decoder)
    
    def __call__(
        self, R: JaxArray, Z: JaxArray, bdry_point: JaxArray, bdry_value: JaxArray
        ) -> JaxArray:
        X = jnp.concatenate((R, Z), axis=1)
        Q = self.pos_encoder(X)
        K = self.key_encoder(bdry_point)
        V = self.value_encoder(bdry_value)
        Q1 = self.MHA1(Q, K, V)
        Q1 = Q1 + Q
        # Q2 = self.mlp2(Q1)
        # Q2 = Q2 + Q1
        Q2 = Q1
        # Q3 = self.MHA3(Q2, K, V)
        # Q3 = Q3 + Q2
        Q3 = Q2
        # Q4 = self.mlp4(Q3)
        # Q4 = Q4 + Q3
        Q4 = Q3
        output = self.decoder(Q4)
        return output
        

def main(iter, nx, ny, device='cuda:0', torch_jax: str='torch'):
    Rmin, Rmax = 0.1, 2.0
    Zmin, Zmax = -1.0, 1.0

    psi_rand = np.random.randn(iter, nx, ny).astype(np.float64)
    dx_rand = np.random.randn(iter, nx, ny).astype(np.float64)
    rhs = np.random.randn(iter, nx, ny).astype(np.float64)
    time_list = []

    start_numerical = datetime.now()
    generator = GSsparse4thOrder(Rmin, Rmax, Zmin, Zmax)
    solver_numerical = createVcycle(nx, ny, generator, nlevels=1, ncycle=1, niter=2, direct=False) 
    for i in range(iter):
        solver_numerical(psi_rand[i, :, :], rhs[i, :, :])
    time_list.append(datetime.now() - start_numerical)
    # solver_numerical_nk = nk_solver_H.nksolver(nx * ny)
    # start_numerical_nk = datetime.now()
    # for i in range(iter):
    #     solver_numerical_nk.Arnoldi_iteration(
    #         x0=psi_rand[i, :, :].reshape(-1),
    #         dx=dx_rand[i, :, :].reshape(-1),
    #         R0=np.ones(nx * ny),
    #         F_function=lambda x, _: x,
    #         args=(1, ),
    #         step_size=1.0,
    #         scaling_with_n=1.0,
    #         target_relative_unexplained_residual=1e-6,
    #         max_n_directions=5,
    #         clip=1.0,
    #         # clip_quantiles=clip_quantiles,
    #     )
    # time_list.append(datetime.now() - start_numerical_nk)

    if torch_jax == 'torch':
        R_torch = torch.linspace(Rmin, Rmax, nx, dtype=torch.float64).to(device)
        Z_torch = torch.linspace(Zmin, Zmax, ny, dtype=torch.float64).to(device)
        R_torch, Z_torch = torch.meshgrid(R_torch, Z_torch, indexing='ij')
        bdry_point_R = torch.concat([
            R_torch[0, :].reshape(-1), R_torch[:, 0].reshape(-1), R_torch[-1, :].reshape(-1), R_torch[:, -1].reshape(-1)
        ]).reshape(-1, 1) 
        bdry_point_Z = torch.concat([
            Z_torch[0, :].reshape(-1), Z_torch[:, 0].reshape(-1), Z_torch[-1, :].reshape(-1), Z_torch[:, -1].reshape(-1)
        ]).reshape(-1, 1)
        R_torch = R_torch.reshape(1, -1, 1)
        Z_torch = Z_torch.reshape(1, -1, 1)
        rhs_torch = torch.from_numpy(rhs).to(device)
        solver_torch = DeepONet_resi(Rmin, Rmax, Zmin, Zmax, nx, ny).to(device).to(torch.float64)
        with torch.no_grad():
            solver_torch_jit = torch.jit.trace(solver_torch, (R_torch, Z_torch, rhs_torch[0:1, :, :].reshape(1, -1)))
            start_torch = datetime.now()    
            for i in range(iter):
                solver_torch_jit(R_torch, Z_torch, rhs_torch[i:i+1, :, :].reshape(1, -1))
        time_list.append(datetime.now() - start_torch)

        solver_pinto = PINTO(Rmin, Rmax, Zmin, Zmax, nx, ny).to(device).to(torch.float64)
        bdry_point = torch.concat([bdry_point_R, bdry_point_Z], dim=1)[None, :, :]
        bdry_value = torch.randn(iter, bdry_point.shape[1], 1).to(torch.float64).to(device)
        with torch.no_grad():
            solver_pinto_jit = torch.jit.trace(solver_pinto, (R_torch, Z_torch, bdry_point, bdry_value[0:1, :, :]))
            start_pinto = datetime.now()    
            for i in range(iter):
                solver_pinto_jit(R_torch, Z_torch, bdry_point, bdry_value[i:i+1, :, :])
        time_list.append(datetime.now() - start_pinto)

        R_torch = R_torch.reshape(-1, 1)
        Z_torch = Z_torch.reshape(-1, 1)
        solver_deeponet_bdry = torch.func.vmap(torch.func.vmap(
            DeepONet_bdry(Rmin, Rmax, Zmin, Zmax, nx, ny).to(device).to(torch.float64).eval(),
            in_dims=(0, 0, None)
            ), in_dims=(None, None, 0))
        with torch.inference_mode():
            solver_deeponet_bdry_jit = torch.compile(solver_deeponet_bdry)
            solver_deeponet_bdry_jit(R_torch, Z_torch, bdry_value[0:1, :, 0])
            start_deeponet_bdry = datetime.now()    
            for i in range(iter):
                solver_deeponet_bdry_jit(
                    R_torch, Z_torch, bdry_value[i:i+1, :, 0]
                    )
        time_list.append(datetime.now() - start_deeponet_bdry)

        time_list = [t.total_seconds() for t in time_list]
        print(time_list)
        return time_list
    else:
        R_jax = jnp.linspace(Rmin, Rmax, nx, dtype=jnp.float64)
        Z_jax = jnp.linspace(Zmin, Zmax, ny, dtype=jnp.float64)
        R_jax, Z_jax = jnp.meshgrid(R_jax, Z_jax, indexing='ij')
        bdry_point_R = jnp.concatenate([
            R_jax[0, :].reshape(-1), R_jax[:, 0].reshape(-1), R_jax[-1, :].reshape(-1), R_jax[:, -1].reshape(-1)
        ]).reshape(-1, 1)
        bdry_point_Z = jnp.concatenate([
            Z_jax[0, :].reshape(-1), Z_jax[:, 0].reshape(-1), Z_jax[-1, :].reshape(-1), Z_jax[:, -1].reshape(-1)
        ]).reshape(-1, 1)
        bdry_point = jnp.concatenate([bdry_point_R, bdry_point_Z], axis=1)
        bdry_value = jax.random.normal(
            jax.random.PRNGKey(1), (iter, bdry_point.shape[0], 1), dtype=jnp.float64
            )
        R_jax = R_jax.reshape(-1, 1)
        Z_jax = Z_jax.reshape(-1, 1)
        rhs_jax = jnp.array(rhs)

        solver_jax = DeepONet_resi_jax(nx, ny, jax.random.PRNGKey(0))
        solver_jax_jit = jax.jit(jax.vmap(solver_jax, in_axes=(0, 0, None)))
        solver_jax_jit(R_jax, Z_jax, rhs_jax[0, :, :].reshape(-1))  # warm up
        start_jax = datetime.now()
        for i in range(iter):
            solver_jax_jit(R_jax, Z_jax, jnp.array(rhs[i, :, :].reshape(-1)))
        time_list.append(datetime.now() - start_jax)

        solver_pinto_jax = PINTO_jax(key=jax.random.PRNGKey(0))
        solver_pinto_jax_jit = jax.jit(jax.vmap(solver_pinto_jax, in_axes=(None, None, None, 0)))
        solver_pinto_jax_jit(R_jax, Z_jax, bdry_point, bdry_value[0:1, :, :])  # warm up
        start_pinto_jax = datetime.now()
        for i in range(iter):
            solver_pinto_jax_jit(
                R_jax, Z_jax, bdry_point, jnp.array(bdry_value[i:i+1, :, :])
                )
        time_list.append(datetime.now() - start_pinto_jax)

        solver_deeponet_bdry_jax = DeepONet_jax(nx, ny, key=jax.random.PRNGKey(42))
        R_jax = R_jax.reshape(nx, ny)
        Z_jax = Z_jax.reshape(nx, ny)
        bdry_point_R = jnp.concatenate([
            R_jax[0, :].reshape(-1), R_jax[:, 0].reshape(-1), R_jax[-1, :].reshape(-1), R_jax[:, -1].reshape(-1)
        ]).reshape(-1, 1)
        bdry_point_Z = jnp.concatenate([
            Z_jax[0, :].reshape(-1), Z_jax[:, 0].reshape(-1), Z_jax[-1, :].reshape(-1), Z_jax[:, -1].reshape(-1)
        ]).reshape(-1, 1)
        bdry_point = jnp.concatenate([bdry_point_R, bdry_point_Z], axis=1)
        bdry_value = jax.random.normal(
            jax.random.PRNGKey(1), (iter, bdry_point.shape[0], 1), dtype=jnp.float64
            )
        R_jax = R_jax.reshape(-1, 1)
        Z_jax = Z_jax.reshape(-1, 1)
        solver_deeponet_bdry_jax_jit = jax.jit(jax.vmap(jax.vmap(
            solver_deeponet_bdry_jax, in_axes=(0, 0, None)
            ), in_axes=(None, None, 0)))
        solver_deeponet_bdry_jax_jit(R_jax, Z_jax, bdry_value[0:1, :, :])  # warm up
        start_deeponet_bdry_jax = datetime.now()
        for i in range(iter):
            solver_deeponet_bdry_jax_jit(
                R_jax, Z_jax, bdry_value[i:i+1, :, :]
                )
        time_list.append(datetime.now() - start_deeponet_bdry_jax)
        time_list = [t.total_seconds() for t in time_list]
        
        print(time_list)
        return time_list

if __name__ == "__main__":
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    _time_list = [] 
    nx, ny = 33, 33
    num_iter = 1000

    mode = 'jax (multigrid numerical)'
    for i in range(6):
        if mode[0] == 't':
            output_main = main(iter=num_iter, nx=nx, ny=ny, device='cuda:0', torch_jax='torch')
        else:
            output_main = main(iter=num_iter, nx=nx, ny=ny, device='cuda:0', torch_jax='jax')
        if i >= 1:
            _time_list.append(output_main)
    time_array = np.array(_time_list)
    min = np.min(time_array, axis=0)
    avg = np.mean(time_array, axis=0)
    max = np.max(time_array, axis=0)
    x = np.arange(4)
    ax[0].bar(x, avg, width=0.6, alpha=0.3)
    ax[0].errorbar(x, avg, yerr=[avg - min, max - avg], fmt='none', capsize=6, linewidth=1.5)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([
        'Numerical', f'DeepONet ({mode})', f'PINTO ({mode})', f'DeepONet-boundary ({mode})'
        ])
    ax[0].set_ylabel("Time (seconds)")
    x = np.arange(3)
    avg_lower = []
    avg_lower.append(2 * avg[0])
    avg_lower.append(avg[1] + avg[2])
    avg_lower.append(avg[1] + avg[3])
    min_lower = []
    min_lower.append(2 * min[0])
    min_lower.append(min[1] + min[2])
    min_lower.append(min[1] + min[3])
    max_lower = []
    max_lower.append(2 * max[0])
    max_lower.append(max[1] + max[2])
    max_lower.append(max[1] + max[3])
    
    avg_lower = np.array(avg_lower)
    min_lower = np.array(min_lower)
    max_lower = np.array(max_lower)
    ax[1].bar(x, avg_lower, width=0.6, alpha=0.3)
    ax[1].errorbar(x, avg_lower, yerr=[avg_lower - min_lower, max_lower - avg_lower], fmt='none', capsize=6, linewidth=1.5)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([
        f'Numerical ({2*num_iter} iterations)',
        f'DeepONet + PINTO ({mode})',
        f'DeepONet + DeepONet-boundary ({mode})'
        ])
    ax[1].set_ylabel("Time (seconds)")
    fig.suptitle(
        rf"Time Comparison for Numerical \& Neural ({mode}) solvers ({num_iter} iterations, {nx}$\times${ny} grid)"
        )
    fig.savefig(f"profile/image/debug_time_comparison_{mode}_{num_iter}_{nx}x{ny}.png")

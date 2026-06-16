import torch
import torch.nn as nn
from freegsdeep.typing import *
from typing import Tuple

class Waveact(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

class Integratednet(nn.Module):
    def __init__(self, nx: int, ny: int) -> None:
        super().__init__()
        self.resi_net = DeepONet_resi(nx, ny)
        self.bdry_net = PINTO(nx, ny)
    
    def forward(
        self, R: Tensor, Z: Tensor, rhs: Tensor, bdry_point: Tensor, bdry_value: Tensor
        ) -> Tensor:
        resi_output = self.resi_net(R, Z, rhs)
        bdry_output = self.bdry_net(R, Z, bdry_point, bdry_value)
        return resi_output, bdry_output
        
class DeepONet_resi(nn.Module):
    
    def __init__(
        self, nx: int, ny: int,
        ) -> None:
        super().__init__()
        self.nx, self.ny = nx, ny
        self.trunk = nn.Sequential(
            nn.Linear(2, 64),
            Waveact(),
            nn.Linear(64, 64),
            Waveact(),
            nn.Linear(64, 10)
        )
        self.branch_source = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(64, 10, kernel_size=3, padding=1),
        )
        # self.branch_source = nn.Sequential(
        #     nn.Linear(nx * ny, (nx // 4) * (ny // 4)),
        #     nn.SiLU(),
        #     nn.Linear((nx // 4) * (ny // 4), 32),
        #     nn.SiLU(),
        #     nn.Linear(32, 10)
        #     )
        # self.branch_mlp = nn.Sequential(
        #     nn.Linear((nx // 4) * (ny // 4), 32),
        #     nn.SiLU(),
        #     nn.Linear(32, 10)
        # )
        self.output_mlp = nn.Sequential(
            nn.Linear(20, 100),
            Waveact(),
            nn.Linear(100, 100),
            Waveact(),
            nn.Linear(100, 1)
        )
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(
        self, R: Tensor, Z: Tensor, rhs: Tensor
        ) -> Tensor:
        input_trunk = torch.concat((R, Z), dim=2)
        output1 = self.trunk(input_trunk)
        # output2 = self.branch_source(rhs) 
        output2 = self.branch_source(rhs)
        # output2 = output2.reshape(output2.shape[0], -1)
        # output2 = self.branch_mlp(output2)
        output1 = output1.expand(output2.shape[0], -1, -1)
        output2 = output2[:, None, :].expand(-1, output1.shape[1], -1)
        source_output = self.output_mlp(torch.concat(
            [output1, output2], dim=2
        ))
        return source_output.squeeze(2)

class FiLM(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.alpha = nn.Linear(in_features, out_features)
        self.beta = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        alpha = self.alpha(cond)
        beta = self.beta(cond)
        return alpha * x + beta

class DeepONet_bdry(nn.Module):
    
    def __init__(
        self, nx: int, ny: int,
        ) -> None:
        super().__init__()
        self.nx, self.ny = nx, ny
        self.linear1 = nn.Linear(2, 16)
        self.act1 = Waveact()
        self.linear2 = nn.Linear(16, 16)
        self.act2 = Waveact()
        self.linear3 = nn.Linear(16, 1)
        self.film1 = FiLM(2 * nx + 2 * ny, 16)
        self.film2 = FiLM(2 * nx + 2 * ny, 16)
        self.film3 = FiLM(2 * nx + 2 * ny, 1)
        
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(
        self, R: Tensor, Z: Tensor, bdry: Tensor
        ) -> Tensor:
        input_trunk = torch.concat((R, Z))
        output = self.linear1(input_trunk) # (1, N, 10)
        output = self.film1(output, bdry)
        output = self.act1(output)
        # output = self.linear2(output)
        # output = self.film2(output, bdry)
        # output = self.act2(output)
        output = self.linear3(output)
        output = self.film3(output, bdry)
        return output

class PINTO(nn.Module):
    def __init__(
        self, nx: int, ny: int,
        ) -> None:
        super().__init__()
        self.nx, self.ny = nx, ny
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, 10)
        )
        self.key_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, 10)
        )
        self.value_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 10)
        )
        self.MHA1 = nn.MultiheadAttention(
            embed_dim=10, num_heads=1, batch_first=True
            )
        self.mlp2 = nn.Sequential(
            nn.Linear(15, 32),
            nn.SiLU(),
            nn.Linear(32, 15)
        )
        self.MHA3 = nn.MultiheadAttention(
            embed_dim=10, num_heads=2, batch_first=True
            )
        self.mlp4 = nn.Sequential(
            nn.Linear(10, 50),
            nn.SiLU(),
            nn.Linear(50, 50)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight.data)
                nn.init.zeros_(m.in_proj_bias.data)
                nn.init.xavier_uniform_(m.out_proj.weight.data)
                nn.init.zeros_(m.out_proj.bias.data)
    
    def forward(self, R: Tensor, Z: Tensor, bdry_point: Tensor, bdry_value: Tensor) -> Tensor:
        X = torch.concat((R, Z), dim=2)
        Q = self.pos_encoder(X)
        K = self.key_encoder(bdry_point)
        V = self.value_encoder(bdry_value)
        Q = Q.expand(V.shape[0], -1, -1)
        K = K.expand(V.shape[0], -1, -1)
        Q1, _ = self.MHA1(Q, K, V)
        Q1 = Q1 + Q
        Q2 = self.mlp2(Q1)
        Q2 = Q2 + Q1
        Q2 = Q1
        Q3, _ = self.MHA3(Q2, K, V)
        Q3 = Q3 + Q2
        Q3 = Q2
        Q4 = self.mlp4(Q3)
        Q4 = Q4 + Q3
        Q4 = Q3
        out = self.decoder(Q4)
        return out


import jax
import jax.numpy as jnp
import equinox as eqx

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
    act1: Waveact_jax
    layer2: eqx.nn.Linear
    act2: Waveact_jax
    layer3: eqx.nn.Linear

    def __init__(self, hidden_dim: int, key) -> None:
        key1, key2, key3 = jax.random.split(key, 3)
        self.layer1 = eqx.nn.Linear(2, 64, key=key1, dtype=jnp.float64)
        self.act1 = Waveact_jax()
        self.layer2 = eqx.nn.Linear(64, 64, key=key2, dtype=jnp.float64)
        self.act2 = Waveact_jax()
        self.layer3 = eqx.nn.Linear(64, hidden_dim, key=key3, dtype=jnp.float64)
    def __call__(self, R: JaxArray, Z: JaxArray) -> JaxArray:
        x = jnp.concatenate((R, Z), axis=0)
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        return x.T

class Branch(eqx.Module):
    layers: eqx.nn.Sequential

    def __init__(
        self, input_channel: int, hidden_dim: int,
        nx: int, ny: int, key) -> None:
        key_conv, key_linear = jax.random.split(key, 2)
        key11, key12, key13 = jax.random.split(key_conv, 3)
        key21, key22, key23 = jax.random.split(key_linear, 3)
        
        self.layers = eqx.nn.Sequential((
            eqx.nn.Conv2d(
                input_channel, 16, kernel_size=3, padding=1,
                key=key11, dtype=jnp.float64
                ),
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Conv2d(
                16, 32, kernel_size=3, padding=1, key=key12, dtype=jnp.float64
                ),
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Conv2d(
                32, 64, kernel_size=3, padding=1, key=key13, dtype=jnp.float64
                ),
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Lambda(jnp.ravel),
            eqx.nn.Linear(
                64 * (nx // 8) * (ny // 8), 256, key=key21, dtype=jnp.float64
                ),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(256, 64, key=key22, dtype=jnp.float64),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(64, hidden_dim, key=key23, dtype=jnp.float64),
        ))

    def __call__(self, x: JaxArray) -> JaxArray:
        return self.layers(x)

class MLP_output(eqx.Module):
    layer1: eqx.nn.Linear
    act1: callable
    layer2: eqx.nn.Linear
    def __init__(self, hidden_dim: int, key) -> None:
        key1, key2 = jax.random.split(key, 2)
        self.layer1 = eqx.nn.Linear(hidden_dim * 2, 64, key=key1, dtype=jnp.float64)
        self.act1 = Waveact_jax()
        self.layer2 = eqx.nn.Linear(64, 1, key=key2, dtype=jnp.float64)
    def __call__(self, trunk: JaxArray, branch: JaxArray) -> JaxArray:
        x = jnp.concatenate((trunk, branch), axis=0)
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        return x

    
class DeepONet_resi_jax(eqx.Module):
    nx: int = eqx.field(static=True)
    ny: int = eqx.field(static=True)
    trunk: Trunk
    branch: Branch
    output_mlp: MLP_output
    def __init__(self, nx: int, ny: int, hidden_dim: int, key) -> None:
        key1, key2, key3 = jax.random.split(key, 3)
        self.nx, self.ny = nx, ny
        self.trunk = Trunk(hidden_dim, key1)
        self.branch = Branch(1, hidden_dim, nx, ny, key2)
        self.output_mlp = MLP_output(hidden_dim, key3)
    
    def __call__(
        self, R: JaxArray, Z: JaxArray, rhs: JaxArray
        ) -> JaxArray:
        output1 = self.trunk(R, Z)
        output2 = self.branch(rhs)
        output = self.output_mlp(output1, output2)
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
            eqx.nn.Linear(2, 64, key=key11),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(64, 30, key=key12)
        ))
        key21, key22 = jax.random.split(key2, 2)
        key_encoder = eqx.nn.Sequential((
            eqx.nn.Linear(2, 64, key=key21),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(64, 30, key=key22)
        ))
        self.key_encoder = eqx.filter_vmap(key_encoder)
        key31, key32 = jax.random.split(key3, 2)
        value_encoder = eqx.nn.Sequential((
            eqx.nn.Linear(1, 64, key=key31),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(64, 30, key=key32)
        ))
        self.value_encoder = eqx.filter_vmap(value_encoder)
        self.MHA1 = eqx.nn.MultiheadAttention(num_heads=2, query_size=30, key=key4)
        key51, key52 = jax.random.split(key5, 2)
        mlp2 = eqx.nn.Sequential((
            eqx.nn.Linear(30, 64, key=key51),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(64, 30, key=key52)
        ))
        self.mlp2 = eqx.filter_vmap(mlp2)
        self.MHA3 = eqx.nn.MultiheadAttention(num_heads=2, query_size=30, key=key6)
        key71, key72 = jax.random.split(key7, 2)
        mlp4 = eqx.nn.Sequential((
            eqx.nn.Linear(30, 64, key=key71),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(64, 30, key=key72)
        ))
        self.mlp4 = eqx.filter_vmap(mlp4)
        key81, key82 = jax.random.split(key8, 2)
        decoder = eqx.nn.Sequential((
            eqx.nn.Linear(30, 64, key=key81),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(64, 1, key=key82)
        ))
        self.decoder = eqx.filter_vmap(decoder)
    
    def __call__(
        self, R: JaxArray, Z: JaxArray, bdry_point: JaxArray, bdry_value: JaxArray
        ) -> JaxArray:
        X = jnp.concatenate((R, Z), axis=0)
        Q = self.pos_encoder(X)
        K = self.key_encoder(bdry_point)
        V = self.value_encoder(bdry_value)
        Q1 = self.MHA1(Q[None, :], K, V)
        Q1 = Q1 + Q
        Q2 = self.mlp2(Q1)
        Q2 = Q2 + Q1
        # Q2 = Q1
        Q3 = self.MHA3(Q2, K, V)
        Q3 = Q3 + Q2
        # Q3 = Q2
        Q4 = self.mlp4(Q3)
        Q4 = Q4 + Q3
        # Q4 = Q3
        output = self.decoder(Q4)
        return output
    
class Integratednet_jax(eqx.Module):
    resi_net: DeepONet_resi_jax
    bdry_net: PINTO_jax
    Rmin: float = eqx.field(static=True)
    Rmax: float = eqx.field(static=True)
    Zmin: float = eqx.field(static=True)
    Zmax: float = eqx.field(static=True)
    nR: int = eqx.field(static=True)
    nZ: int = eqx.field(static=True)

    def __init__(
        self, Rmin: float, Rmax: float, Zmin: float, Zmax: float, 
        nx: int, ny: int, hidden_dim: int, key
        ) -> None:
        key1, key2 = jax.random.split(key, 2)
        self.Rmin, self.Rmax = Rmin, Rmax
        self.Zmin, self.Zmax = Zmin, Zmax
        self.nR, self.nZ = nx, ny
        self.resi_net = DeepONet_resi_jax(nx, ny, hidden_dim, key1)
        self.bdry_net = PINTO_jax(key2)
    
    def transformation(
        self, R: JaxArray, Z: JaxArray
        ) -> Tuple[JaxArray, JaxArray]:
        x = (R - self.Rmin) / (self.Rmax - self.Rmin)
        y = (Z - self.Zmin) / (self.Zmax - self.Zmin)
        return x, y
    
    def lifting(self, x: JaxArray, y: JaxArray, rhs: JaxArray) -> JaxArray:
        zero = jnp.zeros(1, dtype=jnp.float64)
        one = jnp.ones(1, dtype=jnp.float64)
        lb = self.resi_net(x, zero, rhs) + self.resi_net(zero, y, rhs) - \
            self.resi_net(zero, zero, rhs)
        rb = self.resi_net(x, zero, rhs) + self.resi_net(one, y, rhs) - \
            self.resi_net(one, zero, rhs)
        rt = self.resi_net(one, y, rhs) + self.resi_net(x, one, rhs) - \
            self.resi_net(one, one, rhs)
        lt = self.resi_net(zero, y, rhs) + self.resi_net(x, one, rhs) - \
            self.resi_net(zero, one, rhs)
        return (1 - x) * (1 - y) * lb + x * (1 - y) * rb + \
            x * y * rt + (1 - x) * y * lt
        
    def __call__(
        self, R: JaxArray, Z: JaxArray, rhs: JaxArray,
        bdry_point: JaxArray, bdry_value: JaxArray
        ) -> Tuple[JaxArray, JaxArray]:
        scaling_rhs = rhs.max() - rhs.min() if rhs.size > 0 else 1.0
        rhs = rhs / scaling_rhs
        scaling_bdry = bdry_value.max() - bdry_value.min() if bdry_value.size > 0 else 1.0
        bdry_value = bdry_value / scaling_bdry
        x, y = self.transformation(R, Z)
        resi_output = self.resi_net(x, y, rhs)
        resi_output_lift = self.lifting(x, y, rhs)
        bdry_output = self.bdry_net(R, Z, bdry_point, bdry_value).reshape(1)
        return scaling_rhs * (resi_output - resi_output_lift) + scaling_bdry * bdry_output

class NKDeepONet(eqx.Module):
    trunk: Trunk
    branch_psi: Branch
    branch_tokamak: eqx.nn.Sequential
    mlp_output: eqx.nn.Sequential
    
    def __init__(
        self, nx: int, ny: int, hidden_dim: int,
        tokamak_input_len: int, output_dim: int, key
        ) -> None:
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.trunk = Trunk(hidden_dim, key1)
        self.branch_psi = Branch(
            input_channel=2, hidden_dim=hidden_dim, nx=nx, ny=ny, key=key2
            )
        key31, key32, key33 = jax.random.split(key3, 3)
        self.branch_tokamak = eqx.nn.Sequential((
            eqx.nn.Linear(tokamak_input_len, 64, key=key31),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(64, 64, key=key32),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(64, hidden_dim, key=key33)
        ))
        key41, key42 = jax.random.split(key4, 2)
        self.mlp_output = eqx.nn.Sequential((
            eqx.nn.Linear(3 * hidden_dim, 128, key=key41),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(128, output_dim, key=key42)
        ))
        self.initialize_weights_tokamak()
    
    def initialize_weights_tokamak(self) -> None:
        for layer in self.branch_tokamak.layers:
            if isinstance(layer, eqx.nn.Linear):
                layer = eqx.tree_at(
                    lambda l: l.weight, layer, layer.weight * 10e-4
                )
                if layer.bias is not None:
                    layer = eqx.tree_at(
                        lambda l: l.bias, layer, jnp.zeros_like(layer.bias)
                    )
    
    def __call__(
        self, R: JaxArray, Z: JaxArray, psi: JaxArray, tokamak_params: JaxArray
        ) -> JaxArray:
        trunk_out = self.trunk(R, Z)
        branch_psi_out = self.branch_psi(psi)
        branch_tokamak_out = self.branch_tokamak(tokamak_params)
        x = jnp.concatenate(
            (trunk_out, branch_psi_out, branch_tokamak_out), axis=0
            )
        output = self.mlp_output(x)
        return output
        
class XPlimnet(eqx.Module):
    common: eqx.nn.Sequential
    classification: eqx.nn.Sequential
    reconstruction: eqx.nn.Sequential

    def __init__(self, nx: int, ny: int, key) -> None:
        key1, key2, key3 = jax.random.split(key, 3)
        key11, key12 = jax.random.split(key1, 2)
        key21, key22, key23, key24= jax.random.split(key2, 4)
        key31, key32, key33, key34= jax.random.split(key3, 4)
        self.common = eqx.nn.Sequential((
            eqx.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, key=key11),
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, key=key12),
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            eqx.nn.Lambda(jax.nn.relu),
        ))

        self.reconstruction = eqx.nn.Sequential((
            eqx.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, key=key21),
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Lambda(jnp.ravel),
            eqx.nn.Linear(32 * (nx // 8) * (ny // 8), 128, key=key22),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(128, 128, key=key23),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(128, 2, key=key24),
        ))

        self.classification = eqx.nn.Sequential((
            eqx.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, key=key31),
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Lambda(jnp.ravel),
            eqx.nn.Linear(32 * (nx // 8) * (ny // 8), 128, key=key32),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(128, 128, key=key33),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(128, 1, key=key34),
            eqx.nn.Lambda(jax.nn.sigmoid)
        ))
    
    def __call__(self, psi: JaxArray) -> Tuple[JaxArray, JaxArray]:
        features = self.common(psi)
        reconstruction = self.reconstruction(features)
        classification = self.classification(features)
        return reconstruction, classification
    

class SpectralConv2d(eqx.Module):
    kernel_1_r: JaxArray
    kernel_1_i: JaxArray
    kernel_2_r: JaxArray
    kernel_2_i: JaxArray
    modes1: int = eqx.field(static=True)
    modes2: int = eqx.field(static=True)

    def __init__(
        self, in_channels: int, out_channels: int, 
        modes1: int, modes2: int, key
        ) -> None:
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.kernel_1_r = jax.random.normal(
            key1, (in_channels, out_channels, modes1, modes2)
            ) * scale
        self.kernel_1_i = jax.random.normal(
            key2, (in_channels, out_channels, modes1, modes2)
            ) * scale
        self.kernel_2_r = jax.random.normal(
            key3, (in_channels, out_channels, modes1, modes2)
            ) * scale
        self.kernel_2_i = jax.random.normal(
            key4, (in_channels, out_channels, modes1, modes2)
            ) * scale

    def __call__(self, x):
        # x.shape: [batch, in_channels, height, width]

        # Initialize parameters

        # Checking that the modes are not more than the input size

        # The model assumes real inputs and therefore uses a real
        # fft. For a 2D signal, the conjugate symmetry of the
        # transform is exploited to reduce the number of operations.
        # Given an input signal of dimesions (N, C, H, W), the
        # output signal will have dimensions (N, C, H, W//2+1).
        # Therefore the kernel weigths will have different dimensions
        # for the two axis.

        # Perform fft of the input
        x_ft = jnp.fft.rfftn(x, axes=(1, 2))

        # Multiply the center of the spectrum by the kernel
        out_ft = jnp.zeros_like(x_ft)
        s1 = jnp.einsum(
            'cij,coij->oij', x_ft[:, :self.modes1, :self.modes2], 
            self.kernel_1_r + 1j * self.kernel_1_i
            )
        s2 = jnp.einsum(
            'cij,coij->oij', x_ft[:, -self.modes1:, :self.modes2],
            self.kernel_2_r + 1j * self.kernel_2_i
            )
        out_ft = out_ft.at[:, :self.modes1, :self.modes2].set(s1)
        out_ft = out_ft.at[:, -self.modes1:, :self.modes2].set(s2)

        # Go back to the spatial domain
        y = jnp.fft.irfftn(out_ft, axes=(1, 2))
        return y

class FourierStage(eqx.Module):
    activation: Callable
    spectral_conv: SpectralConv2d
    conv: eqx.nn.Conv2d

    def __init__(
        self, in_channels: int, out_channels: int, 
        modes1: int, modes2: int, activation: Callable, key
        ):
        self.activation = activation
        key1, key2 = jax.random.split(key, 2)
        self.spectral_conv = SpectralConv2d(
            in_channels=in_channels, out_channels=out_channels, 
            modes1=modes1, modes2=modes2, key=key1
        )
        self.conv = eqx.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=(1, 1), key=key2
        )

    def __call__(self, x):
        x_fourier = self.spectral_conv(x)
        x_local = self.conv(x)
        return self.activation(x_fourier + x_local)


class FNO2D(eqx.Module):
    '''
    Fourier Neural Operator for 2D signals.

    Implemented from
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py

    Attributes:
        modes1: Number of modes in the first dimension.
        modes2: Number of modes in the second dimension.
        width: Number of channels to which the input is lifted.
        depth: Number of Fourier stages
        channels_last_proj: Number of channels in the hidden layer of the last
        2-layers Fully Connected (channel-wise) network
        activation: Activation function to use
        out_channels: Number of output channels, >1 for non-scalar fields.
    '''
    # modes1: int = 12
    # modes2: int = 12
    # width: int = 32
    # depth: int = 4
    # channels_last_proj: int = 128
    # activation: Callable = nn.gelu
    # out_channels: int = 1
    # padding: int = 0 # Padding for non-periodic inputs
    enc: eqx.nn.Linear
    fourier_stages: list
    film_stages: list
    dec: eqx.nn.Sequential
    padding: int = eqx.field(static=True)

    def __init__(
        self, input_dim: int, output_dim: int, channels_last_proj: int,
        num_constraints: int,
        modes1: int, modes2: int, width: int, depth: int,
        activation: Callable = jax.nn.gelu, key=jax.random.PRNGKey(0),
        padding: int=1
        ) -> None:

        key_enc, key_fourier, key_dec = jax.random.split(key, 3)
        self.padding = padding
        self.enc = eqx.nn.Linear(input_dim + 2, width, key=key_enc)

        self.fourier_stages = []
        self.film_stages = []

        key_fourier = jax.random.split(key_fourier, depth * 3)
        for depthnum in range(depth):
            acti = activation if depthnum < depth - 1 else lambda x: x
            self.fourier_stages.append(
                FourierStage(
                    in_channels=width,
                    out_channels=width,
                    modes1=modes1,
                    modes2=modes2,
                    activation=acti,
                    key=key_fourier[3 * depthnum]
                )
            )
            self.film_stages.append(eqx.nn.Sequential((
                eqx.nn.Linear(
                    num_constraints, width * 2, key=key_fourier[3 * depthnum + 1]
                    ),
                eqx.nn.Lambda(activation),
                eqx.nn.Linear(
                    width * 2, width * 2, key=key_fourier[3 * depthnum + 2]
                    )
            )))
        key_dec1, key_dec2 = jax.random.split(key_dec, 2)
        self.dec = eqx.nn.Sequential((
            eqx.nn.Linear(width, channels_last_proj, key=key_dec1),
            eqx.nn.Lambda(activation),
            eqx.nn.Linear(channels_last_proj, output_dim, key=key_dec2)
        ))
            

    def __call__(self, x: jnp.ndarray, constraints: jnp.ndarray) -> jnp.ndarray:
        # Generate coordinate grid, and append to input channels
        grid = self.get_grid(x)
        x = jnp.concatenate([x, grid], axis=0)

        # Lift the input to a higher dimension
        width, height = x.shape[1], x.shape[2]
        x = jnp.permute_dims(x, (1, 2, 0))
        x = x.reshape(-1, x.shape[-1])
        x = jax.vmap(self.enc)(x)
        x = x.reshape(width, height, x.shape[-1])
        x = jnp.permute_dims(x, (2, 0, 1))

        # Pad input
        if self.padding > 0:
            x = jnp.pad(
                x,
                ((0, 0), (0, self.padding), (0, self.padding)),
                mode='constant'
            )

        # Apply Fourier stages, last one has no activation
        # (can't find this in the paper, but is in the original code)
        for fourier_stage, film_stage in zip(
            self.fourier_stages, self.film_stages
            ):
            x = fourier_stage(x)
            gamma, beta = jnp.split(film_stage(constraints), 2, axis=0)
            x = gamma[:, None, None] * x + beta[:, None, None]

        # Unpad
        if self.padding > 0:
            x = x[:, :-self.padding, :-self.padding]

        # Project to the output channels
        x = jnp.permute_dims(x, (1, 2, 0))
        x = x.reshape(-1, x.shape[-1])
        x = jax.vmap(self.dec)(x)
        x = x.reshape(width, height, -1)
        x = jnp.permute_dims(x, (2, 0, 1))

        return x

    @staticmethod
    def get_grid(x):
        x1 = jnp.linspace(0, 1, x.shape[1])
        x2 = jnp.linspace(0, 1, x.shape[2])
        x1, x2 = jnp.meshgrid(x1, x2, indexing = 'ij')
        grid = jnp.stack([x1, x2], axis=0)
        return grid
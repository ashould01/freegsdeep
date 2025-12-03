import os
import torch
import torch.nn as nn
from freegsdeep.utilstyping import *

class PINTO(nn.Module):
    def __init__(
        self, Rmin: float, Rmax: float, Zmin: float, Zmax: float,
        nx: int, ny: int,
        ) -> None:
        super().__init__()
        self.R_grid = torch.linspace(Rmin, Rmax, nx)
        self.Z_grid = torch.linspace(Zmin, Zmax, ny)
        self.R_grid, self.Z_grid = torch.meshgrid(
            self.R_grid, self.Z_grid, indexing='ij'
            )
        self.nx, self.ny = nx, ny
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, 100),
            nn.SiLU(),
            nn.Linear(100, 100),
            nn.SiLU(),
            nn.Linear(100, 50)
        )
        self.key_encoder = nn.Sequential(
            nn.Linear(2, 100),
            nn.SiLU(),
            nn.Linear(100, 100),
            nn.SiLU(),
            nn.Linear(100, 50)
        )
        self.value_encoder = nn.Sequential(
            nn.Linear(1, 100),
            nn.SiLU(),
            nn.Linear(100, 100),
            nn.SiLU(),
            nn.Linear(100, 50)
        )
        self.MHA1 = nn.MultiheadAttention(
            embed_dim=50, num_heads=2, batch_first=True
            )
        self.mlp2 = nn.Sequential(
            nn.Linear(50, 100),
            nn.SiLU(),
            nn.Linear(100, 50)
        )
        self.MHA3 = nn.MultiheadAttention(
            embed_dim=50, num_heads=2, batch_first=True
            )
        self.mlp4 = nn.Sequential(
            nn.Linear(50, 50),
            nn.SiLU(),
            nn.Linear(50, 50)
        )
        self.decoder = nn.Sequential(
            nn.Linear(50, 100),
            nn.SiLU(),
            nn.Linear(100, 1)
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
        Q3, _ = self.MHA3(Q2, K, V)
        Q3 = Q3 + Q2
        Q4 = self.mlp4(Q3)
        Q4 = Q4 + Q3
        out = self.decoder(Q4)
        return out
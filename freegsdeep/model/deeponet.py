import torch
import torch.nn as nn
from freegsdeep.utilstyping import *
from typing import Tuple

class Waveact(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

class ResnetBlock(nn.Module):
    def __init__(
        self, in_channels: int, middle_channels: int, out_channels: int
        ) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(
            in_channels=in_channels, out_channels=middle_channels, kernel_size=3, padding=1
            ), nn.SiLU(), nn.Conv2d(
                in_channels=middle_channels, out_channels=out_channels,
                kernel_size=3, padding=1
                ))
        
    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        return x + out
        
class DeepONet(nn.Module):
    
    def __init__(
        self, Rmin: float, Rmax: float, Zmin: float, Zmax: float,
        nx: int, ny: int, mode: Literal['plain', 'green'] = 'plain'
        ) -> None:
        super().__init__()
        self.R = torch.linspace(Rmin, Rmax, nx)
        self.Z = torch.linspace(Zmin, Zmax, ny)
        self.nx, self.ny = nx, ny
        self.mode = mode
        self.trunk = nn.Sequential(
            nn.Linear(2, 100),
            Waveact(),
            # nn.Linear(100, 100),
            # Waveact(),
            nn.Linear(100, 100),
            Waveact(),
            nn.Linear(100, 50)
        )
        # Trunk net 에 GNN 구조 !
        self.branch_source = nn.Sequential(
            nn.Linear(nx * ny, 100),
            nn.SiLU(),
            # Waveact(),
            nn.Linear(100, 100),
            nn.SiLU(),
            # nn.SiLU(),
            # nn.Linear(128, 128),
            # nn.SiLU(),
            nn.Linear(100, 50)
        )
        # self.branch_spatial = nn.Sequential(
        #     nn.Linear(2, 10),
        #     nn.SiLU(),
        #     nn.Linear(10, 10)
        # )
        self.branch_boundaryL = nn.Sequential(
            nn.Linear(ny, 100),
            Waveact(),
            nn.Linear(100, 100)
        )
        self.branch_boundaryR = nn.Sequential(
            nn.Linear(ny, 100),
            Waveact(),
            nn.Linear(100, 100)
        )
        self.branch_boundaryU = nn.Sequential(
            nn.Linear(nx, 100),
            Waveact(),
            nn.Linear(100, 100)
        )
        self.branch_boundaryD = nn.Sequential(
            nn.Linear(nx, 100),
            Waveact(),
            nn.Linear(100, 100)
        )
        self.branch_boundary = nn.Sequential(
            nn.Linear(100, 100),
            nn.SiLU(),
            nn.Linear(100, 50)
        )
        self.mlp_source = nn.Sequential(
            ResnetBlock(1, 8, 8),
            nn.SiLU(),
            ResnetBlock(8, 8, 1)
        )
        self.mlp_boundary = nn.Sequential(
            ResnetBlock(1, 8, 8),
            nn.SiLU(),
            ResnetBlock(8, 8, 1)
        )

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)


    def forward(
        self, R: Tensor, Z: Tensor, rhs: Tensor, bdry: Tuple[Tensor],
        ) -> Tensor:
        input_trunk = torch.concat((R, Z), dim=2)
        output1 = self.trunk(input_trunk)
        output2 = self.branch_source(rhs) # B * w
        outputL = self.branch_boundaryL(bdry[:, 0])
        outputD = self.branch_boundaryD(bdry[:, 1])
        outputR = self.branch_boundaryR(bdry[:, 2])
        outputU = self.branch_boundaryU(bdry[:, 3])
        output3 = self.branch_boundary((
            outputL * outputD * outputR * outputU
            ))
        # v = self.mlp_boundary((output1 * output3[:, :, None])) # B * w
        source_output = (output1 * output2[:, None, :]).sum(dim=2)
        boundary_output = (output1 * output3[:, None, :]).sum(dim=2)
        if self.mode == 'plain':
            return source_output + boundary_output
            # return v + 
        elif self.mode == 'green':
            NotImplementedError
            # return torch.mean(self.mlp_source(
            #     (output1 * output2)[:, None, :, :].reshape(output2.shape[0], 1, self.nx, self.ny)
            # ), dim=1), self.boundary_integral(R, Z, v)
            # return None
        # (v_R, ) = torch.autograd.grad(
        #     v.sum(), R, retain_graph=True 
        # )
        # (v_Z, ) = torch.autograd.grad(
        #     v.sum(), Z
        # )

class DeepONet_resi(nn.Module):
    
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
        self.trunk = nn.Sequential(
            nn.Linear(2, 100),
            Waveact(),
            nn.Linear(100, 100),
            Waveact(),
            nn.Linear(100, 10)
        )
        self.branch_source = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        self.branch_mlp = nn.Sequential(
            nn.Linear(nx * ny, 100),
            nn.SiLU(),
            nn.Linear(100, 10)
        )
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
        output2 = self.branch_source(rhs) 
        output2 = output2.reshape(output2.shape[0], -1)
        output2 = self.branch_mlp(output2)
        output1 = output1.expand(output2.shape[0], -1, -1)
        output2 = output2[:, None, :].expand(-1, output1.shape[1], -1)
        source_output = self.output_mlp(torch.concat(
            [output1, output2], dim=2
        ))
        return source_output.squeeze(2)
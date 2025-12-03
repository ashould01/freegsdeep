import torch
import torch.nn as nn
from freegsdeep.utilstyping import *


class Waveact(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

class PINN(nn.Module):
    
    def __init__(self, depth: int = 3, width: int = 20):
        super().__init__()
        seq = [nn.Linear(2, width), Waveact()]
        for i in range(depth - 2):
            seq.append(nn.Linear(width, width))
            seq.append(Waveact())
        seq.append(nn.Linear(width, 1))
        self.fc = nn.Sequential(*seq)
        self.init_layers()
    
    def init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)
    
    def forward(self, R: Tensor, Z: Tensor) -> Tensor:
        x = torch.concat([R, Z], dim=1)
        out = self.fc(x)
        return out
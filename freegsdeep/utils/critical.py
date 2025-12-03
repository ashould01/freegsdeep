import torch
import torch.nn as nn
import torch.func
from freegsdeep.utilstyping import *

class find_critical():
    
    def __init__(
        self, R: Tensor, Z: Tensor, bdry_pt: Tensor,
        solver_resi: nn.Module, solver_bdry: nn.Module
        ) -> None:
        self.R = R
        self.Z = Z
        self.solver_resi = solver_resi
        self.solver_bdry = solver_bdry
        self.bdry_pt = bdry_pt

    def update(self, rhs: Tensor, bdry: Tensor) -> None:
        self.rhs = rhs
        self.bdry = bdry
        
    def __call__(self) -> Tuple[Optional[Array], Optional[Array]]:
        self.solver = lambda R, Z: self.solver_resi(
            R, Z, self.rhs
        ) + self.solver_bdry(
            R, Z, self.bdry_pt, self.bdry
        )
        Bp2 = torch.func.grad(
            self.solver(self.R, self.Z), argnums=(0, 1)
            ) ** 2 / self.R ** 2
        breakpoint()
            
        ...

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
from freegsdeep.model import Waveact, FiLM
from freegsdeep.train import Trainer_bdry_deeponet
from pyhessian import hessian_bdry
from pyhessian.density_plot import get_esd_plot

class DeepONet_bdry(nn.Module):
    
    def __init__(
        self, nx: int, ny: int, width: int, 
        ) -> None:
        super().__init__()
        self.nx, self.ny = nx, ny
        self.linear1 = nn.Linear(2, width)
        self.act1 = Waveact()
        self.linear2 = nn.Linear(width, width)
        self.act2 = Waveact()
        self.linear3 = nn.Linear(width, 1)
        self.film1 = FiLM(2 * nx + 2 * ny, width)
        self.film2 = FiLM(2 * nx + 2 * ny, width)
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
        output = self.linear2(output)
        output = self.film2(output, bdry)
        output = self.act2(output)
        output = self.linear3(output)
        output = self.film3(output, bdry)
        return output
        
mu0 = 4e-7 * torch.pi

model1 = DeepONet_bdry(
    65, 65, 32
).to(torch.float64).to('cuda:0')

model2 = DeepONet_bdry(
    65, 65, 32
).to(torch.float64).to('cuda:0')

model1.load_state_dict(torch.load(
    'logs/deeponet_bdry_251227_032441/model/model_deeponet_5.pt',
    map_location='cuda:0',
    weights_only=True
    ))

model2.load_state_dict(torch.load(
    'logs/deeponet_bdry_251227_032441/model/model_deeponet_6.pt',
    map_location='cuda:0',
    weights_only=True
    ))

batch_size = 25
trainer = Trainer_bdry_deeponet(
    0.1, 2.0, -1.0, 1.0, 65, 65, 100, batch_size, 50, 'data_100_2'
    )
dataloader = DataLoader(trainer.dataset, shuffle=True, batch_size=batch_size)
batch = next(iter(dataloader))
_, bdry, psi, _, _, _, _, U = batch
bdry = bdry.to('cuda:0')
soln = (psi + U * mu0).to('cuda:0').reshape(U.shape[0], -1)

def criterion(bdry: Tensor, soln: Tensor, model: nn.Module) -> Tensor:
    resi_loss_batch, bdry_loss_batch, true_loss_batch = \
        trainer.loss(bdry, soln, model)
    return resi_loss_batch + 30 * bdry_loss_batch 

hessian_comp1 = hessian_bdry(model1, criterion, bdry, soln)
eigen_list_full1, weight_list_full1 = hessian_comp1.density(iter=500, n_v=2)
eigen_list_full1 = np.sort(np.array(eigen_list_full1))

hessian_comp2 = hessian_bdry(model2, criterion, bdry, soln)
eigen_list_full2, weight_list_full2 = hessian_comp2.density(iter=500, n_v=2)
eigen_list_full2 = np.sort(np.array(eigen_list_full2))

get_esd_plot(eigen_list_full1, weight_list_full1, 'esd_bdry_nonconstant.png')
get_esd_plot(eigen_list_full2, weight_list_full2, 'esd_bdry_constant.png')
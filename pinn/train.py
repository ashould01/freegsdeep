import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from freegsdeep.typing import *
from freegsdeep.soap import SOAP
from torch.func import jacrev, vmap
from pinn.model import PINN
from pinn.dataset import *

class PINNTrainer():
    def __init__(
        self, dataset_type: Literal[
            'SolovevDataset', 'SolovevTriangularity', 'IterationDataset'
            ],
        num_resi: int = 4_000, num_bdry: int = 1_000, nR: int = 65, nZ: int = 65,
        Rmin: float = 0.1, Rmax: float = 2.0,
        Zmin: float = -1.0, Zmax: float = 1.0,
        kappa0: float = 1.5, q0: float = 1.5,
        a: float = 0.5, b: float = 0.7,
        ep: float = 0.3, lamda: float = 0.0, R0: float = 5 / 3,
        device: str = 'cuda:0', adam_epoch: int = 1000, lbfgs_epoch: int = 100,
        ) -> None:

        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.num_resi = num_resi
        self.num_bdry = num_bdry
        self.nR = nR
        self.nZ = nZ
        self.kappa0 = kappa0
        self.q0 = q0
        self.a = a
        self.b = b
        self.ep = ep
        self.lamda = lamda
        self.R0 = R0
        self.dataset_type = dataset_type
        self.device = device
        self.adam_epoch = adam_epoch
        self.lbfgs_epoch = lbfgs_epoch
        self.model = PINN(depth=5, width=100).to(device).to(torch.float64)

    def load_dataset(self) -> None:
        if self.dataset_type == 'SolovevDataset':
            self.dataset = SolovevDataset(
                self.Rmin, self.Rmax, self.Zmin, self.Zmax, 
                self.num_resi, self.num_bdry, self.kappa0, self.q0
                )
        elif self.dataset_type == 'SolovevTriangularity':
            self.dataset = SolovevTriangularity(
                self.num_resi, self.num_bdry, 
                self.a, self.b, self.ep, self.lamda, self.R0
                )
        elif self.dataset_type == 'IterationDataset':
            self.dataset = IterationDataset(
                self.Rmin, self.Rmax, self.Zmin, self.Zmax, self.nR, self.nZ
            )
        else:
            NotImplementedError

    def weight_update(
        self, losses: Tuple[float], weight: Tensor, alpha: float = 0.0
        ):
        params_named = [p for _, p in self.model.named_parameters() if p.requires_grad]
        global_norm = []
        chunks = []
        for loss in losses:
            grads = torch.autograd.grad(
                loss, params_named, allow_unused=True, retain_graph=True
                )
            for g in grads:
                if g is None:
                    continue
                chunks.append(g.reshape(-1))
            if len(chunks) == 0:
                norm = 0.0
            else:
                norm = torch.concat(chunks).norm()
            global_norm.append(norm)
            loss.detach()
        global_norm = torch.tensor(global_norm)
        update_weight = alpha * weight + (1 - alpha) * global_norm.sum() / global_norm
        update_weight_sum = update_weight.sum()
        return update_weight / update_weight_sum
        
    def _compute_single_batch(self, R: Tensor, Z: Tensor):
        psi = self.model.forward(
            R.unsqueeze(0).unsqueeze(-1),
            Z.unsqueeze(0).unsqueeze(-1)
            )
        return psi.squeeze(0).squeeze(-1)
        
    def lossiter(
        self, X_resi: Tensor, X_bdry: Tensor, rhs: Tensor, true_bdry: Tensor,
        ) -> Tuple[float]:

        R_resi, Z_resi = X_resi[:, 0], X_resi[:, 1]
        grad_point_single_R = jacrev(self._compute_single_batch, argnums=0)
        grad_point_single_Z = jacrev(self._compute_single_batch, argnums=1)
        grad_point_single_RR = jacrev(grad_point_single_R, argnums=0)
        grad_point_single_ZZ = jacrev(grad_point_single_Z, argnums=1)
        psi_R = vmap(grad_point_single_R)(R_resi, Z_resi)
        psi_RR = vmap(grad_point_single_RR)(R_resi, Z_resi)
        psi_ZZ = vmap(grad_point_single_ZZ)(R_resi, Z_resi)

        psi_resi_loss = torch.mean(
            ((psi_RR - psi_R / R_resi + psi_ZZ) - rhs) ** 2
        )
        psi_bdry = self.model.forward(X_bdry[:, 0:1], X_bdry[:, 1:2]).squeeze(-1)
        psi_bdry_loss = torch.mean((psi_bdry - true_bdry) ** 2)
        x_pt_R = torch.tensor([1.1, 1.1], dtype=torch.float64, device=self.device)
        x_pt_Z = torch.tensor([-0.6, 0.8], dtype=torch.float64, device=self.device)
        psi_x_R = vmap(grad_point_single_R)(x_pt_R, x_pt_Z)
        psi_x_Z = vmap(grad_point_single_Z)(x_pt_R, x_pt_Z)
        psi_Br = -psi_x_Z / x_pt_R + self.psi_coil_Br
        psi_Bz = psi_x_R / x_pt_R + self.psi_coil_Bz
        psi_x_loss = torch.mean(psi_Br ** 2 + psi_Bz ** 2)
        psi_isoflux = self.model.forward(
            torch.tensor([[1.1], [1.1]], dtype=torch.float64, device=self.device),
            torch.tensor([[-0.6], [0.6]], dtype=torch.float64, device=self.device)
            ).squeeze()
        psi_isoflux = psi_isoflux + self.psi_coil_isoflux
        psi_isoflux_loss = ((psi_isoflux[0] - psi_isoflux[1]) ** 2).squeeze()
        
        return psi_resi_loss, psi_bdry_loss, psi_x_loss, psi_isoflux_loss

    def loss(
        self, X_resi: Tensor, X_bdry: Tensor, rhs: Tensor, true_bdry: Tensor,
        ) -> Tuple[float]:

        R_resi, Z_resi = X_resi[:, 0], X_resi[:, 1]
        grad_point_single_R = jacrev(self._compute_single_batch, argnums=0)
        grad_point_single_Z = jacrev(self._compute_single_batch, argnums=1)
        grad_point_single_RR = jacrev(grad_point_single_R, argnums=0)
        grad_point_single_ZZ = jacrev(grad_point_single_Z, argnums=1)
        psi_R = vmap(grad_point_single_R)(R_resi, Z_resi)
        psi_RR = vmap(grad_point_single_RR)(R_resi, Z_resi)
        psi_ZZ = vmap(grad_point_single_ZZ)(R_resi, Z_resi)

        psi_resi_loss = torch.mean(
            ((psi_RR - psi_R / R_resi + psi_ZZ) - rhs) ** 2
        )
        psi_bdry = self.model.forward(X_bdry[:, 0:1], X_bdry[:, 1:2]).squeeze(-1)
        psi_bdry_loss = torch.mean((psi_bdry - true_bdry) ** 2)
        
        return psi_resi_loss, psi_bdry_loss
    
    def losstriang(
        self, X_resi: Tensor, X_bdry: Tensor, rhs: Tensor, true_bdry: Tensor
        ) -> Tuple[float]:
        
        _x, _y = X_resi[:, 0], X_resi[:, 1]
        grad_point_single_x = jacrev(self._compute_single_batch, argnums=0)
        grad_point_single_y = jacrev(self._compute_single_batch, argnums=1)
        grad_point_single_xx = jacrev(grad_point_single_x, argnums=0)
        grad_point_single_yy = jacrev(grad_point_single_y, argnums=1)
        psi_x = vmap(grad_point_single_x)(_x, _y)
        psi_xx = vmap(grad_point_single_xx)(_x, _y)
        psi_yy = vmap(grad_point_single_yy)(_x, _y)

        psi_resi_loss = torch.mean(
            ((psi_xx + psi_yy - self.ep / ((self.ep + 1) * _x) * psi_x) + rhs) ** 2
        )
        psi_bdry = self.model.forward(X_bdry[:, 0:1], X_bdry[:, 1:2])
        psi_bdry_loss = torch.mean(psi_bdry.squeeze() ** 2)

        return psi_resi_loss, psi_bdry_loss
    
    def _compute_boundary(self, x: Tensor):
        return self.b / self.a * torch.sqrt(
            (1 - (x - 0.5 * self.ep * (1 - x ** 2)) ** 2) / \
            ((1 - 0.25 * self.ep ** 2) * (1 + self.ep * x) ** 2 + \
                self.lamda * x * (1 + 0.5 * self.ep * x))
        )

    def image(self, epoch: int, optim: Literal['adam', 'lbfgs'], save_name: str):
        if self.dataset_type == 'SolovevDataset':
            R, Z = np.meshgrid(
                np.linspace(self.Rmin, self.Rmax, 65),
                np.linspace(self.Zmin, self.Zmax, 65),
                indexing='ij'
                )
            R = R.reshape(-1, 1)
            Z = Z.reshape(-1, 1)
            extent = [R.min(), R.max(), Z.min(), Z.max()]
            levels = np.arange(0.0, 0.8, 0.1)
            R_input = torch.from_numpy(R)
            Z_input = torch.from_numpy(Z)
            R_input = R_input.to(self.device)
            Z_input = Z_input.to(self.device)
            psi_predict = self.model(R_input, Z_input)
            psi_predict = psi_predict.detach().cpu().numpy().reshape(65, 65)
            psi_true = (R ** 2 * Z ** 2 + self.kappa0 ** 2 / 4 * (
                R ** 2 - 1
                ) ** 2) / (2 * self.kappa0 * self.q0)
            psi_true = psi_true.reshape(65, 65)
            fig, ax = plt.subplots(1, 3, figsize=(22, 7))
            c0 = ax[0].imshow(psi_predict, extent=extent, aspect='equal')
            cs0 = ax[0].contour(
                R.reshape(65, 65), Z.reshape(65, 65), psi_predict,
                levels=levels, colors='white', linestyles='--'
            )
            ax[0].clabel(cs0, inline=True, fontsize=10)
            fig.colorbar(c0, ax=ax[0])
            c1 = ax[1].imshow(psi_true, extent=extent, aspect='equal')
            cs1 = ax[1].contour(
                R.reshape(65, 65), Z.reshape(65, 65), psi_true,
                levels=levels, colors='white', linestyles='--'
            )
            ax[1].clabel(cs1, inline=True, fontsize=10)
            fig.colorbar(c1, ax=ax[1])
            c2 = ax[2].imshow(
                abs(psi_predict - psi_true), extent=extent,
                aspect='equal', cmap='hot'
                )
            fig.colorbar(c2, ax=ax[2])
            fig.savefig(os.path.join(
                'figures', f'PINN_{self.dataset_type}', save_name, 
                f'PINN_{optim}_{epoch+1}.png'))
            plt.close()
            print(f'Absolute L2 error : {np.linalg.norm((psi_predict - psi_true), 2):.6e}')
            print(f'Relative L2 error : {np.linalg.norm((psi_predict - psi_true), 2) / \
                np.linalg.norm(psi_true, 2):.6e}')
        elif self.dataset_type == 'IterationDataset':
            R, Z = np.meshgrid(
                np.linspace(self.Rmin, self.Rmax, 65),
                np.linspace(self.Zmin, self.Zmax, 65),
                indexing='ij'
                )
            R_input = R.reshape(-1, 1)
            Z_input = Z.reshape(-1, 1)
            extent = [R.min(), R.max(), Z.min(), Z.max()]
            levels = np.arange(0.0, 0.3, 0.03)
            R_input = torch.from_numpy(R_input)
            Z_input = torch.from_numpy(Z_input)
            R_input = R_input.to(self.device)
            Z_input = Z_input.to(self.device)
            psi_predict = self.model(R_input, Z_input)
            psi_predict = psi_predict.detach().cpu().numpy().reshape(65, 65)
            psi_true = self.dataset.true.numpy()

            fig, ax = plt.subplots(2, 3, figsize=(22, 15))
            c0 = ax[0, 0].imshow(
                psi_predict.T, extent=extent, aspect='equal', origin='lower'
                )
            cs0 = ax[0, 0].contour(
                R.reshape(65, 65), Z.reshape(65, 65), psi_predict,
                levels=levels, colors='white', linestyles='--'
            )
            ax[0, 0].clabel(cs0, inline=True, fontsize=10)
            fig.colorbar(c0, ax=ax[0, 0])

            c1 = ax[0, 1].imshow(
                psi_true.T, extent=extent, aspect='equal', origin='lower'
                )
            cs1 = ax[0, 1].contour(
                R.reshape(65, 65), Z.reshape(65, 65), psi_true,
                levels=levels, colors='white', linestyles='--'
            )
            ax[0, 1].clabel(cs1, inline=True, fontsize=10)
            fig.colorbar(c1, ax=ax[0, 1])
            c2 = ax[0, 2].imshow(
                abs(psi_predict - psi_true).T, extent=extent,
                aspect='equal', cmap='hot', origin='lower'
                )
            fig.colorbar(c2, ax=ax[0, 2])
            c3 = ax[1, 0].imshow(
                (psi_predict + self.psi_coil_grid).T, extent=extent,
                aspect='equal', origin='lower'
                )
            fig.colorbar(c3, ax=ax[1, 0])
            ax[1, 0].scatter([1.1, 1.1], [-0.6, 0.8], marker='x', color='red')
            ax[1, 0].scatter([1.1, 1.1], [-0.6, 0.6], marker='o', color='blue')
            psi_x_value = self.model.forward(
                torch.tensor([[1.1], [1.1]], dtype=torch.float64, device=self.device),
                torch.tensor([[-0.6], [0.8]], dtype=torch.float64, device=self.device)
                )
            psi_x_value = psi_x_value.detach().cpu().numpy().squeeze()
            psi_x_value = (psi_x_value + self.dataset.psi_coil_x)[np.argmax(np.abs(
                psi_x_value + self.dataset.psi_coil_x
                ))]
            ax[1, 0].contour(
                R, Z, psi_predict + self.psi_coil_grid,
                levels=[psi_x_value.item()], colors='black', origin='lower'
                )
            ax[1, 1].scatter([1.1, 1.1], [-0.6, 0.8], marker='x', color='red')
            ax[1, 1].scatter([1.1, 1.1], [-0.6, 0.6], marker='o', color='blue')
            c4 = ax[1, 1].imshow(
                (psi_true + self.psi_coil_grid).T, extent=extent,
                aspect='equal', origin='lower'
                )
            fig.colorbar(c4, ax=ax[1, 1])
            ax[1, 1].contour(
                R, Z, psi_true + self.psi_coil_grid,
                levels=[self.dataset.bdry_val_list[-1]], colors='black', origin='lower'
                )
            fig.savefig(os.path.join(
                'figures', f'PINN_{self.dataset_type}', save_name, 
                f'PINN_{optim}_{epoch+1}.png'))
            plt.close()
            print('Absolute L2 error : ', np.mean((psi_predict - psi_true) ** 2))
            print('Relative L2 error : ', np.mean(
                (psi_predict - psi_true) ** 2
                ) / np.mean(psi_true ** 2))
        
        elif self.dataset_type == 'SolovevTriangularity':
            x_lin = np.linspace(-1.0, 1.0, 65)
            y_lin = self._compute_boundary(
                torch.from_numpy(x_lin)
                ).detach().numpy()
            x, y = np.meshgrid(
                x_lin,
                np.linspace(-(y_lin.max()), y_lin.max(), 65),
                indexing='ij'
                )
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            extent = [x_lin.min(), x_lin.max(), -(y_lin.max()), y_lin.max()]
            levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            x_input = torch.from_numpy(x)
            y_input = torch.from_numpy(y)
            x_input = x_input.to(self.device)
            y_input = y_input.to(self.device)
            psi_predict = self.model.forward(x_input, y_input)
            psi_predict = psi_predict.detach().cpu().numpy().reshape(65, 65)
            psi_predict = psi_predict
            psi_true = 1 - (x - 0.5 * self.ep * (1 - x ** 2)) ** 2 - \
                ((1 - 0.25 * self.ep ** 2) * (1 + self.ep * x) ** 2 + \
                self.lamda * x * (1 + 0.5 * self.ep * x)) * (self.a / self.b * y) ** 2
            psi_true = psi_true.squeeze().reshape(65, 65)
            fig, ax = plt.subplots(1, 3, figsize=(22, 7))
            c0 = ax[0].imshow(psi_predict, extent=extent, aspect='equal')
            cs0 = ax[0].contour(
                x.reshape(65, 65), y.reshape(65, 65), psi_predict,
                levels=levels
            )
            ax[0].clabel(cs0, inline=True, fontsize=10)
            fig.colorbar(c0, ax=ax[0])
            ax[0].contour(
                x.reshape(65, 65), y.reshape(65, 65), psi_predict,
                levels=[0.0], colors='black'
                )
            c1 = ax[1].imshow(psi_true, extent=extent, aspect='equal')
            fig.colorbar(c1, ax=ax[1])
            ax[1].plot(x_lin, y_lin, color='black')
            ax[1].plot(x_lin, -y_lin, color='black')
            cs1 = ax[1].contour(
                x.reshape(65, 65), y.reshape(65, 65), psi_true,
                levels=levels
            )
            ax[1].clabel(cs0, inline=True, fontsize=10)
            c2 = ax[2].imshow(
                abs(psi_predict - psi_true), extent=extent,
                aspect='equal', cmap='hot'
                )
            fig.colorbar(c2, ax=ax[2])
            fig.savefig(os.path.join(
                'figures', f'PINN_{self.dataset_type}', save_name, 
                f'PINN_{optim}_{epoch+1}.png'))
            plt.close()
            print('Absolute L2 error : ', torch.mean((psi_predict - psi_true) ** 2))
            print('Relative L2 error : ', torch.mean(
                (psi_predict - psi_true) ** 2
                ) / torch.mean(psi_true ** 2))
        else:
            NotImplementedError
            
    def train(self, save_name: str) -> None:
        self.load_dataset()
        os.makedirs(f'figures/PINN_{self.dataset_type}/{save_name}', exist_ok=True)
        print("Phase 1 : Adam")
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-2)
        # optim = SOAP(self.model.parameters(), lr=3e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.ExponentialLR(
            optim, 0.9, last_epoch=-1
        )
        resi = self.dataset.residual
        bdry = self.dataset.boundary
        rhs = self.dataset.source
        true_bdry = self.dataset.true_bdry

        if self.dataset_type == 'SolovevDataset':
            loss_func = self.loss
            fig, ax0 = plt.subplots(figsize=(15, 15))
            ax0.scatter(
                resi[:, 0], resi[:, 1], color='red', s=100.0, marker='x',
                )
            ax0.scatter(
                bdry[:, 0], bdry[:, 1], color='black', s=25.0, marker='o',
                )
            ax0.set_aspect('equal')
            ax0.set_xlabel('R', fontsize=15)
            ax0.set_ylabel('Z', fontsize=15)
            fig.suptitle('Collocation Points and Boundary Points', fontsize=20)
            fig.savefig(f'figures/PINN_{self.dataset_type}/{save_name}/collocation.png')
        elif self.dataset_type == 'IterationDataset':
            loss_func = self.lossiter
            self.psi_coil_isoflux = self.dataset.psi_coil_isoflux.to(self.device)
            self.psi_coil_Br = self.dataset.psi_coil_Br.to(self.device)
            self.psi_coil_Bz = self.dataset.psi_coil_Bz.to(self.device)
            self.psi_coil_grid = self.dataset.psi_coil_grid

            fig = plt.figure(figsize=(22, 7))
            gs = fig.add_gridspec(1, 3, wspace=0.25)
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.scatter(resi[:, 0], resi[:, 1], color='red', s=3.0, marker='x')
            ax0.scatter(bdry[:, 0], bdry[:, 1], color='black', s=1.5, marker='o')
            extent = [
                resi[:, 0].min(), resi[:, 0].max(),
                resi[:, 1].min(), resi[:, 1].max()
                ]
            ax1 = fig.add_subplot(gs[0, 1])
            c = ax1.imshow(rhs.reshape(65, 65).T, extent=extent, origin='lower')
            # ax1.contour(
            #     resi[:, 0].reshape(65, 65), resi[:, 1].reshape(65, 65), rhs.reshape(65, 65),
            #     levels=[-0.001], colors='black'
            #     )
            fig.colorbar(c, ax=ax1)
            ax2 = fig.add_subplot(gs[0, 2], projection='3d')
            len_plt_bdry = len(bdry) // 4
            for ii in range(4):
                ax2.plot(
                    bdry[ii * len_plt_bdry:(ii + 1) * len_plt_bdry, 0],
                    bdry[ii * len_plt_bdry:(ii + 1) * len_plt_bdry, 1],
                    true_bdry[ii * len_plt_bdry:(ii + 1) * len_plt_bdry]
                )
            fig.savefig(f'figures/PINN_{self.dataset_type}/{save_name}/collocation.png')
        elif self.dataset_type == 'SolovevTriangularity':
            loss_func = self.losstriang

        resi = resi.to(self.device)
        rhs = rhs.to(self.device)
        bdry = bdry.to(self.device)
        true_bdry = true_bdry.to(self.device)

        if self.dataset_type in ['SolovevDataset', 'SolovevTriangularity']:
            weight = torch.tensor((0.5, 0.5), dtype=torch.float64)
            for epoch in range(self.adam_epoch):
                psi_resi_loss, psi_bdry_loss \
                    = loss_func(resi, bdry, rhs, true_bdry)
                optim.zero_grad()
                loss = weight[0] * psi_resi_loss + \
                    weight[1] * psi_bdry_loss
                loss.backward()
                psi_resi_loss = psi_resi_loss.detach().item()
                psi_bdry_loss = psi_bdry_loss.detach().item()
                optim.step()
                sched.step()
                
                print(
                    f"Epoch {epoch:04d} | Loss : {loss.item():.4e} = " \
                    f"{weight[0]:.4f} * {psi_resi_loss:.4e} + " \
                    f"{weight[1]:.4f} * {psi_bdry_loss:.4e}" \
                    )
                if (epoch + 1) % 10 == 0:
                    psi_resi_loss_batch, psi_bdry_loss_batch \
                        = loss_func(resi, bdry, rhs, true_bdry)
                    weight = self.weight_update(
                        (psi_resi_loss_batch, psi_bdry_loss_batch),
                        weight,
                    )
                if (epoch + 1) % 100 == 0:
                    self.image(epoch, 'adam', save_name)

        elif self.dataset_type == 'IterationDataset':
            weight = torch.tensor((0.25, 10 * 0.25, 0.25, 0.25), dtype=torch.float64)
            for epoch in range(self.adam_epoch):
                psi_resi_loss, psi_bdry_loss, psi_x_loss, psi_isoflux_loss \
                    = loss_func(resi, bdry, rhs, true_bdry)
                optim.zero_grad()
                loss = weight[0] * psi_resi_loss \
                    + weight[1] * psi_bdry_loss \
                    + weight[2] * psi_x_loss \
                    + weight[3] * psi_isoflux_loss
                loss.backward()
                psi_resi_loss = psi_resi_loss.detach().item()
                psi_bdry_loss = psi_bdry_loss.detach().item()
                optim.step()
                sched.step()
                
                print(
                    f"Epoch {epoch:04d} | Loss : {loss.item():.4e} = " \
                    f"{weight[0]:.4f} * {psi_resi_loss:.4e} + " \
                    f"{weight[1]:.4f} * {psi_bdry_loss:.4e} + " \
                    f"{weight[2]:.4f} * {psi_x_loss:.4e} + " \
                    f"{weight[3]:.4f} * {psi_isoflux_loss:.4e}" \
                    )
                if (epoch + 1) % 10 == 0:
                    psi_resi_loss, psi_bdry_loss, \
                    psi_x_loss, psi_isoflux_loss \
                        = loss_func(resi, bdry, rhs, true_bdry)
                    weight = self.weight_update((
                        psi_resi_loss, psi_bdry_loss,
                        psi_x_loss, psi_isoflux_loss
                        ), weight)
                if (epoch + 1) % 100 == 0:
                    self.image(epoch, 'adam', save_name)

        if self.dataset_type != 'IterationDataset':
            self.load_dataset()
        optim = torch.optim.LBFGS(self.model.parameters())
            
        print("Phase 2 : L-BFGS")
        resi = self.dataset.residual
        bdry = self.dataset.boundary
        rhs = self.dataset.source
        true_bdry = self.dataset.true_bdry
        resi = resi.to(self.device)
        rhs = rhs.to(self.device)
        bdry = bdry.to(self.device)
        true_bdry = true_bdry.to(self.device)

        if self.dataset_type in ['SolovevDataset', 'SolovevTriangularity']:
            for epoch in range(self.lbfgs_epoch):
                psi_resi_loss_track = [] 
                psi_bdry_loss_track = [] 
                def closure():
                    psi_resi_loss, psi_bdry_loss \
                        = loss_func(resi, bdry, rhs, true_bdry)
                    optim.zero_grad()
                    loss = weight[0] * psi_resi_loss + \
                        weight[1] * psi_bdry_loss
                    loss.backward()
                    psi_resi_loss_track.append(psi_resi_loss.item())
                    psi_bdry_loss_track.append(psi_bdry_loss.item())
                    return loss
                loss_batch = optim.step(closure)
                loss = loss_batch.item()

                psi_resi_loss = psi_resi_loss_track[-1]
                psi_bdry_loss = psi_bdry_loss_track[-1]
                
                print(
                    f"Epoch {epoch:04d} | Loss : {loss:.4e} = " \
                    f"{psi_resi_loss:.4e} + " \
                    f"{psi_bdry_loss:.4e}" \
                    )
                if (epoch + 1) % 100 == 0:
                    self.image(epoch, 'lbfgs', save_name)

        elif self.dataset_type == 'IterationDataset':
            for epoch in range(self.lbfgs_epoch):
                psi_resi_loss_track, psi_bdry_loss_track = [], []
                psi_x_loss_track, psi_isoflux_loss_track = [], []
                def closure():
                    psi_resi_loss, psi_bdry_loss, psi_x_loss, psi_isoflux_loss \
                        = loss_func(resi, bdry, rhs, true_bdry)
                    optim.zero_grad()
                    loss = weight[0] * psi_resi_loss \
                        + weight[1] * psi_bdry_loss \
                        # + weight[2] * psi_x_loss + \
                        # + weight[3] * psi_isoflux_loss
                    loss.backward()
                    psi_resi_loss_track.append(psi_resi_loss.item())
                    psi_bdry_loss_track.append(psi_bdry_loss.item())
                    psi_x_loss_track.append(psi_x_loss.item())
                    psi_isoflux_loss_track.append(psi_isoflux_loss.item())
                    return loss
                loss = optim.step(closure)

                psi_resi_loss = psi_resi_loss_track[-1]
                psi_bdry_loss = psi_bdry_loss_track[-1]
                psi_x_loss = psi_x_loss_track[-1]
                psi_isoflux_loss = psi_isoflux_loss_track[-1]
                
                print(
                    f"Epoch {epoch:04d} | Loss : {loss.item():.4e} = " \
                    f"{weight[0]:.4f} * {psi_resi_loss:.4e} + " \
                    f"{weight[1]:.4f} * {psi_bdry_loss:.4e} + " \
                    f"{weight[2]:.4f} * {psi_x_loss:.4e} + " \
                    f"{weight[3]:.4f} * {psi_isoflux_loss:.4e}" \
                    )

                if (epoch + 1) % 10 == 0:
                    psi_resi_loss, psi_bdry_loss, \
                    psi_x_loss, psi_isoflux_loss \
                        = loss_func(resi, bdry, rhs, true_bdry)
                    weight = self.weight_update((
                        psi_resi_loss, psi_bdry_loss,
                        psi_x_loss, psi_isoflux_loss
                        ), weight)

                if (epoch + 1) % 50 == 0:
                    self.image(epoch, 'lbfgs', save_name)

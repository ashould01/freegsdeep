import os
tex_bin = "/data1/home/ahn40200393/texlive/2025/bin/x86_64-linux"
os.environ["PATH"] = tex_bin + ":" + os.environ.get("PATH", "")
import math
import torch
from torch.func import jacfwd, jacrev, vmap, hessian
from torch.utils.data import DataLoader, random_split, Subset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from datetime import datetime
from freegsdeep.model import PINTO, DeepONet_resi, DeepONet_bdry
from freegsdeep.typing import *
from freegsdeep.dataset.dataset import GSrhsdataset
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from matplotlib import colors
from matplotlib.ticker import FuncFormatter

class Trainer_resi():
    def __init__(
        self, Rmin: float, Rmax: float, Zmin: float, Zmax: float,
        nR: int, nZ: int, len_data: int, batch_size: int,
        epoch: int, hard: bool,
        data_load_path: Optional[str]=None, data_save_path: Optional[str]=None,
        device: Optional[str] = 'cuda:0'
        ) -> None:
        self.device = device
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.nR = nR
        self.nZ = nZ
        self.batch_size = batch_size
        self.epoch = epoch
        self.hard = hard
        self.dataset = GSrhsdataset(
            self.Rmin, self.Rmax, self.Zmin, self.Zmax, self.nR, self.nZ, 
            len_data, load_path=data_load_path, save_path=data_save_path
        )

    def _compute_single_batch_trunk(self, R: Tensor, Z: Tensor):
        psi = self.model.trunk.forward(torch.concat(
            [R.unsqueeze(0).unsqueeze(-1), Z.unsqueeze(0).unsqueeze(-1)], dim=-1
            ))
        return psi.squeeze(0).squeeze(-1)

    def _compute_single_batch_mlp(self, h: Tensor, b: Tensor):
        y = self.model.output_mlp.forward(torch.concat(
            [h.unsqueeze(0).unsqueeze(1), b.unsqueeze(0).unsqueeze(1)], dim=-1
        ))
        return y.squeeze(0).squeeze(1).squeeze(-1)

    def zero_bdry(
        self, R: Tensor, Z: Tensor, rhs: Tensor
        ) -> Tuple[Tensor]:
        nonzero_idx = torch.isclose(
            rhs.abs(), torch.zeros_like(rhs), atol=1e-6
            ).to(torch.float64).to(self.device).squeeze(1)
        R_grid, Z_grid = torch.meshgrid(
            torch.linspace(self.Rmin, self.Rmax, self.nR),
            torch.linspace(self.Zmin, self.Zmax, self.nZ),
            indexing='ij'
        )
        R_grid = R_grid.to(self.device)
        Z_grid = Z_grid.to(self.device)
        R_center = (R_grid * nonzero_idx).sum(dim=1).sum(dim=1) / nonzero_idx.sum(dim=1).sum(dim=1)
        Z_center = (Z_grid * nonzero_idx).sum(dim=1).sum(dim=1) / nonzero_idx.sum(dim=1).sum(dim=1)
        R_center = R_center[:, None]
        Z_center = Z_center[:, None]
        pR = R_center - self.Rmin
        qR = self.Rmax - R_center
        pZ = Z_center - self.Zmin
        qZ = self.Zmax - Z_center
        R = R[None, :]
        Z = Z[None, :]
        b = torch.where(
            ((R - self.Rmin) < 0.02) | ((self.Rmax - R) < 0.02) | \
            ((Z - self.Zmin) < 0.02) | ((self.Zmax - Z) < 0.02),
            torch.tensor(0.0, dtype=torch.float64, device=self.device),
            (R - self.Rmin) ** pR * (self.Rmax - R) ** qR * \
            (Z - self.Zmin) ** pZ * (self.Zmax - Z) ** qZ
        )
        db_dR = torch.where(
            ((R - self.Rmin) < 0.02) | ((self.Rmax - R) < 0.02) | \
            ((Z - self.Zmin) < 0.02) | ((self.Zmax - Z) < 0.02),
            torch.tensor(0.0, dtype=torch.float64, device=self.device),
            ((pR * (R - self.Rmin) ** (pR - 1) * (self.Rmax - R) ** qR) - \
            (qR * (R - self.Rmin) ** pR * (self.Rmax - R) ** (qR - 1))) * \
            (Z - self.Zmin) ** pZ * (self.Zmax - Z) ** qZ, 
            )
        db_dZ = torch.where(
            ((R - self.Rmin) < 0.02) | ((self.Rmax - R) < 0.02) | \
            ((Z - self.Zmin) < 0.02) | ((self.Zmax - Z) < 0.02),
            torch.tensor(0.0, dtype=torch.float64, device=self.device),
            ((pZ * (Z - self.Zmin) ** (pZ - 1) * (self.Zmax - Z) ** qZ) - \
            (qZ * (Z - self.Zmin) ** pZ * (self.Zmax - Z) ** (qZ - 1))) * \
            (R - self.Rmin) ** pR * (self.Rmax - R) ** qR,
            )
        d2b_dR2 = torch.where(
            ((R - self.Rmin) < 0.02) | ((self.Rmax - R) < 0.02) | \
            ((Z - self.Zmin) < 0.02) | ((self.Zmax - Z) < 0.02),
            torch.tensor(0.0, dtype=torch.float64, device=self.device),
            ((pR * (pR - 1) * (R - self.Rmin) ** (pR - 2) * (self.Rmax - R) ** qR) + \
            (qR * (qR - 1) * (R - self.Rmin) ** pR * (self.Rmax - R) ** (qR - 2)) - \
            (2 * pR * qR * (R - self.Rmin) ** (pR - 1) * (self.Rmax - R) ** (qR - 1))) * \
            (Z - self.Zmin) ** pZ * (self.Zmax - Z) ** qZ,
            )
        d2b_dZ2 = torch.where(
            ((R - self.Rmin) < 0.02) | ((self.Rmax - R) < 0.02) | \
            ((Z - self.Zmin) < 0.02) | ((self.Zmax - Z) < 0.02),
            torch.tensor(0.0, dtype=torch.float64, device=self.device),
            ((pZ * (pZ - 1) * (Z - self.Zmin) ** (pZ - 2) * (self.Zmax - Z) ** qZ) + \
            (qZ * (qZ - 1) * (Z - self.Zmin) ** pZ * (self.Zmax - Z) ** (qZ - 2)) - \
            (2 * pZ * qZ * (Z - self.Zmin) ** (pZ - 1) * (self.Zmax - Z) ** (qZ - 1))) * \
            (R - self.Rmin) ** pR * (self.Rmax - R) ** qR,
            )
        b_max = pR ** pR * qR ** qR * pZ ** pZ * qZ ** qZ
        return b / b_max, db_dR / b_max, db_dZ / b_max, d2b_dR2 / b_max, d2b_dZ2 / b_max

    def loss(
        self, rhs: Tensor, soln: Tensor
        ) -> Tuple[float]:

        R = self.resi[:, 0].requires_grad_(True)
        Z = self.resi[:, 1].requires_grad_(True)
        
        scaling_max = rhs.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        scaling_min = rhs.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        rhs = rhs / (scaling_max - scaling_min)
        soln = soln / (scaling_max - scaling_min).squeeze(2).squeeze(2)
        n_bdry = 200

        bdry_L_R = self.Rmin * torch.ones(n_bdry, 1, dtype=torch.float64, device=self.device)
        bdry_L_Z = torch.linspace(
            self.Zmin, self.Zmax, n_bdry, dtype=torch.float64, device=self.device
            )[:, None]
        bdry_D_R = torch.linspace(
            self.Rmin, self.Rmax, n_bdry, dtype=torch.float64, device=self.device
            )[:, None]
        bdry_D_Z = self.Zmin * torch.ones(n_bdry, 1, dtype=torch.float64, device=self.device)
        bdry_R_R = self.Rmax * torch.ones(n_bdry, 1, dtype=torch.float64, device=self.device)
        bdry_R_Z = torch.linspace(
            self.Zmin, self.Zmax, n_bdry, dtype=torch.float64, device=self.device
            )[:, None]
        bdry_U_R = torch.linspace(
            self.Rmin, self.Rmax, n_bdry, dtype=torch.float64, device=self.device
            )[:, None]
        bdry_U_Z = self.Zmax * torch.ones(n_bdry, 1, dtype=torch.float64, device=self.device)

        y = self.model.forward(
            R.unsqueeze(0).unsqueeze(2), Z.unsqueeze(0).unsqueeze(2),
            rhs.reshape(rhs.shape[0], -1)
            )

        grad_point_single_R = jacfwd(self._compute_single_batch_trunk, argnums=0)
        grad_point_single_Z = jacfwd(self._compute_single_batch_trunk, argnums=1)
        grad_point_single_RR = jacfwd(grad_point_single_R, argnums=0)
        grad_point_single_ZZ = jacfwd(grad_point_single_Z, argnums=1)
        grad_point_single_h = jacrev(self._compute_single_batch_mlp, argnums=0)
        hessian_point_single_h = hessian(self._compute_single_batch_mlp, argnums=0)

        output_branch = self.model.branch_source(rhs.reshape(rhs.shape[0], -1)).reshape(rhs.shape[0], -1)
        # output_branch = self.model.branch_mlp(output_branch)[:, None, :] \
        #     .expand(-1, len(R), -1)
        output_branch = output_branch[:, None, :].expand(-1, len(R), -1)
        output_trunk = self.model.trunk.forward(torch.concat(
            [R[None, :, None], Z[None, :, None]], dim=2
            )).repeat(rhs.shape[0], 1, 1)
        dy_dh = vmap(vmap(grad_point_single_h))(output_trunk, output_branch)
        d2y_dh2 = vmap(vmap(hessian_point_single_h))(output_trunk, output_branch)
        dh_dr = vmap(grad_point_single_R)(R, Z)[None, ...]
        dh_dz = vmap(grad_point_single_Z)(R, Z)[None, ...]
        dh2_dr2 = vmap(grad_point_single_RR)(R, Z)[None, ...]
        dh2_dz2 = vmap(grad_point_single_ZZ)(R, Z)[None, ...]
        dy_dr = (dy_dh * dh_dr).sum(dim=2)
        dy_dz = (dy_dh * dh_dz).sum(dim=2)
        d2y_dr2 = torch.einsum('bxij, bxi, bxj -> bx', d2y_dh2, dh_dr, dh_dr) + \
            (dy_dh * dh2_dr2).sum(dim=2)
        d2y_dz2 = torch.einsum('bxij, bxi, bxj -> bx', d2y_dh2, dh_dz, dh_dz) + \
            (dy_dh * dh2_dz2).sum(dim=2)

        if self.hard == True:
            b, db_dR, db_dZ, d2b_dR2, d2b_dZ2 = self.zero_bdry(R, Z, rhs)
            dpsi_dr = dy_dr * b + y * db_dR
            d2psi_dr2 = d2y_dr2 * b + 2 * dy_dr * db_dR + y * d2b_dR2
            d2psi_dz2 = d2y_dz2 * b + 2 * dy_dz * db_dZ + y * d2b_dZ2
            psi = y * b
        else:
            dpsi_dr = dy_dr
            d2psi_dr2 = d2y_dr2
            d2psi_dz2 = d2y_dz2
            psi = y
        psi_resi_loss = torch.sum(torch.mean((
            d2psi_dr2 - dpsi_dr / R[None, :] + d2psi_dz2 - rhs.reshape(rhs.shape[0], -1)
            ) ** 2, dim=1))
        psi_true_loss = torch.sum(torch.mean((soln - psi) ** 2, dim=1)) 

        if self.hard == True:
            psi_bdry_loss = torch.tensor(0.0)
        else:
            psi_bdry_L = self.model.forward(
            bdry_L_R.unsqueeze(0), bdry_L_Z.unsqueeze(0), rhs
            )
            psi_bdry_D = self.model.forward(
                bdry_D_R.unsqueeze(0), bdry_D_Z.unsqueeze(0), rhs
            )
            psi_bdry_R = self.model.forward(
                bdry_R_R.unsqueeze(0), bdry_R_Z.unsqueeze(0), rhs
            )
            psi_bdry_U = self.model.forward(
                bdry_U_R.unsqueeze(0), bdry_U_Z.unsqueeze(0), rhs
            )
            psi_bdry = torch.cat([psi_bdry_L, psi_bdry_D, psi_bdry_R, psi_bdry_U], dim=1)
            psi_bdry_loss = torch.sum(torch.mean(psi_bdry ** 2, dim=1))

        return psi_resi_loss, psi_bdry_loss, psi_true_loss
    
    def weight_update(
        self, losses: Tuple[float], weight: Tuple[float], alpha: float = 0.0
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
        return alpha * weight + (1 - alpha) * global_norm.sum() / global_norm

    def latex_sci_formatter(self, x: float, pos) -> str:
        if x == 0:
            return r"$0.0$"

        exp = int(math.floor(math.log10(abs(x))))
        if exp >= -2:
            s = f"{x:.3f}".rstrip("0").rstrip(".")
            return rf"${s}$"
        mant = x / (10 ** exp)
        return rf"${mant:.2f} \times 10^{{{exp}}}$"

    def image(
        self, epoch: int, psi_predict: Tensor, psi_true: Tensor, save_path: str
        ) -> None:

        R = self.resi[:, 0].clone().detach().cpu().numpy()
        Z = self.resi[:, 1].clone().detach().cpu().numpy()
        psi_predict = psi_predict.detach().cpu().numpy().reshape(self.nR, self.nZ)
        psi_true = psi_true.detach().cpu().numpy().reshape(self.nR, self.nZ)
        extent = [R.min(), R.max(), Z.min(), Z.max()]

        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        cmax, cmin = psi_true.max(), psi_true.min()
        cdelta = 0.95 * cmax / 7 
        levels = np.arange(0.0, 0.95 * cmax, cdelta)
        cnorm = colors.Normalize(vmin=cmin, vmax=cmax)
        c0 = ax[0].imshow(
            psi_predict.T, extent=extent, norm=cnorm, aspect='equal', origin='lower'
            )
        cs0 = ax[0].contour(
            R.reshape(65, 65), Z.reshape(65, 65), psi_predict,
            levels=levels, colors='white', linestyles='--'
        )
        ax[0].clabel(
            cs0, fmt=lambda v: self.latex_sci_formatter(v, None), 
            inline=True, fontsize=13
            )
        cbar0 = fig.colorbar(c0, ax=ax[0])
        cbar0.formatter = FuncFormatter(self.latex_sci_formatter)
        c1 = ax[1].imshow(
            psi_true.T, extent=extent, aspect='equal', origin='lower'
            )
        cs1 = ax[1].contour(
            R.reshape(65, 65), Z.reshape(65, 65), psi_true,
            levels=levels, colors='white', linestyles='--'
        )
        ax[1].clabel(
            cs1, fmt=lambda v: self.latex_sci_formatter(v, None), 
            inline=True, fontsize=13
            )
        cbar1 = fig.colorbar(c1, ax=ax[1])
        cbar1.formatter = FuncFormatter(self.latex_sci_formatter)
        c2 = ax[2].imshow(
            abs(psi_predict - psi_true).T, extent=extent,
            aspect='equal', cmap='hot', origin='lower'
            )
        cbar2 = fig.colorbar(c2, ax=ax[2])
        cbar2.formatter = FuncFormatter(self.latex_sci_formatter)
        fig.savefig(os.path.join(save_path, 'figures', f'deeponet_{epoch+1}.png'))
        plt.close()

    def train(self, save_name: str, load_name: Optional[Dict[int, str]]) -> None:    
        # (1) Load dataset and set dataloader
        self.model = DeepONet_resi(
            self.Rmin, self.Rmax, self.Zmin, self.Zmax, 
            self.nR, self.nZ
            )
        self.model = self.model.to(self.device).to(torch.float64)
        save_path = os.path.join('logs', save_name)
        self.resi = self.dataset.residual.to(self.device)
        self.bdry = self.dataset.boundary.to(self.device)
        data_size = len(self.dataset)
        train_size = int(0.9 * data_size)
        test_size = data_size - train_size
        if load_name:
            load_path = os.path.join('logs', load_name['name'])
            load_epoch = load_name['epoch']
            load_name = load_name['name']
            self.model.load_state_dict(torch.load(os.path.join(
                load_path, 'model', f'model_deeponet_{load_epoch}.pt'
            ), map_location=self.device, weights_only=True))
            dataset_idx = torch.load(os.path.join(
                load_path, 'split_idx.pt'
            ), weights_only=True)
            train_dataset = Subset(self.dataset, dataset_idx['train'])
            test_dataset = Subset(self.dataset, dataset_idx['test'])
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
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=self.batch_size)
        mu0 = 4e-7 * torch.pi

        # (2) Iteratively train to find the parameter value
        optim = torch.optim.LBFGS(self.model.parameters(), line_search_fn='strong_wolfe')
        print("Length for the dataloader :", len(train_dataloader))
        for epoch in range(load_epoch, self.epoch):
            loss = psi_resi_loss = psi_bdry_loss = psi_true_loss = 0.0
            loss_list = []
            for i, batch in enumerate(train_dataloader):
                _stats = []
                def closure():
                    rhs, bdry, soln, psi_coil_isoflux, \
                        psi_coil_Br, psi_coil_Bz, coil, U = batch
                    rhs = rhs.to(self.device)[:, None, :, :]
                    bdry = torch.zeros_like(bdry).to(self.device)
                    soln = U.to(self.device).reshape(U.shape[0], -1) * -mu0
                    psi_resi_loss_batch, psi_bdry_loss_batch, psi_true_loss_batch = \
                        self.loss(rhs, soln)
                    loss_batch = psi_resi_loss_batch + 10 * psi_bdry_loss_batch \
                        # + 5 * psi_true_loss_batch
                    optim.zero_grad()
                    loss_batch.backward()
                    _stats.append([
                        loss_batch.item(), psi_resi_loss_batch.item(),
                        psi_bdry_loss_batch.item(), psi_true_loss_batch.item(),
                        len(rhs)
                        ])
                    return loss_batch
                optim.step(closure)
                loss_list.append(_stats[-1])
                if epoch < 5 and (i + 1) % 50 == 0:
                    rhs, bdry, soln, _, _, _, coil, U = batch
                    rhs = rhs.to(self.device)[:, None, :, :]
                    bdry = torch.zeros_like(bdry).to(self.device)
                    soln = U.to(self.device).reshape(U.shape[0], -1) * -mu0
                    sample_rhs = rhs[0:1]
                    sample_rhs_norm = sample_rhs / (sample_rhs.max() - sample_rhs.min())
                    sample_forward = self.model.forward(
                        self.resi[None, :, 0:1], self.resi[None, :, 1:2],
                        sample_rhs_norm.reshape(sample_rhs_norm.shape[0], -1)
                        ).squeeze(0)
                    if self.hard == True:
                        b, _, _, _, _ = self.zero_bdry(
                            self.resi[:, 0], self.resi[:, 1], sample_rhs_norm
                            )
                        sample_forward = sample_forward * b
                    predict = sample_forward * (sample_rhs.max() - sample_rhs.min())
                    self.image(
                        i, predict, soln[0], save_path=save_path
                        )
                    print(
                        f"Batch {i:03d} | Loss : {loss_list[-1][0]:.4f} = " \
                        f"{loss_list[-1][1]:.4f} + {loss_list[-1][2]:.4f} " \
                        f"+ {loss_list[-1][3]:.4f}"
                        )
            loss_list = np.array(loss_list)
            loss = loss_list[:, 0].sum() / loss_list[:, 4].sum()
            psi_resi_loss = loss_list[:, 1].sum() / loss_list[:, 4].sum()
            psi_bdry_loss = loss_list[:, 2].sum() / loss_list[:, 4].sum()
            psi_true_loss = loss_list[:, 3].sum() / loss_list[:, 4].sum()
            
            if (epoch + 1) % 1 == 0:
                batch = next(iter(test_dataloader))
                rhs, bdry, soln, _, _, _, coil, U = batch
                rhs = rhs.to(self.device)[:, None, :, :]
                bdry = torch.zeros_like(bdry).to(self.device)
                soln = U.to(self.device).reshape(U.shape[0], -1) * -mu0
                sample_rhs = rhs[0:1]
                sample_rhs_norm = sample_rhs / (sample_rhs.max() - sample_rhs.min())
                sample_forward = self.model.forward(
                    self.resi[None, :, 0:1], self.resi[None, :, 1:2],
                    sample_rhs_norm.reshape(1, -1)
                    ).squeeze(0)
                if self.hard == True:
                    b, _, _, _, _ = self.zero_bdry(
                        self.resi[:, 0], self.resi[:, 1], sample_rhs_norm
                        )
                    sample_forward = sample_forward * b
                predict = sample_forward * (sample_rhs.max() - sample_rhs.min())
                self.image(
                    epoch, predict, soln[0], save_path=save_path
                    )
                torch.save(self.model.state_dict(), os.path.join(
                    save_path, 'model', f'model_deeponet_{epoch}.pt'
                    ))
            psi_test_true_loss = 0.0
            psi_test_rel_loss = 0.0

            for batch in test_dataloader:
                rhs, bdry, soln, psi_coil_isoflux, \
                    psi_coil_Br, psi_coil_Bz, coil, U = batch
                rhs = rhs.to(self.device)[:, None, :, :]
                bdry = torch.zeros_like(bdry).to(self.device)
                soln = U.to(self.device).reshape(U.shape[0], -1) * -mu0
                R = self.resi[:, 0].requires_grad_(True)
                Z = self.resi[:, 1].requires_grad_(True)
                
                scaling_max = rhs.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
                scaling_min = rhs.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
                rhs = rhs / (scaling_max - scaling_min)
                predict = self.model.forward(
                    R.unsqueeze(0).unsqueeze(2), Z.unsqueeze(0).unsqueeze(2), rhs.reshape(rhs.shape[0], -1)
                    ).squeeze(0)
                if self.hard == True:
                    b, _, _, _, _ = self.zero_bdry(R, Z, rhs)
                    predict = predict * b
                predict = predict * (scaling_max - scaling_min).squeeze(3).squeeze(2)
                psi_test_true_loss += torch.sum(torch.sqrt(
                    torch.mean((soln - predict) ** 2, dim=1)
                    )).item()
                psi_test_rel_loss += torch.sum(torch.sqrt(
                    torch.mean((soln - predict) ** 2, dim=1)
                    ) / torch.sqrt(
                    torch.mean(soln ** 2 + 1e-12, dim=1)
                    )).item()
            psi_test_true_loss /= test_size
            psi_test_rel_loss /= test_size
            print(
                f"Epoch {epoch:04d} | Training Loss : {loss:.4f} = " \
                f"{psi_resi_loss:.4f} + {psi_bdry_loss:.4f} " \
                f"+ {psi_true_loss:.4f} | " \
                f"Test Loss : Abs {psi_test_true_loss:.4f} Rel {psi_test_rel_loss:.4f}"
                )
        print("Training done. Save the model weight")
        torch.save(self.model.state_dict(), os.path.join(save_path, 'model', 'model_deeponet.pt'))
    
class Trainer_bdry():
    def __init__(
        self, Rmin: float, Rmax: float, Zmin: float, Zmax: float,
        nR: int, nZ: int, len_data: int, batch_size: int, epoch: int,
        num_resi_pt: int,
        data_load_path: Optional[str]=None, data_save_path: Optional[str]=None,
        device: Optional[str] = 'cuda:0'
        ) -> None:
        self.device = device
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.nR = nR
        self.nZ = nZ
        self.batch_size = batch_size
        self.epoch = epoch
        self.num_resi_pt = num_resi_pt
        self.dataset = GSrhsdataset(
            self.Rmin, self.Rmax, self.Zmin, self.Zmax, self.nR, self.nZ, 
            len_data, load_path=data_load_path, save_path=data_save_path
        )
        plt.rcParams['text.usetex'] = True

    def latex_sci_formatter(self, x: float, pos) -> str:
        if x == 0:
            return r"$0.0$"

        exp = int(math.floor(math.log10(abs(x))))
        if exp >= -2:
            s = f"{x:.3f}".rstrip("0").rstrip(".")
            return rf"${s}$"
        mant = x / (10 ** exp)
        return rf"${mant:.2f} \times 10^{{{exp}}}$"
    
    def _compute_single_batch_pinto(
        self, R: Tensor, Z: Tensor, bdry_pt: Tensor, bdry: Tensor
        ) -> Tensor:
        psi = self.model.forward(
            R.unsqueeze(0).unsqueeze(1).unsqueeze(2),
            Z.unsqueeze(0).unsqueeze(1).unsqueeze(2),
            bdry_pt.unsqueeze(0), bdry.unsqueeze(0)
            )
        return psi.squeeze(0).squeeze(0).squeeze(0)

    def loss(self, bdry: Tensor, soln: Tensor) -> Tuple[float]:
        R = self.resi_pt[:, :, 0].clone().detach().requires_grad_(True).squeeze(0)
        Z = self.resi_pt[:, :, 1].clone().detach().requires_grad_(True).squeeze(0)
        scaling_max = bdry.max(dim=1, keepdim=True)[0]
        scaling_min = bdry.min(dim=1, keepdim=True)[0]
        bdry = bdry / (scaling_max - scaling_min)
        soln = soln / (scaling_max - scaling_min).squeeze(1)
        psi = self.model.forward(
            self.resi[None, :, 0:1], self.resi[None, :, 1:2], self.bdry_pt, bdry
            ).squeeze(2)

        grad_point_single_R = jacrev(self._compute_single_batch_pinto, argnums=0)
        grad_point_single_Z = jacrev(self._compute_single_batch_pinto, argnums=1)
        grad_point_single_RR = jacrev(grad_point_single_R, argnums=0)
        grad_point_single_ZZ = jacrev(grad_point_single_Z, argnums=1)
        dpsi_dr = vmap(
            vmap(grad_point_single_R, in_dims=(0, 0, None, None)),
            in_dims=(None, None, None, 0)
            )(R, Z, self.bdry_pt.squeeze(0), bdry)
        d2psi_dr2 = vmap(
            vmap(grad_point_single_RR, in_dims=(0, 0, None, None)),
            in_dims=(None, None, None, 0)
            )(R, Z, self.bdry_pt.squeeze(0), bdry)
        d2psi_dz2 = vmap(
            vmap(grad_point_single_ZZ, in_dims=(0, 0, None, None)),
            in_dims=(None, None, None, 0)
            )(R, Z, self.bdry_pt.squeeze(0), bdry)

        psi_resi_loss = torch.sum(torch.mean((
            d2psi_dr2 - dpsi_dr / R[None, :] + d2psi_dz2
            ) ** 2, dim=1))
        psi_true_loss = torch.sum(torch.mean((soln - psi) ** 2, dim=1)) 
        psi_bdry = self.model.forward(
            self.bdry_pt[:, :, 0:1], self.bdry_pt[:, :, 1:2], self.bdry_pt,
            bdry 
        )
        psi_bdry_loss = torch.sum(torch.mean((psi_bdry - bdry) ** 2, dim=(1, 2)))
        return psi_resi_loss, psi_bdry_loss, psi_true_loss

    def image(
        self, epoch: int, bdry: Tensor, 
        psi_predict: Tensor, psi_true: Tensor, save_path: str
        ):

        R = self.resi[:, 0].clone().detach().cpu().numpy()
        Z = self.resi[:, 1].clone().detach().cpu().numpy()
        bdry = bdry.detach().cpu().numpy()
        psi_predict = psi_predict.detach().cpu().numpy().reshape(self.nR, self.nZ)
        psi_true = psi_true.detach().cpu().numpy().reshape(self.nR, self.nZ)
        extent = [R.min(), R.max(), Z.min(), Z.max()]

        fig, ax = plt.subplots(2, 3, figsize=(21, 14))
        gs = ax[0, 0].get_gridspec()
        for axis in ax[1, :]:
            axis.remove()
        ax_bottom = fig.add_subplot(gs[1, :])
        cmax, cmin = psi_true.max(), psi_true.min()
        cdelta = 0.95 * cmax / 7 
        levels = np.arange(0.0, 0.95 * cmax, cdelta)
        cnorm = colors.Normalize(vmin=cmin, vmax=cmax)
        c0 = ax[0, 0].imshow(
            psi_predict.T, extent=extent, norm=cnorm, aspect='equal', origin='lower'
            )
        cs0 = ax[0, 0].contour(
            R.reshape(65, 65), Z.reshape(65, 65), psi_predict,
            levels=levels, colors='white', linestyles='--'
        )
        ax[0, 0].clabel(
            cs0, fmt=lambda v: self.latex_sci_formatter(v, None), 
            inline=True, fontsize=13
            )
        cbar0 = fig.colorbar(c0, ax=ax[0, 0])
        cbar0.formatter = FuncFormatter(self.latex_sci_formatter)
        c1 = ax[0, 1].imshow(
            psi_true.T, extent=extent, aspect='equal', origin='lower'
            )
        cs1 = ax[0, 1].contour(
            R.reshape(65, 65), Z.reshape(65, 65), psi_true,
            levels=levels, colors='white', linestyles='--'
        )
        ax[0, 1].clabel(
            cs1, fmt=lambda v: self.latex_sci_formatter(v, None), 
            inline=True, fontsize=13
            )
        cbar1 = fig.colorbar(c1, ax=ax[0, 1])
        cbar1.formatter = FuncFormatter(self.latex_sci_formatter)
        c2 = ax[0, 2].imshow(
            abs(psi_predict - psi_true).T, extent=extent,
            aspect='equal', cmap='hot', origin='lower'
            )
        cbar2 = fig.colorbar(c2, ax=ax[0, 2])
        cbar2.formatter = FuncFormatter(self.latex_sci_formatter)
        psi_predict_bdry = np.concatenate([
            psi_predict[0, :], psi_predict[:, 0],
            psi_predict[-1, :], psi_predict[:, -1]
        ])
        ax_bottom.plot(psi_predict_bdry, color='C0', label='Predicted')
        ax_bottom.plot(bdry, color='C1', label='True')
        ax_bottom.legend(fontsize=13)
        
        fig.savefig(os.path.join(save_path, 'figures', f'pinto_{epoch+1}.png'))
        plt.close()

    def train(self, save_name: str, load_name: Optional[dict] = None) -> None:    
        # (1) Load dataset and set dataloader
        self.model = PINTO(
            self.Rmin, self.Rmax, self.Zmin, self.Zmax, 
            self.nR, self.nZ
            )
        self.model = self.model.to(self.device).to(torch.float64)
        save_path = os.path.join('logs', save_name)
        self.resi = self.dataset.residual.to(self.device)
        self.resi_pt = torch.rand(
            (1, self.num_resi_pt, 2), dtype=torch.float64, device=self.device
        )
        self.resi_pt[:, :, 0] = self.resi_pt[:, :, 0] * (self.Rmax - self.Rmin) + self.Rmin
        self.resi_pt[:, :, 1] = self.resi_pt[:, :, 1] * (self.Zmax - self.Zmin) + self.Zmin
        self.bdry_pt = self.dataset.boundary.to(self.device).reshape(1, -1, 2)
        if load_name:
            load_path = os.path.join('logs', load_name['name'])
            load_epoch = load_name['epoch']
            load_name = load_name['name']
            self.model.load_state_dict(torch.load(os.path.join(
                load_path, 'model', f'model_pinto_{load_epoch}.pt'
            ), map_location=self.device, weights_only=True))
            dataset_idx = torch.load(os.path.join(
                load_path, 'split_idx.pt'
            ), weights_only=True)
            
            train_dataset = Subset(self.dataset, dataset_idx['train'])
            test_dataset = Subset(self.dataset, dataset_idx['test'])
            train_size = len(train_dataset)
            test_size = len(test_dataset)
            _preprocess_train = []
            for idx, data in enumerate(train_dataset):
                bdry = data[1]
                if (bdry.max() - bdry.min()) < 100.0:
                    _preprocess_train.append(idx)
            self.dataset = Subset(self.dataset, _preprocess_train)
            _preprocess_test = []
            for idx, data in enumerate(test_dataset):
                bdry = data[1]
                if (bdry.max() - bdry.min()) < 100.0:
                    _preprocess_test.append(idx)
            test_dataset = Subset(test_dataset, _preprocess_test)
        else:
            _preprocess = []
            for idx, data in enumerate(self.dataset):
                bdry = data[1]
                if (bdry.max() - bdry.min()) < 100.0:
                    _preprocess.append(idx)
            self.dataset = Subset(self.dataset, _preprocess)
            data_size = len(self.dataset)
            train_size = int(0.9 * data_size)
            test_size = data_size - train_size
            load_epoch = 0 
            train_dataset, test_dataset = random_split(
                self.dataset, [train_size, test_size]
                )
        splits = {
            'train': train_dataset.indices,
            'test' : test_dataset.indices
        }
        torch.save(splits, os.path.join(save_path, 'split_idx.pt'))
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        mu0 = 4e-7 * torch.pi
        optim = torch.optim.LBFGS(self.model.parameters(), line_search_fn='strong_wolfe')

        print("Length for the dataloader :", len(train_dataloader))
        for epoch in range(load_epoch, self.epoch):
            loss = psi_resi_loss = psi_bdry_loss = psi_true_loss = 0.0
            loss_list = []
            for i, batch in enumerate(train_dataloader):
                _stats = []
                def closure():
                    _, bdry, psi, _, _, _, _, U = batch
                    bdry = bdry.reshape(bdry.shape[0], -1, 1).to(self.device)
                    soln = (psi + U * mu0).to(self.device).reshape(U.shape[0], -1)
                    psi_resi_loss_batch, psi_bdry_loss_batch, psi_true_loss_batch = \
                        self.loss(bdry, soln)
                    loss_batch = psi_resi_loss_batch + 30 * psi_bdry_loss_batch \
                        + 5 * psi_true_loss_batch
                    optim.zero_grad()
                    loss_batch.backward()
                    _stats.append([
                        loss_batch.item(), psi_resi_loss_batch.item(),
                        psi_bdry_loss_batch.item(), psi_true_loss_batch.item(),
                        len(bdry)
                        ])
                    return loss_batch
                optim.step(closure)
                loss_list.append(_stats[-1])
                if epoch < 5 and (i + 1) % 20 == 0:
                    _, bdry, psi, _, _, _, coil, U = batch
                    bdry = bdry.reshape(bdry.shape[0], -1, 1).to(self.device)
                    soln = (psi + U * mu0).to(self.device).reshape(U.shape[0], -1)
                    sample_bdry = bdry[0:1]
                    norm_bdry = sample_bdry / (sample_bdry.max() - sample_bdry.min())
                    sample_forward = self.model.forward(
                        self.resi[None, :, 0:1], self.resi[None, :, 1:2],
                        self.bdry_pt, norm_bdry
                        ).squeeze(0)
                    predict = sample_forward * (sample_bdry.max() - sample_bdry.min())
                    self.image(
                        i, bdry[0], predict, soln[0], save_path=save_path
                        )
                    print(
                        f"Batch {i:03d} | Loss : {loss_list[-1][0]:.4f} = " \
                        f"{loss_list[-1][1]:.4f} + {loss_list[-1][2]:.4f} " \
                        f"+ {loss_list[-1][3]:.4f}"
                        )
            loss_list = np.array(loss_list)
            loss = loss_list[:, 0].sum() / loss_list[:, 4].sum()
            psi_resi_loss = loss_list[:, 1].sum() / loss_list[:, 4].sum()
            psi_bdry_loss = loss_list[:, 2].sum() / loss_list[:, 4].sum()
            psi_true_loss = loss_list[:, 3].sum() / loss_list[:, 4].sum()
            
            if (epoch + 1) % 1 == 0:
                batch = next(iter(test_dataloader))
                _, bdry, psi, _, _, _, _, U = batch
                bdry = bdry.reshape(bdry.shape[0], -1, 1).to(self.device)
                soln = (psi + U * mu0).to(self.device).reshape(U.shape[0], -1)
                sample_bdry = bdry[0:1]
                norm_bdry = sample_bdry / (sample_bdry.max() - sample_bdry.min())
                sample_forward = self.model.forward(
                    self.resi[None, :, 0:1], self.resi[None, :, 1:2],
                    self.bdry_pt, norm_bdry
                    ).squeeze(0)
                predict = sample_forward * (sample_bdry.max() - sample_bdry.min())
                self.image(
                    epoch, bdry[0], predict,
                    soln[0], save_path=save_path
                    )
                torch.save(self.model.state_dict(), os.path.join(
                    save_path, 'model', f'model_pinto_{epoch}.pt'
                    ))
            psi_test_true_loss = 0.0
            psi_test_rel_loss = 0.0

            for batch in test_dataloader:
                _, bdry, psi, _, _, _, _, U = batch
                bdry = bdry.reshape(bdry.shape[0], -1, 1).to(self.device)
                soln = (psi + U * mu0).to(self.device).reshape(U.shape[0], -1)
                scaling_max = bdry.max(dim=1, keepdim=True)[0]
                scaling_min = bdry.min(dim=1, keepdim=True)[0]
                norm_bdry = bdry / (scaling_max - scaling_min)
                with torch.no_grad():
                    sample_forward = self.model.forward(
                        self.resi[None, :, 0:1], self.resi[None, :, 1:2],
                        self.bdry_pt, norm_bdry
                        ).squeeze(2)
                predict = sample_forward * (scaling_max - scaling_min).squeeze(2)
                psi_test_true_loss += torch.sum(torch.sqrt(
                    torch.sum((soln - predict) ** 2, dim=1)
                    )).item()
                psi_test_rel_loss += torch.sum(torch.sqrt(
                    torch.sum((soln - predict) ** 2, dim=1)
                    ) / torch.sqrt(
                    torch.sum(soln ** 2 + 1e-12, dim=1)
                    )).item()
            psi_test_true_loss /= test_size
            psi_test_rel_loss /= test_size
            print(
                f"Epoch {epoch:04d} | Training Loss : {loss:.4f} = " \
                f"{psi_resi_loss:.4f} + {psi_bdry_loss:.4f} " \
                f"+ {psi_true_loss:.4f} | " \
                f"Test Loss : Abs {psi_test_true_loss:.4f} Rel {psi_test_rel_loss:.4f}"
                )

        print("Training done. Save the model weight")
        torch.save(self.model.state_dict(), os.path.join(save_path, 'model', 'model_pinto.pt'))

class Trainer_bdry_deeponet():
    def __init__(
        self, Rmin: float, Rmax: float, Zmin: float, Zmax: float,
        nR: int, nZ: int, len_data: int, batch_size: int,
        epoch: int, 
        data_load_path: Optional[str]=None, data_save_path: Optional[str]=None,
        device: Optional[str] = 'cuda:0'
        ) -> None:
        self.device = device
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.nR = nR
        self.nZ = nZ
        self.batch_size = batch_size
        self.epoch = epoch
        self.dataset = GSrhsdataset(
            self.Rmin, self.Rmax, self.Zmin, self.Zmax, self.nR, self.nZ, 
            len_data, load_path=data_load_path, save_path=data_save_path
        )
        self.resi = self.dataset.residual.to(self.device)
        self.bdry = self.dataset.boundary.to(self.device)
        self.bdry_pt = self.dataset.boundary.to(self.device).reshape(1, -1, 2)
        plt.rcParams['text.usetex'] = True

    def _compute_single_batch(self, R: Tensor, Z: Tensor, bdry: Tensor) -> Tensor:
        psi = self.model.forward(R.unsqueeze(0), Z.unsqueeze(0), bdry)
        return psi.squeeze(0)
        
    def loss(
        self, bdry: Tensor, soln: Tensor, model: Optional[torch.nn.Module]
        ) -> Tuple[float]:
        if model is not None:
            self.model = model

        R = self.resi[:, 0].requires_grad_(True)
        Z = self.resi[:, 1].requires_grad_(True)
        R_resi = torch.rand(
            (500,), dtype=torch.float64, device=self.device
        ).requires_grad_(True)
        Z_resi = torch.rand(
            (500,), dtype=torch.float64, device=self.device
        ).requires_grad_(True)
        R = R * (self.Rmax - self.Rmin) / 2 + (3 * self.Rmin + self.Rmax) / 4
        Z = Z * (self.Zmax - self.Zmin) / 2 + (self.Zmax + 3 * self.Zmin) / 4
        
        scaling_max = bdry.max(dim=1, keepdim=True)[0]
        scaling_min = bdry.min(dim=1, keepdim=True)[0]
        bdry = bdry / (scaling_max - scaling_min)
        soln = soln / (scaling_max - scaling_min)

        psi = vmap(vmap(
            self.model.forward, in_dims=(0, 0, None)
            ), in_dims=(None, None, 0))(
            R.unsqueeze(1), Z.unsqueeze(1), bdry.reshape(bdry.shape[0], -1)
            ).squeeze(2)
        grad_point_single_R = jacrev(self._compute_single_batch, argnums=0)
        grad_point_single_Z = jacrev(self._compute_single_batch, argnums=1)
        grad_point_single_RR = jacrev(grad_point_single_R, argnums=0)
        grad_point_single_ZZ = jacrev(grad_point_single_Z, argnums=1)
        dpsi_dr = vmap(
            vmap(grad_point_single_R, in_dims=(0, 0, None)),
            in_dims=(None, None, 0)
            )(R_resi, Z_resi, bdry)
        d2psi_dr2 = vmap(
            vmap(grad_point_single_RR, in_dims=(0, 0, None)),
            in_dims=(None, None, 0)
            )(R_resi, Z_resi, bdry)
        d2psi_dz2 = vmap(
            vmap(grad_point_single_ZZ, in_dims=(0, 0, None)),
            in_dims=(None, None, 0)
            )(R_resi, Z_resi, bdry)

        psi_resi_loss = torch.sum(torch.mean((
            d2psi_dr2 - dpsi_dr / R_resi[None, :] + d2psi_dz2
            ) ** 2, dim=1))
        psi_true_loss = torch.sum(torch.mean((soln - psi) ** 2, dim=1)) 

        psi_bdry = vmap(vmap(
            self.model.forward, in_dims=(0, 0, None)
            ), in_dims=(None, None, 0))(
            self.bdry_pt[0, :, 0:1], self.bdry_pt[0, :, 1:2], bdry
            ).squeeze(2)
        psi_bdry_loss = torch.sum(torch.mean((psi_bdry - bdry) ** 2, dim=1))

        return psi_resi_loss, psi_bdry_loss, psi_true_loss
    
    def latex_sci_formatter(self, x: float, pos) -> str:
        if x == 0:
            return r"$0.0$"

        exp = int(math.floor(math.log10(abs(x))))
        if exp >= -2:
            s = f"{x:.3f}".rstrip("0").rstrip(".")
            return rf"${s}$"
        mant = x / (10 ** exp)
        return rf"${mant:.2f} \times 10^{{{exp}}}$"

    def image(
        self, epoch: int, bdry: int, 
        psi_predict: Tensor, psi_true: Tensor, save_path: str
        ) -> None:

        R = self.resi[:, 0].clone().detach().cpu().numpy()
        Z = self.resi[:, 1].clone().detach().cpu().numpy()
        bdry = bdry.detach().cpu().numpy()
        psi_predict = psi_predict.detach().cpu().numpy().reshape(self.nR, self.nZ)
        psi_true = psi_true.detach().cpu().numpy().reshape(self.nR, self.nZ)
        extent = [R.min(), R.max(), Z.min(), Z.max()]

        fig, ax = plt.subplots(2, 3, figsize=(21, 14))
        gs = ax[0, 0].get_gridspec()
        for axis in ax[1, :]:
            axis.remove()
        ax_bottom = fig.add_subplot(gs[1, :])
        cmax, cmin = psi_true.max(), psi_true.min()
        cdelta = 0.95 * cmax / 7 
        levels = np.arange(0.0, 0.95 * cmax, cdelta)
        cnorm = colors.Normalize(vmin=cmin, vmax=cmax)
        c0 = ax[0, 0].imshow(
            psi_predict.T, extent=extent, norm=cnorm, aspect='equal', origin='lower'
            )
        cs0 = ax[0, 0].contour(
            R.reshape(65, 65), Z.reshape(65, 65), psi_predict,
            levels=levels, colors='white', linestyles='--'
        )
        ax[0, 0].clabel(
            cs0, fmt=lambda v: self.latex_sci_formatter(v, None), 
            inline=True, fontsize=13
            )
        cbar0 = fig.colorbar(c0, ax=ax[0, 0])
        cbar0.formatter = FuncFormatter(self.latex_sci_formatter)
        c1 = ax[0, 1].imshow(
            psi_true.T, extent=extent, aspect='equal', origin='lower'
            )
        cs1 = ax[0, 1].contour(
            R.reshape(65, 65), Z.reshape(65, 65), psi_true,
            levels=levels, colors='white', linestyles='--'
        )
        ax[0, 1].clabel(
            cs1, fmt=lambda v: self.latex_sci_formatter(v, None), 
            inline=True, fontsize=13
            )
        cbar1 = fig.colorbar(c1, ax=ax[0, 1])
        cbar1.formatter = FuncFormatter(self.latex_sci_formatter)
        c2 = ax[0, 2].imshow(
            abs(psi_predict - psi_true).T, extent=extent,
            aspect='equal', cmap='hot', origin='lower'
            )
        cbar2 = fig.colorbar(c2, ax=ax[0, 2])
        cbar2.formatter = FuncFormatter(self.latex_sci_formatter)
        psi_predict_bdry = np.concatenate([
            psi_predict[0, :], psi_predict[:, 0],
            psi_predict[-1, :], psi_predict[:, -1]
        ])
        ax_bottom.plot(psi_predict_bdry, color='C0', label='Predicted')
        ax_bottom.plot(bdry, color='C1', label='True')
        ax_bottom.legend(fontsize=13)
        
        fig.savefig(os.path.join(save_path, 'figures', f'deeponet_{epoch+1}.png'))
        plt.close()

    def train(self, save_name: str, load_name: Optional[Dict[int, str]]) -> None:    
        # (1) Load dataset and set dataloader
        self.model = DeepONet_bdry(
            self.Rmin, self.Rmax, self.Zmin, self.Zmax, 
            self.nR, self.nZ
            )
        self.model = self.model.to(self.device).to(torch.float64)
        save_path = os.path.join('logs', save_name)
        data_size = len(self.dataset)
        train_size = int(0.9 * data_size)
        test_size = data_size - train_size
        if load_name:
            load_path = os.path.join('logs', load_name['name'])
            load_epoch = load_name['epoch']
            load_name = load_name['name']
            self.model.load_state_dict(torch.load(os.path.join(
                load_path, 'model', f'model_pinto_{load_epoch}.pt'
            ), map_location=self.device, weights_only=True))
            dataset_idx = torch.load(os.path.join(
                load_path, 'split_idx.pt'
            ), weights_only=True)
            
            train_dataset = Subset(self.dataset, dataset_idx['train'])
            test_dataset = Subset(self.dataset, dataset_idx['test'])
            train_size = len(train_dataset)
            test_size = len(test_dataset)
            _preprocess_train = []
            for idx, data in enumerate(train_dataset):
                bdry = data[1]
                if (bdry.max() - bdry.min()) < 100.0:
                    _preprocess_train.append(idx)
            self.dataset = Subset(self.dataset, _preprocess_train)
            _preprocess_test = []
            for idx, data in enumerate(test_dataset):
                bdry = data[1]
                if (bdry.max() - bdry.min()) < 100.0:
                    _preprocess_test.append(idx)
            test_dataset = Subset(test_dataset, _preprocess_test)
        else:
            _preprocess = []
            for idx, data in enumerate(self.dataset):
                bdry = data[1]
                if (bdry.max() - bdry.min()) < 100.0:
                    _preprocess.append(idx)
            self.dataset = Subset(self.dataset, _preprocess)
            data_size = len(self.dataset)
            train_size = int(0.9 * data_size)
            test_size = data_size - train_size
            load_epoch = 0 
            train_dataset, test_dataset = random_split(
                self.dataset, [train_size, test_size]
                )
        splits = {
            'train': train_dataset.indices,
            'test' : test_dataset.indices
        }
        torch.save(splits, os.path.join(save_path, 'split_idx.pt'))
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=self.batch_size)
        mu0 = 4e-7 * torch.pi

        # (2) Iteratively train to find the parameter value
        optim = torch.optim.LBFGS(self.model.parameters(), line_search_fn='strong_wolfe')
        print("Length for the dataloader :", len(train_dataloader))
        for epoch in range(load_epoch, self.epoch):
            loss = psi_resi_loss = psi_bdry_loss = psi_true_loss = 0.0
            loss_list = []
            for i, batch in enumerate(train_dataloader):
                _stats = []
                def closure():
                    _, bdry, psi, _, _, _, _, U = batch
                    bdry = bdry.to(self.device)
                    soln = (psi + U * mu0).to(self.device).reshape(U.shape[0], -1)
                    psi_resi_loss_batch, psi_bdry_loss_batch, psi_true_loss_batch = \
                        self.loss(bdry, soln)
                    loss_batch = psi_resi_loss_batch + 30 * psi_bdry_loss_batch \
                        # + 5 * psi_true_loss_batch
                    optim.zero_grad()
                    loss_batch.backward()
                    _stats.append([
                        loss_batch.item(), psi_resi_loss_batch.item(),
                        psi_bdry_loss_batch.item(), psi_true_loss_batch.item(),
                        len(bdry)
                        ])
                    return loss_batch
                optim.step(closure)
                loss_list.append(_stats[-1])
                if epoch < 5 and (i + 1) % 30 == 0:
                    _, bdry, psi, _, _, _, _, U = batch
                    bdry = bdry.to(self.device)
                    soln = (psi + U * mu0).to(self.device).reshape(U.shape[0], -1)
                    sample_bdry = bdry[0].squeeze(-1)
                    sample_bdry_norm = sample_bdry / (sample_bdry.max() - sample_bdry.min())
                    sample_forward = vmap(self.model.forward, in_dims=(0, 0, None))(
                        self.resi[:, 0:1], self.resi[:, 1:2],
                        sample_bdry_norm
                        )
                    predict = sample_forward * (sample_bdry.max() - sample_bdry.min())
                    self.image(
                        i, bdry[0], predict, soln[0], save_path=save_path
                        )
                    print(
                        f"Batch {i:03d} | Loss : {loss_list[-1][0]:.4f} = " \
                        f"{loss_list[-1][1]:.4f} + {loss_list[-1][2]:.4f} " \
                        f"+ {loss_list[-1][3]:.4f}"
                        )
            loss_list = np.array(loss_list)
            loss = loss_list[:, 0].sum() / loss_list[:, 4].sum()
            psi_resi_loss = loss_list[:, 1].sum() / loss_list[:, 4].sum()
            psi_bdry_loss = loss_list[:, 2].sum() / loss_list[:, 4].sum()
            psi_true_loss = loss_list[:, 3].sum() / loss_list[:, 4].sum()
            
            if (epoch + 1) % 1 == 0:
                batch = next(iter(test_dataloader))
                _, bdry, psi, _, _, _, coil, U = batch
                bdry = bdry.to(self.device)
                soln = (psi + U * mu0).to(self.device).reshape(U.shape[0], -1)
                sample_bdry = bdry[0].squeeze(-1)
                sample_bdry_norm = sample_bdry / (sample_bdry.max() - sample_bdry.min())
                sample_forward = vmap(self.model.forward, in_dims=(0, 0, None))(
                    self.resi[:, 0:1], self.resi[:, 1:2],
                    sample_bdry_norm
                    )
                predict = sample_forward * (sample_bdry.max() - sample_bdry.min())
                self.image(
                    epoch, bdry[0], predict, soln[0], save_path=save_path
                    )
                torch.save(self.model.state_dict(), os.path.join(
                    save_path, 'model', f'model_deeponet_{epoch}.pt'
                    ))
            psi_test_true_loss = 0.0
            psi_test_rel_loss = 0.0

            for batch in test_dataloader:
                _, bdry, psi, _, _, _, _, U = batch
                bdry = torch.zeros_like(bdry).to(self.device)
                soln = (psi + U * mu0).to(self.device).reshape(U.shape[0], -1) * -mu0
                scaling_max = bdry.max(dim=1, keepdim=True)[0]
                scaling_min = bdry.min(dim=1, keepdim=True)[0]
                predict = vmap(vmap(
                    self.model.forward, in_dims=(0, 0, None)
                    ), in_dims=(None, None, 0))(
                    self.resi[:, 0:1], self.resi[:, 1:2], bdry
                    ).squeeze(2)
                predict = predict * (scaling_max - scaling_min)
                psi_test_true_loss += torch.sum(torch.sqrt(
                    torch.mean((soln - predict) ** 2, dim=1)
                    )).item()
                psi_test_rel_loss += torch.sum(torch.sqrt(
                    torch.mean((soln - predict) ** 2, dim=1)
                    ) / torch.sqrt(
                    torch.mean(soln ** 2 + 1e-12, dim=1)
                    )).item()
            psi_test_true_loss /= test_size
            psi_test_rel_loss /= test_size
            print(
                f"Epoch {epoch:04d} | Training Loss : {loss:.4f} = " \
                f"{psi_resi_loss:.4f} + {psi_bdry_loss:.4f} " \
                f"+ {psi_true_loss:.4f} | " \
                f"Test Loss : Abs {psi_test_true_loss:.4f} Rel {psi_test_rel_loss:.4f}"
                )
        print("Training done. Save the model weight")
        torch.save(self.model.state_dict(), os.path.join(save_path, 'model', 'model_deeponet.pt'))
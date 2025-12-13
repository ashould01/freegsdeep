import torch
import torch.nn as nn
from torch.func import jacrev, jacfwd, hessian, vmap
from freegsdeep.utilstyping import *
from functools import partial

class critical():
    
    def __init__(
        self, solver_resi: nn.Module, solver_bdry: nn.Module,
        bdry_pt: Tensor, nx: int, ny: int, Rmin: float, Rmax: float, 
        Zmin: float, Zmax: float, device: torch.device,
        ) -> None:
        self.solver_resi = solver_resi
        self.solver_bdry = solver_bdry
        self.bdry_pt = bdry_pt
        self.nx = nx
        self.ny = ny
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.device = device
        self.R_grid, self.Z_grid = torch.meshgrid(
            torch.linspace(
                self.Rmin, self.Rmax, self.nx, dtype=torch.float64, device=device
                ),
            torch.linspace(
                self.Zmin, self.Zmax, self.ny, dtype=torch.float64, device=device
                ),
            indexing='ij'
        )
        
    def zero_bdry(
        self, R: Tensor, Z: Tensor, rhs: Tensor
        ) -> Tuple[Tensor]:
        nonzero_idx = torch.isclose(
            rhs.abs(), torch.zeros_like(rhs), atol=1e-6
            ).to(torch.float64).to(self.device).squeeze(1)
        R_center = (self.R_grid * nonzero_idx).sum(dim=1).sum(dim=1) / \
            nonzero_idx.sum(dim=1).sum(dim=1)
        Z_center = (self.Z_grid * nonzero_idx).sum(dim=1).sum(dim=1) / \
            nonzero_idx.sum(dim=1).sum(dim=1)
        R_center = R_center
        Z_center = Z_center
        pR = R_center - self.Rmin
        qR = self.Rmax - R_center
        pZ = Z_center - self.Zmin
        qZ = self.Zmax - Z_center
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
        d2b_dRdZ = torch.where(
            ((R - self.Rmin) < 0.02) | ((self.Rmax - R) < 0.02) | \
            ((Z - self.Zmin) < 0.02) | ((self.Zmax - Z) < 0.02),
            torch.tensor(0.0, dtype=torch.float64, device=self.device),
            ((pR * (R - self.Rmin) ** (pR - 1) * (self.Rmax - R) ** qR) - \
            (qR * (R - self.Rmin) ** pR * (self.Rmax - R) ** (qR - 1))) * \
            ((pZ * (Z - self.Zmin) ** (pZ - 1) * (self.Zmax - Z) ** qZ) - \
            (qZ * (Z - self.Zmin) ** pZ * (self.Zmax - Z) ** (qZ - 1)))
            )
        b_max = pR ** pR * qR ** qR * pZ ** pZ * qZ ** qZ
        return b / b_max, db_dR / b_max, db_dZ / b_max, d2b_dR2 / b_max, \
            d2b_dZ2 / b_max, d2b_dRdZ / b_max

    def _compute_single_batch_trunk(self, R: Tensor, Z: Tensor):
        psi = self.solver_resi.trunk.forward(torch.concat(
            [R.unsqueeze(0).unsqueeze(-1), Z.unsqueeze(0).unsqueeze(-1)], dim=-1
            ))
        return psi.squeeze(0).squeeze(-1)

    def _compute_single_batch_mlp(self, h: Tensor, b: Tensor):
        y = self.solver_resi.output_mlp.forward(torch.concat(
            [h.unsqueeze(0).unsqueeze(1), b.unsqueeze(0).unsqueeze(1)], dim=-1
        ))
        return y.squeeze(0).squeeze(1).squeeze(-1)
    
    def _compute_single_batch_bdry(
        self, R: Tensor, Z: Tensor, bdry_pt: Tensor, bdry: Tensor
        ) -> Tensor:
        psi = self.solver_bdry.forward(
            R.unsqueeze(0).unsqueeze(1).unsqueeze(2),
            Z.unsqueeze(0).unsqueeze(1).unsqueeze(2),
            bdry_pt, bdry
            )
        return psi.squeeze(0).squeeze(0).squeeze(0)
    
    def compute_grad(
        self, R: Tensor, Z: Tensor, rhs: Tensor, bdry: Tensor, 
        scaling_rhs: Tensor, scaling_bdry: Tensor, synthetic: bool = False
        ) -> Tuple[Tensor]:
        if synthetic:
            xx = (R - self.Rmin) / (self.Rmax - self.Rmin)
            yy = (Z - self.Zmin) / (self.Zmax - self.Zmin)
            psi_val = torch.exp(
                -((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.4 ** 2
                )
            dpsi_dR1 = psi_val * -2 * (xx - 0.5) / 0.4 ** 2 / (
                self.Rmax - self.Rmin
            )
            dpsi_dZ1 = psi_val * -2 * (yy - 0.5) / 0.4 ** 2 / (
                self.Zmax - self.Zmin
            )
            d2psi_dR12 = psi_val * (
                4 * (xx - 0.5) ** 2 / 0.4 ** 4 - 2 / 0.4 ** 2
                ) / (self.Rmax - self.Rmin) ** 2
            d2psi_dR1dZ1 = psi_val * (
                4 * (xx - 0.5) * (yy - 0.5) / 0.4 ** 4
            ) / (self.Rmax - self.Rmin) / (self.Zmax - self.Zmin)
            d2psi_dZ12 = psi_val * (
                4 * (yy - 0.5) ** 2 / 0.4 ** 4 - 2 / 0.4 ** 2
                ) / (self.Zmax - self.Zmin) ** 2
            return psi_val, dpsi_dR1, dpsi_dZ1, d2psi_dR12, d2psi_dZ12, d2psi_dR1dZ1
        else:
            R = R.requires_grad_(True)
            Z = Z.requires_grad_(True)
            b, db_dR, db_dZ, d2b_dR2, d2b_dZ2, d2b_dRdZ = self.zero_bdry(R, Z, rhs)

            grad_point_single_R = jacfwd(self._compute_single_batch_trunk, argnums=0)
            grad_point_single_Z = jacfwd(self._compute_single_batch_trunk, argnums=1)
            grad_point_single_RR = jacfwd(grad_point_single_R, argnums=0)
            grad_point_single_RZ = jacfwd(grad_point_single_R, argnums=1)
            grad_point_single_ZZ = jacfwd(grad_point_single_Z, argnums=1)
            grad_point_single_h = jacrev(self._compute_single_batch_mlp, argnums=0)
            hessian_point_single_h = hessian(self._compute_single_batch_mlp, argnums=0)

            output_branch = self.solver_resi.branch_source(
                rhs / scaling_rhs
                ).reshape(rhs.shape[0], -1)
            output_branch = self.solver_resi.branch_mlp(output_branch).expand(len(R), -1)
            h = self.solver_resi.trunk.forward(torch.concat(
                [R[:, None], Z[:, None]], dim=1
                ))
            y = self.solver_resi.output_mlp.forward(torch.concat(
                [h, output_branch], dim=1
            )).squeeze(-1) * scaling_rhs
            dy_dh = vmap(grad_point_single_h)(h, output_branch)
            d2y_dh2 = vmap(hessian_point_single_h)(h, output_branch)
            dh_dR = vmap(grad_point_single_R)(R, Z)
            dh_dZ = vmap(grad_point_single_Z)(R, Z)
            dh2_dR2 = vmap(grad_point_single_RR)(R, Z)
            dh2_dZ2 = vmap(grad_point_single_ZZ)(R, Z)
            dh2_dRdZ = vmap(grad_point_single_RZ)(R, Z)
            dy_dR = (dy_dh * dh_dR).sum(dim=1) * scaling_rhs
            dy_dZ = (dy_dh * dh_dZ).sum(dim=1) * scaling_rhs
            d2y_dR2 = torch.einsum('xij, xi, xj -> x', d2y_dh2, dh_dR, dh_dR) + \
                (dy_dh * dh2_dR2).sum(dim=1) * scaling_rhs
            d2y_dZ2 = torch.einsum('xij, xi, xj -> x', d2y_dh2, dh_dZ, dh_dZ) + \
                (dy_dh * dh2_dZ2).sum(dim=1) * scaling_rhs
            d2y_dRdZ = torch.einsum('xij, xi, xj -> x', d2y_dh2, dh_dR, dh_dZ) + \
                (dy_dh * dh2_dRdZ).sum(dim=1) * scaling_rhs

            psi_resi = y * b
            dpsi_resi_dR = (dy_dR * b + y * db_dR).detach()
            dpsi_resi_dZ = (dy_dZ * b + y * db_dZ).detach()
            d2psi_resi_dR2 = (d2y_dR2 * b + 2 * dy_dR * db_dR + y * d2b_dR2).detach()
            d2psi_resi_dZ2 = (d2y_dZ2 * b + 2 * dy_dZ * db_dZ + y * d2b_dZ2).detach()
            d2psi_resi_dRdZ = (d2y_dRdZ * b + dy_dR * db_dZ + dy_dZ * db_dR + y * d2b_dRdZ).detach()

            psi_bdry = self.solver_bdry.forward(
                R[None, :, None], Z[None, :, None], self.bdry_pt,
                bdry / scaling_bdry
                ).squeeze(2).squeeze(0) * scaling_bdry

            grad_point_single_R = jacrev(self._compute_single_batch_bdry, argnums=0)
            grad_point_single_Z = jacrev(self._compute_single_batch_bdry, argnums=1)
            grad_point_single_RR = jacrev(grad_point_single_R, argnums=0)
            grad_point_single_RZ = jacrev(grad_point_single_R, argnums=1)
            grad_point_single_ZZ = jacrev(grad_point_single_Z, argnums=1)
            dpsi_bdry_dR = vmap(
                grad_point_single_R, in_dims=(0, 0, None, None)
                )(R, Z, self.bdry_pt, bdry / scaling_bdry) * scaling_bdry
            dpsi_bdry_dZ = vmap(
                grad_point_single_Z, in_dims=(0, 0, None, None)
                )(R, Z, self.bdry_pt, bdry / scaling_bdry) * scaling_bdry
            d2psi_bdry_dR2 = vmap(
                grad_point_single_RR, in_dims=(0, 0, None, None)
                )(R, Z, self.bdry_pt, bdry / scaling_bdry) * scaling_bdry
            d2psi_bdry_dZ2 = vmap(
                grad_point_single_ZZ, in_dims=(0, 0, None, None)
                )(R, Z, self.bdry_pt, bdry / scaling_bdry) * scaling_bdry
            d2psi_bdry_dRdZ = vmap(
                grad_point_single_RZ, in_dims=(0, 0, None, None)
                )(R, Z, self.bdry_pt, bdry / scaling_bdry) * scaling_bdry

            psi = psi_resi + psi_bdry
            dpsi_dR = dpsi_resi_dR + dpsi_bdry_dR
            dpsi_dZ = dpsi_resi_dZ + dpsi_bdry_dZ
            d2psi_dR2 = d2psi_resi_dR2 + d2psi_bdry_dR2
            d2psi_dZ2 = d2psi_resi_dZ2 + d2psi_bdry_dZ2
            d2psi_dRdZ = d2psi_resi_dRdZ + d2psi_bdry_dRdZ

        return psi, dpsi_dR, dpsi_dZ, d2psi_dR2, d2psi_dZ2, d2psi_dRdZ


    def find_critical(
        self, R: Tensor, Z: Tensor, rhs: Tensor, bdry: Tensor, psi: Tensor,
        tokamak, synthetic: bool = False
        ) -> Tuple[Tensor]:
        if rhs.shape != (1, 1, self.nx, self.ny):
            breakpoint()
        scaling_rhs = rhs.max() - rhs.min()
        scaling_bdry = bdry.max() - bdry.min() if (bdry.max() - bdry.min()) > 1e-6 else 1.0
        rhs = rhs / scaling_rhs
        bdry = bdry / scaling_bdry

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]
        dpsi_dR = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dR)
        dpsi_dZ = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dZ)

        Bp2 = (dpsi_dR ** 2 + dpsi_dZ ** 2) / (R[1:-1, 1:-1] ** 2)
        radius_sq = 9 * (dR ** 2 + dZ ** 2)
        J = torch.zeros((2, 2), dtype=torch.float64, device=self.device)

        xpoint = []
        opoint = []
        for i in range(1, self.nx - 3):
            for j in range(1, self.ny - 3):
                if (
                    (Bp2[i, j] < Bp2[i + 1, j + 1])
                    and (Bp2[i, j] < Bp2[i + 1, j])
                    and (Bp2[i, j] < Bp2[i + 1, j - 1])
                    and (Bp2[i, j] < Bp2[i - 1, j + 1])
                    and (Bp2[i, j] < Bp2[i - 1, j])
                    and (Bp2[i, j] < Bp2[i - 1, j - 1])
                    and (Bp2[i, j] < Bp2[i, j + 1])
                    and (Bp2[i, j] < Bp2[i, j - 1])
                ):
                    R0 = R[i + 1, j + 1][None]
                    Z0 = Z[i + 1, j + 1][None]
                    R1 = R0
                    Z1 = Z0
                    count = 0
                    while True:
                        psi_val, dpsi_dR1, dpsi_dZ1, d2psi_dR12, d2psi_dZ12, \
                            d2psi_dR1dZ1 = self.compute_grad(
                                R1, Z1, rhs, bdry, scaling_rhs, scaling_bdry, synthetic
                            )
                        psi_val += tokamak.psi(R1, Z1)
                        dpsi_dR1 += tokamak.dpsi_dR(R1, Z1)
                        dpsi_dZ1 += tokamak.dpsi_dZ(R1, Z1)
                        d2psi_dR12 += tokamak.d2psi_dR2(R1, Z1)
                        d2psi_dZ12 += tokamak.d2psi_dZ2(R1, Z1)
                        d2psi_dR1dZ1 += tokamak.d2psi_dRdZ(R1, Z1)
                        
                        Br = -dpsi_dZ1 / R1
                        Bz = dpsi_dR1 / R1
                        if Br ** 2 + Bz ** 2 < 1e-6:
                            D = d2psi_dR12 * d2psi_dZ12 - d2psi_dR1dZ1 ** 2
                            if D < 0.0:
                                xpoint.append((R1, Z1, psi_val))
                            else:
                                opoint.append((R1, Z1, psi_val))
                            break

                        J[0, 0] = - Br / R1 - d2psi_dR1dZ1 / R1
                        J[0, 1] = - d2psi_dZ12 / R1
                        J[1, 0] = - Bz / R1 + d2psi_dR12 / R1
                        J[1, 1] = d2psi_dR1dZ1 / R1
                        d = torch.linalg.solve(J, torch.tensor(
                            [Br, Bz], dtype=torch.float64, device=self.device
                            ))
                        R1 = R1 - d[0]
                        Z1 = Z1 - d[1]
                        count += 1
                        
                        if (
                            (R1 - R0) ** 2 + (Z1 - Z0) ** 2 > radius_sq
                            ) or (count > 100):
                            break
        xpoint = self.remove_dup(xpoint)
        opoint = self.remove_dup(opoint)
        if len(opoint) == 0:
            print("Warning[critical]: No O-points found")
            return opoint, xpoint
        
        Rmid = 0.5 * (self.Rmax + self.Rmin)
        Zmid = 0.5 * (self.Zmax + self.Zmin)
        opoint.sort(key=lambda x: (x[0] - Rmid) ** 2 + (x[1] - Zmid) ** 2)

        # Draw a line from the O-point to each X-point. Psi should be
        # monotonic; discard those which are not

        Ro, Zo, Po = opoint[0]  # The primary O-point
        xpt_keep = []
        for xpt in xpoint:
            Rx, Zx, Px = xpt

            rline = torch.linspace(
                Ro.squeeze(), Rx.squeeze(), steps=50, dtype=torch.float64,
                device=self.device
                )
            zline = torch.linspace(
                Zo.squeeze(), Zx.squeeze(), steps=50, dtype=torch.float64,
                device=self.device
                )
            b, _, _, _, _, _ = self.zero_bdry(
                rline, zline, rhs
            )
            rline = rline.reshape(1, -1, 1)
            zline = zline.reshape(1, -1, 1)
            if synthetic:
                pline = torch.exp(
                    -(((rline - self.Rmin) / (self.Rmax - self.Rmin) - 0.5) ** 2 +
                    ((zline - self.Zmin) / (self.Zmax - self.Zmin) - 0.5) ** 2) / 0.4 ** 2
                    ).squeeze()
                pline = pline + tokamak.psi(rline, zline).squeeze()
            else:
                pline_resi = scaling_rhs * self.solver_resi(
                    rline, zline, rhs / scaling_rhs
                    ).squeeze()
                pline_resi *= b
                pline_bdry = scaling_bdry * \
                    self.solver_bdry(
                        rline, zline, bdry / scaling_bdry
                        ).squeeze()
                pline = pline_resi + pline_bdry + tokamak.psi(
                    rline, zline
                ).squeeze()

            if Px < Po:
                pline *= -1.0  # Reverse, so pline is maximum at X-point

            # Now check that pline is monotonic
            # Tried finding maximum (argmax) and testing
            # how far that is from the X-point. This can go
            # wrong because psi can be quite flat near the X-point
            # Instead here look for the difference in psi
            # rather than the distance in space

            maxp = torch.amax(pline)
            if (maxp - pline[-1]) / (maxp - pline[0]) > 0.001:
                # More than 0.1% drop in psi from maximum to X-point
                # -> Discard
                continue

            ind = torch.argmin(pline)  # Should be at O-point
            if (
                (rline.squeeze()[ind] - Ro.squeeze()) ** 2 + \
                (zline.squeeze()[ind] - Zo.squeeze()) ** 2
                ) > 1e-4:
                # Too far, discard
                continue
            xpt_keep.append(xpt)
        xpoint = xpt_keep

        # Sort X-points by distance to primary O-point in psi space
        psi_axis = opoint[0][2]
        xpoint.sort(key=lambda x: (x[2] - psi_axis) ** 2)

        return opoint, xpoint

    def core_mask(
        self, R: Tensor, Z: Tensor, psi: Tensor, opoint: Tuple[Tensor], 
        xpoint: Tensor = [], psi_bndry: Optional[float] = None
        ) -> Tensor:
        mask = torch.zeros_like(psi, dtype=torch.float64, device=self.device)
        nx, ny = psi.shape

        Ro, Zo, psi_axis = opoint[0]
        if psi_bndry is None:
            _, _, psi_bndry = xpoint[0]
        psin = (psi - psi_axis) / (psi_bndry - psi_axis)
        xpt_inds = []
        for rx, zx, _ in xpoint:
            ix = torch.argmin(abs(R[:, 0] - rx)).item()
            jx = torch.argmin(abs(Z[0, :] - zx)).item()
            xpt_inds.append((ix, jx))
            for i in np.clip([ix - 1, ix, ix + 1], 0, nx - 1):
                for j in np.clip([jx - 1, jx, jx + 1], 0, ny - 1):
                    mask[i, j] = 2 
        rind = torch.argmin(abs(R[:, 0] - Ro))
        zind = torch.argmin(abs(Z[0, :] - Zo))

        stack = [(rind.item(), zind.item())]  # List of points to inspect in future

        while stack:  # Whilst there are any points left
            i, j = stack.pop()  # Remove from list

            # Check the point to the left (i,j-1)
            if (j > 0) and (psin[i, j - 1] < 1.0) and (mask[i, j - 1] < 0.5):
                stack.append((i, j - 1))

            # Scan along a row to the right
            while True:
                mask[i, j] = 1  # Mark as in the core

                if (i < nx - 1) and (psin[i + 1, j] < 1.0) and (mask[i + 1, j] < 0.5):
                    stack.append((i + 1, j))
                if (i > 0) and (psin[i - 1, j] < 1.0) and (mask[i - 1, j] < 0.5):
                    stack.append((i - 1, j))

                if j == ny - 1:  # End of the row
                    break
                if (psin[i, j + 1] >= 1.0) or (mask[i, j + 1] > 0.5):
                    break  # Finished this row
                j += 1  # Move to next point along

        # Now return to X-point locations
        for ix, jx in xpt_inds:
            for i in np.clip([ix - 1, ix, ix + 1], 0, nx - 1):
                for j in np.clip([jx - 1, jx, jx + 1], 0, ny - 1):
                    if psin[i, j] < 1.0:
                        mask[i, j] = 1
                    else:
                        mask[i, j] = 0

        return mask
    
    def remove_dup(self, points: List[Tuple[Tensor]]) -> List[Tuple[Tensor]]:
        result = []
        for p in points:
            is_duplicate = False
            for p2 in result:
                if torch.norm(p[0] - p2[0]) < 1e-5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                result.append(p)
        return result
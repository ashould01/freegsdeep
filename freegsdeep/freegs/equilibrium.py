import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from freegsdeep.freegs.utils import romberg as romb
from freegsdeep.freegs.critical import critical
from freegs.freegs import polygons
from freegsdeep.freegs import machine
from freegsdeep.freegs.spline import TorchBicubicSpline2D
from freegs.freegs.gradshafranov import GSsparse
# from freegs.freegs.boundary import freeBoundaryHagenow as freeBoundaryHagenow_numerical
import freegs.freegs.multigrid as multigrid
import matplotlib.pyplot as plt

from freegsdeep.freegs.boundary import freeBoundaryHagenow
from freegsdeep.typing import *
from freegsdeep.model import DeepONet_resi, PINTO

class Equilibrium:
    def __init__(
        self,
        tokamak: machine.Machine,  
        Rmin: float, Rmax: float, Zmin: float, Zmax: float,
        nx: int, ny: int, boundary: freeBoundaryHagenow,
        load_path_resi: str, load_path_bdry: str,
        psi: Optional[np.ndarray] = None,
        current: float = 0.0, check_limited: bool = False, 
        device: str = 'cuda:0'
        ):
        self.tokamak = tokamak
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.nx = nx
        self.ny = ny
        self.R_1D = torch.linspace(Rmin, Rmax, nx, device=device, dtype=torch.float64)
        self.Z_1D = torch.linspace(Zmin, Zmax, ny, device=device, dtype=torch.float64)
        self.R, self.Z = torch.meshgrid(self.R_1D, self.Z_1D, indexing='ij')
        self.dR = self.R[1, 0] - self.R[0, 0]
        self.dZ = self.Z[0, 1] - self.Z[0, 0]

        _R_bdry = torch.from_numpy(np.linspace(Rmin, Rmax, nx))
        _Z_bdry = torch.from_numpy(np.linspace(Zmin, Zmax, ny))
        self.boundary_L = torch.tensor(
            [[Rmin, _Z_bdry[i]] for i in range(ny)]
        )
        self.boundary_D = torch.tensor(
            [[_R_bdry[i], Zmin] for i in range(nx)]
        )
        self.boundary_R = torch.tensor(
            [[Rmax, _Z_bdry[i]] for i in range(ny)]
        )
        self.boundary_U = torch.tensor(
            [[_R_bdry[i], Zmax] for i in range(nx)]
        )
        self.boundary_pt = torch.concatenate([
            self.boundary_L, self.boundary_D, self.boundary_R, self.boundary_U
        ], dim=0).reshape(1, -1, 2).to(device)
        
        self.R_flat = self.R.reshape(1, -1, 1)
        self.Z_flat = self.Z.reshape(1, -1, 1)

        self.check_limited = check_limited
        self.is_limited = False
        self.Rlim = None
        self.Zlim = None
        self.device = device
        self.mu0 = 4e-7 * np.pi
        
        if psi is None: 
            xx, yy = torch.meshgrid(
                torch.linspace(0, 1, nx, dtype=torch.float64, device=device),
                torch.linspace(0, 1, ny, dtype=torch.float64, device=device),
                indexing='ij'
            )
            psi = torch.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.4 ** 2)
            psi[0, :] = 0.0
            psi[-1, :] = 0.0
            psi[:, 0] = 0.0
            psi[:, -1] = 0.0

        self._pgreen = tokamak.createPsiGreens(self.R, self.Z)
        self._current = current
        self.Jtor = None
        
        self._solver_resi = DeepONet_resi(
            Rmin, Rmax, Zmin, Zmax, nx, ny
        ).to(torch.float64).to(device)
        self._solver_bdry = PINTO(
            Rmin, Rmax, Zmin, Zmax, nx, ny
        ).to(torch.float64).to(device)
        self.call_parameter(load_path_resi, load_path_bdry)
        self._applyBoundary = boundary(
            self.R, self.Z, self.boundary_pt.reshape(-1, 2), nx, ny, 
            self.dR.item(), self.dZ.item(), device=self.device, 
            solver=self._solver_resi,
            )
        self.critical = critical(device=device)
        self.plasma_psi = psi

        self._updateBoundaryPsi(psi)
        self.psi_func = TorchBicubicSpline2D(
            self.R[:, 0], self.Z[0, :], self.plasma_psi
            )

        # Debugging for boundary update
        generator = GSsparse(Rmin, Rmax, Zmin, Zmax)
        self._solver = multigrid.createVcycle(
            nx, ny, generator, nlevels=1, ncycle=1, niter=2, direct=True
        )
        self.trapz_matrix = torch.zeros(
            (nx, ny), dtype=torch.float64,
            device=self.device
            )
        self.trapz_matrix[0, 0] = self.trapz_matrix[-1, 0] = \
            self.trapz_matrix[0, -1] = self.trapz_matrix[-1, -1] = self.dR * self.dZ / 4
        self.trapz_matrix[0, 1:-1] = self.trapz_matrix[-1, 1:-1] = \
            self.trapz_matrix[1:-1, 0] = self.trapz_matrix[1:-1, -1] = self.dR * self.dZ / 2
        self.trapz_matrix[1:-1, 1:-1] = self.dR * self.dZ
    
    def callSolver(self, psi, rhs):
        return self._solver(psi, rhs)

    def call_parameter(self, load_path_resi: str, load_path_bdry: str): 
        self._solver_resi.load_state_dict(
            torch.load(load_path_resi, map_location=self.device, weights_only=True)
        )
        self._solver_bdry.load_state_dict(
            torch.load(load_path_bdry, map_location=self.device, weights_only=True)
        )
        self._solver_resi.eval()
        self._solver_bdry.eval()
    
    def psi(self):
        return self.plasma_psi + \
            self.tokamak.calcPsiFromGreens(self._pgreen)
    
    def _profiles(self, profiles):
        self._profiles = profiles
        
    def _updateBoundaryPsi(
        self, psi: Optional[Array] = None
        ):
        if psi is None:
            psi = self.psi()
            
        opt, xpt = self.critical.find_critical(self.R, self.Z, psi)

        if opt:
            self.psi_axis = opt[0][2]
            if self.check_limited and self.tokamak.wall:
                Rlimit = self.tokamak.limit_points_R
                Zlimit = self.tokamak.limit_points_Z
                if xpt:
                    limit_args = np.ravel(
                        np.argwhere(abs(Zlimit) < abs(0.75 * xpt[0][1]))
                    )
                    Rlimit = Rlimit[limit_args]
                    Zlimit = Zlimit[limit_args]
                R = np.asarray(self.R[:, 0])
                Z = np.asarray(self.Z[0, :])
                psi_2d = TorchBicubicSpline2D(R, Z, psi.T)
                
                psi_limit_points = np.zeros(len(Rlimit))
                for i in range(len(Rlimit)):
                    psi_limit_points[i] = psi_2d(Rlimit[i], Zlimit[i])[0]
                indMax = np.argmax(psi_limit_points)
                self.Rlim = Rlimit[indMax]
                self.Zlim = Zlimit[indMax]
                self.psi_limit = psi_limit_points[indMax]
                if xpt:
                    self.psi_xpt = xpt[0][2]
                    self.psi_bndry = max(self.psi_limit, self.psi_xpt)
                    if self.psi_bndry == self.psi_limit:
                        self.is_limited = True
                    else:
                        self.is_limited = False
                else:
                    self.psi_bndry = self.psi_limit
                    self.is_limited = True
                    self.mask = None
            else:
                if xpt:
                    self.psi_xpt = xpt[0][2]
                    self.psi_bndry = self.psi_xpt
                    self.mask = self.critical.core_mask(self.R, self.Z, psi, opt, xpt)
                    self.mask_func = TorchBicubicSpline2D(
                        self.R[:, 0], self.Z[0, :], self.mask
                    )
                else:
                    self.psi_bndry = None
                    self.mask = None
                self.is_limited = False
                    
    def zero_bdry(
        self, R: Tensor, Z: Tensor, rhs: Tensor
        ) -> Tuple[Tensor]:
        nonzero_idx = torch.isclose(
            rhs.abs(), torch.zeros_like(rhs), atol=1e-6
            ).to(torch.float64).to(self.device).squeeze(1)
        R_grid = self.R.unsqueeze(0)
        Z_grid = self.Z.unsqueeze(0)
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
        b_max = pR ** pR * qR ** qR * pZ ** pZ * qZ ** qZ
        return b / b_max
    
    def solve(self, Jtor=None, psi=None, psi_bndry=None):

        if Jtor is None:
            if psi is None:
                psi = self.psi()
            Jtor = self._profiles.Jtor(
                self.R, self.Z, psi, psi_bndry=psi_bndry
                )
        else:
            Jtor = Jtor.reshape(self.nx, self.ny).detach().cpu().numpy()
        R_times_Jtor = self.R * Jtor
        plasma_psi_resi, psi_bdry = self._applyBoundary(
            R_times_Jtor[None, None, ...],
            self.zero_bdry(
                self.R_flat.squeeze(), self.Z_flat.squeeze(),
                R_times_Jtor.reshape(1, self.nx, self.ny)
                ))

        scaling_bdry = torch.max(psi_bdry) - torch.min(psi_bdry)
        with torch.no_grad():
            plasma_psi_bdry = self._solver_bdry.forward(
                self.R_flat, self.Z_flat, self.boundary_pt, psi_bdry / scaling_bdry
                ) * scaling_bdry
        self.plasma_psi = plasma_psi_resi.reshape(self.nx, self.ny) + \
            plasma_psi_bdry.reshape(self.nx, self.ny)
        self.psi_func = TorchBicubicSpline2D(
            self.R[:, 0], self.Z[0, :], self.plasma_psi
        )
        
        self._updateBoundaryPsi()
        self._current = torch.sum(Jtor * self.trapz_matrix) * self.dR * self.dZ
        self.Jtor = Jtor
        print(self.psi_bndry)

    def getMachine(self):
        """
        Returns the handle of the machine, including coils
        """
        return self.tokamak

    def plasmaCurrent(self):
        """
        Plasma current [Amps]
        """
        return self._current

    def plasmaVolume(self):
        """Calculate the volume of the plasma in m^3"""

        dR = self.dR
        dZ = self.dZ
        # Volume element
        dV = 2.0 * np.pi * self.R_cpu * dR * dZ

        if self.mask is not None:  # Only include points in the core
            dV *= self.mask

        # Integrate volume in 2D
        return romb(romb(dV))

    def plasmaBr(self, R, Z):
        """
        Radial magnetic field due to plasma
        Br = -1/R dpsi/dZ
        """
        return -self.psi_func(R, Z, dy=1, grid=False) / R

    def plasmaBz(self, R, Z):
        """
        Vertical magnetic field due to plasma
        Bz = (1/R) dpsi/dR
        """
        return self.psi_func(R, Z, dx=1, grid=False) / R

    def Br(self, R, Z):
        """
        Total radial magnetic field
        """
        return self.plasmaBr(R, Z) + self.tokamak.Br(R, Z)

    def Bz(self, R, Z):
        """
        Total vertical magnetic field
        """
        return self.plasmaBz(R, Z) + self.tokamak.Bz(R, Z)

    def Bpol(self, R, Z):
        """
        Total poloidal magnetic field
        """
        Br = self.Br(R, Z)
        Bz = self.Bz(R, Z)
        return np.sqrt(Br * Br + Bz * Bz)

    def Btor(self, R, Z):
        """
        Toroidal magnetic field
        """
        # Normalised psi
        psi_norm = (self.psiRZ(R, Z) - self.psi_axis) / (self.psi_bndry - self.psi_axis)

        # Get f = R * Btor in the core. May be invalid outside the core
        fpol = self.fpol(psi_norm)

        if self.mask is not None:
            # Get the values of the core mask at the requested R,Z locations
            # This is 1 in the core, 0 outside
            mask = self.mask_func(R, Z, grid=False)
            fpol = fpol * mask + (1.0 - mask) * self.fvac()

        return fpol / R

    def Btot(self, R, Z):
        """
        Total magnetic field
        """
        Br = self.Br(R, Z)
        Bz = self.Bz(R, Z)
        Btor = self.Btor(R, Z)
        return np.sqrt(Br * Br + Bz * Bz + Btor * Btor)

    def psiN(self):
        """
        Total poloidal flux (psi), including contribution from
        plasma and external coils. Normalised such that psiN = 0 on
        the magnetic axis and 1 on the LCFS.
        """
        # return self.plasma_psi + self.tokamak.psi(self.R, self.Z)
        return (self.psi() - self.psi_axis) / (self.psi_bndry - self.psi_axis)

    def psiRZ(self, R, Z):
        """
        Return poloidal flux psi at given (R,Z) location
        """
        return self.psi_func(R, Z, grid=False) + self.tokamak.psi(R, Z)

    def psiNRZ(self, R, Z):
        """
        Return poloidal flux psi at given (R,Z) location. Normalised such
        that psiN = 0 on the magnetic axis and 1 on the LCFS.
        """
        return (self.psiRZ(R, Z) - self.psi_axis) / (self.psi_bndry - self.psi_axis)

    def fpol(self, psinorm):
        """
        Return f = R*Bt at specified values of normalised psi
        """
        return self._profiles.fpol(psinorm)

    def fvac(self):
        """
        Return vacuum f = R*Bt
        """
        return self._profiles.fvac()
from scipy.integrate import quad
# from scipy.integrate import romb
import numpy as np
import abc
from freegsdeep.typing import *
from freegsdeep.freegs.equilibrium import Equilibrium
from freegsdeep.freegs.critical import critical
from freegsdeep.freegs.utils import romberg as romb

class Profile(abc.ABC):
    def pressure(self, psinorm, out=None):
        if not hasattr(psinorm, "shape"):
            val, _ = quad(self.pprime, psinorm, 1.0)
            return val * (self.psi_axis - self.psi_bndry)
    
        if out is None:
            out = np.zeros_like(psinorm)
        
        pvals = psinorm.reshape(-1)
        ovals = out.reshape(-1)
        if len(pvals) != len(ovals):
            raise ValueError(
                f"Input and output array has different lengths; " \
                f"Input {len(pvals)}, Output {len(ovals)}"
                )
        for i in range(len(pvals)):
            val, _ = quad(self.pprime, pvals[i], 1.0)
            val *= self.psi_axis - self.psi_bndry
            ovals[i] = val
        
        return ovals.reshape(psinorm.shape)
    
    def fpol(self, psinorm, out=None):
        
        if not hasattr(psinorm, "__len__"):
            # Assume a single value

            val, _ = quad(self.ffprime, psinorm, 1.0)
            # Convert from integral in normalised psi to integral in psi
            val *= self.psi_axis - self.psi_bndry

            # ffprime = 0.5*d/dpsi(f^2)
            # Apply boundary condition at psinorm=1 val = fvac**2
            return np.sqrt(2.0 * val + self.fvac() ** 2)

        # Assume it's a NumPy array, or can be converted to one
        psinorm = np.array(psinorm)

        if out is None:
            out = np.zeros(psinorm.shape)

        pvals = psinorm.reshape(-1)
        ovals = out.reshape(-1)

        if len(pvals) != len(ovals):
            raise ValueError(
                f"Input and output array has different lengths; " \
                f"Input {len(pvals)}, Output {len(ovals)}"
                )
        for i in range(len(pvals)):
            val, _ = quad(self.ffprime, pvals[i], 1.0)
            # Convert from integral in normalised psi to integral in psi
            val *= self.psi_axis - self.psi_bndry

            # ffprime = 0.5*d/dpsi(f^2)
            # Apply boundary condition at psinorm=1 val = fvac**2
            ovals[i] = np.sqrt(2.0 * val + self.fvac() ** 2)

        return ovals.reshape(psinorm.shape)

    @abc.abstractmethod
    def Jtor(
        self, R: Array, Z: Array, psi: Array, psi_bndry: Optional[float]=None
        ) -> Array:
        ...
        
    @abc.abstractmethod
    def pprime(self, psinorm: float) -> float:
        ...

    @abc.abstractmethod
    def ffprime(self, psinorm: float) -> float:
        ...
    
    @abc.abstractmethod
    def fvac(self) -> float:
        ...

class ConstrainPaxisIp(Profile):
    def __init__(
        self, eq: Equilibrium, paxis: float, Ip: float, fvac: float,
        alpha_m: float=1.0, alpha_n: float=2.0, Raxis: float=1.0
        ) -> None:
    
        if alpha_m <= 0.0 or alpha_n <= 0.0:
            raise ValueError(
                "Alpha parameters must be positive;" \
                f" got alpha_m={alpha_m}, alpha_n={alpha_n}"
                ) 
        self.paxis = paxis
        self.Ip = Ip
        self._fvac = fvac
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n
        self.Raxis = Raxis
        self.eq = eq
        self.mu0 = 4e-7 * torch.pi
        self.trapz_matrix = self.eq.trapz_matrix
        self.dR = self.eq.dR
        self.dZ = self.eq.dZ
        
    def Jtor(
        self, R: Tensor, Z: Tensor, psi: Tensor, psi_bndry: Optional[float]=None
        ) -> Array:
        self.eq._updateBoundaryPsi(psi)
        psi_bndry = self.eq.psi_bndry 

        # Analyse the equilibrium, finding O- and X-points
        opt, xpt = self.eq.critical.find_critical(R, Z, psi)
        if not opt:
            print("Warning[Jtor]: No O-points found!")
            return None
        psi_axis = opt[0][2]

        if psi_bndry is not None:
            mask = self.eq.critical.core_mask(R, Z, psi, opt, xpt, psi_bndry)
        elif xpt:
            psi_bndry = xpt[0][2]
            mask = self.eq.critical.core_mask(R, Z, psi, opt, xpt)
        else:
            # No X-points
            psi_bndry = psi[0, 0]
            mask = None

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        # Current profile shape
        jtorshape = (1.0 - torch.clip(psi_norm, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n

        if mask is not None:
            # If there is a masking function (X-points, limiters)
            jtorshape *= mask

        # Now apply constraints to define constants

        # Need integral of jtorshape to calculate paxis
        # Note factor to convert from normalised psi integral
        shapeintegral, _ = quad(
            lambda x: (1.0 - x**self.alpha_m) ** self.alpha_n, 0.0, 1.0
        )
        shapeintegral *= psi_bndry - psi_axis

        # Pressure on axis is
        # paxis = - (L*Beta0/Raxis) * shapeintegral

        # Integrate current components
        IR = romb(
            romb(jtorshape * R / self.Raxis, dx=self.dZ),
            dx=self.dR
            )
        I_R = romb(
            romb(jtorshape * self.Raxis / R, dx=self.dZ), 
            dx=self.dR
            )
        # IR = torch.sum(
        #     self.trapz_matrix * jtorshape * R / self.Raxis
        #     )
        # I_R = torch.sum(
        #     self.trapz_matrix * jtorshape * self.Raxis / R
        #     )
        # IR = torch.trapz(
        #     torch.trapz(jtorshape * R / self.Raxis, dx=dZ, dim=1), dx=dR, dim=0
        # )
        # I_R = torch.trapz(
        #     torch.trapz(jtorshape * self.Raxis / R, dx=dZ, dim=1), dx=dR, dim=0
        # )

        # Toroidal plasma current Ip is
        # Ip = L * (Beta0 * IR + (1-Beta0)*I_R)

        LBeta0 = -self.paxis * self.Raxis / shapeintegral

        L = self.Ip / I_R - LBeta0 * (IR / I_R - 1)
        Beta0 = LBeta0 / L

        # print("Constraints: L = %e, Beta0 = %e" % (L, Beta0))

        # Toroidal current
        Jtor = L * (Beta0 * R / self.Raxis + (1 - Beta0) * self.Raxis / R) * jtorshape

        self.L = L
        self.Beta0 = Beta0
        self.psi_bndry = psi_bndry
        self.psi_axis = psi_axis

        return Jtor

    def pprime(self, pn):
        """
        dp/dpsi as a function of normalised psi. 0 outside core
        Calculate pprimeshape inside the core only
        """
        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        return self.L * self.Beta0 / self.Raxis * shape

    def ffprime(self, pn):
        """
        f * df/dpsi as a function of normalised psi. 0 outside core.
        Calculate ffprimeshape inside the core only.
        """
        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        return self.mu0 * self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        return self._fvac
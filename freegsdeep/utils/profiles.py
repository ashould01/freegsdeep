from scipy.integrate import romb, quad
import numpy as np
import abc
from freegsdeep.utilstyping import *
from freegsdeep.utils.equilibrium import Equilibrium

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
        
    def Jtor(self, R: Array, Z: Array, psi: Array) -> Array:
        self.eq._updateBoundaryPsi(psi)
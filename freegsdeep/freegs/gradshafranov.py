import torch
from freegsdeep.utilstyping import *

mu0 = 4e-7 * torch.pi

def Greens(Rc: float, Zc: float, R: Tensor, Z: Tensor) -> Tensor:
    """
    Calculate poloidal flux at (R,Z) due to a unit current
    at (Rc,Zc) using Greens function

    """

    def ellipke(k2: Tensor, n_iter: int = 10) -> Tensor:
        device = k2.device
        a = torch.ones_like(k2, dtype=torch.float64, device=device)
        b = torch.sqrt(1.0 - k2)
        S = torch.zeros_like(k2, dtype=torch.float64, device=device)
        coef = 0.5 * torch.ones_like(k2, dtype=torch.float64, device=device)
        for i in range(1, n_iter + 1):
            S = S + coef * abs(a ** 2 - b ** 2)
            an = 0.5 * (a + b)
            bn = torch.sqrt(a * b)
            a, b = an, bn
            coef = coef * 2.0
        K = torch.pi / 2.0 / a
        E = (1.0 - S) * K
        K = torch.where(k2 == 0, torch.full_like(K, torch.pi / 2.0), K)
        E = torch.where(k2 == 0, torch.full_like(E, torch.pi / 2.0), E)
        return K, E

    # Calculate k^2
    k2 = 4.0 * R * Rc / ((R + Rc) ** 2 + (Z - Zc) ** 2)

    # Clip to between 0 and 1 to avoid nans e.g. when coil is on grid point
    k2 = torch.clip(k2, 1e-10, 1.0 - 1e-10)
    k = torch.sqrt(k2)

    # Note definition of ellipk, ellipe in scipy is K(k^2), E(k^2)
    ellipk, ellipke = ellipke(k2)
    return (
        (mu0 / (2.0 * torch.pi))
        * torch.sqrt(R * Rc)
        * ((2.0 - k2) * ellipk - 2.0 * ellipke)
        / k
    )

def GreensBz(Rc, Zc, R, Z, eps=1e-3):
    """
    Calculate vertical magnetic field at (R,Z)
    due to unit current at (Rc, Zc)

    Bz = (1/R) d psi/dR
    """
    return (Greens(Rc, Zc, R + eps, Z) - Greens(Rc, Zc, R - eps, Z)) / (2.0 * eps * R)

def GreensBr(Rc, Zc, R, Z, eps=1e-3):
    """
    Calculate radial magnetic field at (R,Z)
    due to unit current at (Rc, Zc)

    Br = -(1/R) d psi/dZ
    """
    return (Greens(Rc, Zc, R, Z - eps) - Greens(Rc, Zc, R, Z + eps)) / (2.0 * eps * R)

def dGreens_dR(Rc, Zc, R, Z, eps=1e-3):
    return (Greens(Rc, Zc, R + eps, Z) - Greens(Rc, Zc, R - eps, Z)) / (2.0 * eps)

def dGreens_dZ(Rc, Zc, R, Z, eps=1e-3):
    return (Greens(Rc, Zc, R, Z + eps) - Greens(Rc, Zc, R, Z - eps)) / (2.0 * eps)

def d2Greens_dR2(Rc, Zc, R, Z, eps=1e-3):
    return (
        Greens(Rc, Zc, R + eps, Z)
        - 2.0 * Greens(Rc, Zc, R, Z)
        + Greens(Rc, Zc, R - eps, Z)
    ) / (eps ** 2)

def d2Greens_dZ2(Rc, Zc, R, Z, eps=1e-3):
    return (
        Greens(Rc, Zc, R, Z + eps)
        - 2.0 * Greens(Rc, Zc, R, Z)
        + Greens(Rc, Zc, R, Z - eps)
    ) / (eps ** 2)
    
def d2Greens_dRdZ(Rc, Zc, R, Z, eps=1e-3):
    return (
        Greens(Rc, Zc, R + eps, Z + eps)
        - Greens(Rc, Zc, R + eps, Z - eps)
        - Greens(Rc, Zc, R - eps, Z + eps)
        + Greens(Rc, Zc, R - eps, Z - eps)
    ) / (4.0 * eps ** 2)
import torch
import numpy as np
import matplotlib.pyplot as plt
from freegsdeep.freegs.equilibrium import Equilibrium
from freegs.freegs.jtor import Profile
from freegs.freegs.plotting import plotEquilibrium
from datetime import datetime

def solve(
    eq: Equilibrium, profiles: Profile, constrain=None, rtol=1e-3, atol=1e-10,
    blend=0.0, show=False, axis=None, pause=0.0001, psi_bndry=None, maxits=100,
    convergenceInfo=False, check_limited=False, wait_for_limited=False,
    limit_it=0, 
) -> None:
    # (1) Initialize psi and apply constraints if any
    if constrain is not None:
        constrain(eq)
    psi = eq.psi()
    iter = 0
    psi_relchange = 10.0
    bndry = 0.0
    bndry_change = 10.0
    has_been_limited = False
    ok_to_break = False
    psi_maxchange_iterations, psi_relchange_iterations = [], []
    eq._profiles(profiles)
    while True:
        # (2) Store last psi and boundary for convergence check
        psi_last = psi.clone()
        bndry_last = bndry
        # (3) Solve GS equation 
        if (iter >= limit_it or has_been_limited) and check_limited:
            eq.check_limited = True
            eq.solve(psi=psi, psi_bndry=eq.psi_bndry)
        else:
            eq.check_limited = False
            eq.solve(psi=psi, psi_bndry=psi_bndry)

        # (4) Check for limiter and convergence
        if eq.is_limited:
            has_been_limited = True
        
        if eq.psi_bndry is not None:
            bndry = eq.psi_bndry
            bndry_change = abs(bndry_last - bndry)
            bndry_relchange = bndry_change / max(abs(bndry), 1e-10)
        else:
            bndry_relchange = 2.0 * rtol           
        
        psi = eq.psi()
        psi_change = psi_last - psi
        psi_maxchange = torch.amax(torch.abs(psi_change))
        psi_relchange = psi_maxchange / (
            torch.amax(psi) - torch.amin(psi) + 1e-10
        )
        psi_maxchange_iterations.append(psi_maxchange)
        psi_relchange_iterations.append(psi_relchange)
        
        if not wait_for_limited:
            ok_to_break = True
        elif wait_for_limited and eq.is_limited:
            ok_to_break = True
        else:
            ok_to_break = False
        
        if (
            ((psi_maxchange < atol) or (psi_relchange < rtol))
            and ((bndry_relchange < rtol) or (abs(bndry_change) < atol))
            and ok_to_break
        ):
            break
        # (5) Re-constrain for the change in psi
        if constrain is not None:
            constrain(eq)

        # # Plotting
        # fig, ax = plt.subplots(figsize=(15, 15))
        # R = eq.R_cpu
        # Z = eq.Z_cpu
        # psi = eq.psi()

        # levels = np.linspace(np.amin(psi), np.amax(psi), 100)

        # ax.contour(R, Z, psi, levels=levels)
        # ax.set_aspect("equal")
        # ax.set_xlabel("Major radius [m]")
        # ax.set_ylabel("Height [m]")

        # opt, xpt = find_critical(eq.R_cpu, eq.Z_cpu, psi)
        # if opt is not None:
        #     for r, z, _ in opt:
        #         ax.plot(r, z, "go")
        #     ax.plot([], [], "go", label="O-points")
        # if xpt is not None:
        #     for r, z, _ in xpt:
        #         ax.plot(r, z, "rx")

        #     psi_bndry = eq.psi_bndry  # xpt[0][2]
        #     ax.contour(eq.R_cpu, eq.Z_cpu, psi, levels=[psi_bndry], colors="r")

        #     # Add legend
        #     ax.plot([], [], "rx", label="X-points")
        #     ax.plot([], [], "r", label="Separatrix")

        # fig.legend()
        # fig.savefig(f"debug/equilibrium_{iter}.png")
        # plt.close()

        # print("psi_relchange: " + str(psi_relchange))
        # print("bndry_relchange: " + str(bndry_relchange))
        # print("bndry_change: " + str(bndry_change))
        # print("\n")
        # if constrain is not None:
            # if the model considers constraints, then implement this part
            # constrain(eq)

        psi = eq.psi() * (1.0 - blend) + psi_last * blend
        iter += 1
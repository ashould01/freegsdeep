import os
import torch
import freegs.freegs as freegs
import freegsdeep
from freegsdeep.freegs.equilibrium import Equilibrium
# from freegs.freegs.boundary import freeBoundaryHagenow
from freegsdeep.freegs.boundary import freeBoundaryHagenow
from freegsdeep.freegs.machine import TestTokamak
from freegsdeep.freegs.control import constrain
from freegsdeep.freegs.profiles import ConstrainPaxisIp
import cProfile
import pstats
import io

def main(device: str = 'cuda:7'):

    tokamak = TestTokamak()

    # Please implement not to reload the model weight
    eq = Equilibrium(
        tokamak=tokamak, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, 
        nx=65, ny=65, boundary=freeBoundaryHagenow, device=device,
        load_path_resi=os.path.join(
            "logs/plain_lbfgs_large_dataset_hard_bdry_251121_195926",
            "model/model_deeponet.pt"
        ),
        load_path_bdry=os.path.join(
            "logs/pinto_251125_125536",
            "model/model_pinto.pt"
        )
        )

    profiles = freegsdeep.freegs.profiles.ConstrainPaxisIp(
        eq, 1e3, 2e5, 2.0
    )

    xpoints = torch.tensor(
        [[1.1, -0.6], [1.1, 0.8]], dtype=torch.float64, device=device
        )

    isoflux = torch.tensor(
        [[1.1, -0.6, 1.1, 0.8]], dtype=torch.float64, device=device
        )

    constraint = constrain(xpoints=xpoints, isoflux=isoflux)

    freegsdeep.solve(
        eq, profiles, constraint, show=True
    )

if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    main(device='cuda:0')
    # profiler.disable()
    # with open("profile_output.txt", "w") as f:
    #     ps = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
    #     ps.print_stats()

# print(f"Plasma Current : {eq.plasmaCurrent()} (A)")
# print(f"Plasma Pressure on axis : {eq.pressure(0.0)} (Pa)")
# print(f"Poloidal Beta : {eq.poloidalBeta()}")

# tokamak.printCurrents()
# tokamak.printMeasurements()
# eq.printForces()

# with open("logs/geqdsk/test.geqdsk", "w") as f:
#     geqdsk.write(eq, f)
    
# axis = eq.plot(show=False)
# eq.tokamak.plot(axis=axis, show=False)
# constrain.plot(axis=axis, show=False)

# import matplotlib.pyplot as plt
# plt.plot(*eq.q())
# plt.xlabel(r"Normalised $\psi$")
# plt.ylabel("Safety factor")
# # debug ...
# plt.savefig("logs/image/test.png")
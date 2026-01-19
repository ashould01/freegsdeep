import os
import torch
import freegs.freegs as freegs
import freegsdeep
from freegsdeep.freegs.machine import TestTokamak
from freegsdeep.freegs.equilibrium import Equilibrium
from freegsdeep.freegs.profiles import ConstrainPaxisIp
# from freegs.freegs.boundary import freeBoundaryHagenow
from freegsdeep.freegs.boundary import freeBoundaryHagenow
from freegsdeep.freegs.control import constrain
from datetime import datetime
import cProfile
import pstats
import io

def main(device: str = 'cuda:7', iterationstep: int = 1000):
    start = datetime.now()

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
        eq, profiles, iteration=iterationstep, constrain=constraint, 
    )
    print("Solve time: ", datetime.now() - start)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main(device='cuda:0', iterationstep=10)
    profiler.disable()
    with open("profile_output_bdryupdate.txt", "w") as f:
        ps = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
        ps.print_stats()

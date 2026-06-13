import numpy as np
import pickle
from freegsnke.freegsnke import (
    build_machine, 
    equilibrium_update,
    GSstaticsolver
)
import torch
from freegsnke.freegsnke.jtor_update import ConstrainPaxisIp

def main(initial_psi=None):
    tokamak = build_machine.tokamak(
        active_coils_path=f"freegsnke/machine_configs/MAST-U/MAST-U_like_active_coils.pickle",
        passive_coils_path=f"freegsnke/machine_configs/MAST-U/MAST-U_like_passive_coils.pickle",
        limiter_path=f"freegsnke/machine_configs/MAST-U/MAST-U_like_limiter.pickle",
        wall_path=f"freegsnke/machine_configs/MAST-U/MAST-U_like_wall.pickle",
    )

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,      # provide tokamak object
        Rmin=0.1, Rmax=2.0,   # radial range
        Zmin=-2.2, Zmax=2.2,  # vertical range
        nx=65,                # number of grid points in the radial direction (needs to be of the form (2**n + 1) with n being an integer)
        ny=129,               # number of grid points in the vertical direction (needs to be of the form (2**n + 1) with n being an integer
        psi=initial_psi
    )
    profiles = ConstrainPaxisIp(
        eq=eq,        # equilibrium object
        paxis=8e3,    # profile object
        Ip=6e5,       # plasma current
        fvac=0.5,     # fvac = rB_{tor}
        alpha_m=1.8,  # profile function parameter
        alpha_n=1.2   # profile function parameter
    )

    GSStaticSolver = GSstaticsolver.NKGSsolver(eq)    
    with open('freegsnke/examples/data/simple_diverted_currents_PaxisIp.pk', 'rb') as f:
        currents_dict = pickle.load(f)
        
    # assign currents to the eq object
    for key in currents_dict.keys():
        eq.tokamak.set_coil_current(coil_label=key, current_value=currents_dict[key])

    GSStaticSolver.solve(
        eq=eq, 
        profiles=profiles, 
        constrain=None, 
        target_relative_tolerance=1e-9,
        verbose=True, # print output
        )

if __name__ == "__main__":
    data_psi = torch.load(
        'data/data_debug_one_sample/psi_f.pt', map_location=torch.device('cpu'),
        weights_only=True
        )
    data_F_psi = torch.load(
        'data/data_debug_one_sample/res0_f.pt', map_location=torch.device('cpu'),
        weights_only=True
    )
    data_psi = np.stack(data_psi.numpy(), axis=1)
    data_F_psi = np.stack(data_F_psi.numpy(), axis=1)
    mean_x = np.mean(data_psi, axis=1, keepdims=True)
    mean_f = np.mean(data_F_psi, axis=1, keepdims=True)
    Xc = data_psi - mean_x
    Fc = data_F_psi - mean_f
    Ux, Sx, _ = np.linalg.svd(Xc, full_matrices=False)
    Uf, Sf, _ = np.linalg.svd(Fc, full_matrices=False)
    d = min(Ux.shape[1], Uf.shape[1], 10)

    

    breakpoint()
    
    main()
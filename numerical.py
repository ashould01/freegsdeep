import freegs
import freegs.boundary
import freegs.jtor
from freegs.machine import Coil, Wall
import numpy as np

tokamak = freegs.machine.TestTokamak()
boundary = freegs.boundary.freeBoundaryHagenow

wall1 = ([0.75] * 101, np.linspace(-0.85, 0.85, 101))  # Left vertical
wall2 = (np.linspace(0.75, 1.5, 101), [0.85] * 101)  # Top horizontal
wall3 = (np.linspace(1.5, 1.8, 101), np.linspace(0.85, 0.25, 101))  # Top right slant
wall4 = ([1.8] * 101, np.linspace(0.25, -0.25, 101))  # Right vertical
wall5 = (np.linspace(1.8, 1.5, 101), np.linspace(-0.25, -0.85, 101))  # Bottom right slant
wall6 = (np.linspace(1.5, 0.75, 101), [-0.85] * 101)  # Bottom horizontal

tokamak.wall = Wall(
    np.concatenate([wall1[0], wall2[0], wall3[0], wall4[0], wall5[0], wall6[0]]),
    np.concatenate([wall1[1], wall2[1], wall3[1], wall4[1], wall5[1], wall6[1]])
    )
eq = freegs.Equilibrium(
    tokamak=tokamak, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0,
    nx=65, ny=65, boundary=boundary
)

# profiles = freegs.jtor.ConstrainBetapIp(
#     eq, betap=0.8, Ip=-5*1e4, fvac=1.0
# )
profiles = freegs.jtor.ConstrainPaxisIp(
    eq, 1e3, Ip=2e5, fvac=2.0
)

# tokamak.coils[0][1].current = -5e4
# tokamak.coils[1][1].current = 5e4

# tokamak.coils[2][1].current = 1e4
# tokamak.coils[3][1].current = -1e4
xpoints = [(1.1, -0.8), (1.1, 0.8)]
isoflux = [(1.1, -0.6, 1.1, 0.6)]
constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)
# constrain = None

freegs.solve(
    eq, profiles, constrain, show=True
)

print(f"Plasma Current : {eq.plasmaCurrent():.4f}")
print(f"Plasma Pressure on axis : {eq.pressure(0.0):.4f}")
print(f"Poloidal beta : {eq.poloidalBeta():.4f}")

tokamak.printCurrents()
eq.printForces()
from freegsdeep.utilstyping import *
from freegsdeep.freegs.coil import Coil, ShapedCoil
from scipy.interpolate import interp1d


class Machine:
    """
    Represents the machine (Tokamak), including
    coils and power supply circuits

    coils[(label, Coil|Circuit|Solenoid] - List of coils

    Note: a list is used rather than a dict, so that the coils
    remain ordered, and so can be updated easily by the control system.
    Instead __getitem__ is implemented to allow access to coils

    """

    def __init__(self, coils: Coil, wall=None, sensors=None, nlimit=500, R0=1.0):
        """
        coils - A list of coils [(label, Coil|Circuit|Solenoid)]
        sensors - A list of sensors
        R0 - The major radius to be written to GEQDSK [meters].
             By convention EFIT has a value of R0 for each tokamak.
        """

        self.coils = coils
        self.wall = wall
        self.sensors = sensors
        self.R0 = R0

        self.limit_points_R = None
        self.limit_points_Z = None

        if self.wall is not None:
            self.limit_points_R, self.limit_points_Z = self.generate_limit_points(
                nlimit
            )

    def __repr__(self):
        return "Machine(coils={coils}, wall={wall})".format(
            coils=self.coils, wall=self.wall
        )

    def __eq__(self, other):
        # Other Machine might be equivalent except for order of
        # coils. Assume this doesn't actually matter
        return sorted(self.coils) == sorted(other.coils) and self.wall == other.wall

    def __ne__(self, other):
        return not self == other

    def __getitem__(self, name):
        for label, coil in self.coils:
            if label == name:
                return coil
        raise KeyError("Machine does not contain coil with label '{0}'".format(name))

    def generate_limit_points(self, nlimit):
        """
        Generate points along the machine wall that may be used to check
        if the plasma is limited or not.
        """

        # Interpolate wall limit points.
        # Make an interpolator for point location as function of normalised distance
        # along the wall
        points = np.array([self.wall.R, self.wall.Z]).T
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        distance = np.insert(distance, 0, 0) / distance[-1]

        interpolator = interp1d(distance, points, kind="linear", axis=0)
        new_distances = np.linspace(0, 1, nlimit, endpoint=True)
        interpolated_points = interpolator(new_distances)

        R = np.asarray(interpolated_points[:, 0])
        Z = np.asarray(interpolated_points[:, 1])

        return R, Z

    def psi(self, R, Z):
        """
        Poloidal flux due to coils
        """
        psi_coils = 0.0
        for label, coil in self.coils:
            psi_coils += coil.psi(R, Z)
        return psi_coils

    def dpsi_dR(self, R, Z):
        dpsi_dR_coils = 0.0
        for label, coil in self.coils:
            dpsi_dR_coils += coil.dpsi_dR(R, Z)
        return dpsi_dR_coils

    def dpsi_dZ(self, R, Z):
        dpsi_dZ_coils = 0.0
        for label, coil in self.coils:
            dpsi_dZ_coils += coil.dpsi_dZ(R, Z)
        return dpsi_dZ_coils

    def d2psi_dR2(self, R, Z):
        d2psi_dR2_coils = 0.0
        for label, coil in self.coils:
            d2psi_dR2_coils += coil.d2psi_dR2(R, Z)
        return d2psi_dR2_coils
        
    def d2psi_dZ2(self, R, Z):
        d2psi_dZ2_coils = 0.0
        for label, coil in self.coils:
            d2psi_dZ2_coils += coil.d2psi_dZ2(R, Z)
        return d2psi_dZ2_coils
    
    def d2psi_dRdZ(self, R, Z):
        d2psi_dRdZ_coils = 0.0
        for label, coil in self.coils:
            d2psi_dRdZ_coils += coil.d2psi_dRdZ(R, Z)
        return d2psi_dRdZ_coils
    
    def createPsiGreens(self, R, Z):
        """
        An optimisation, which pre-computes the Greens functions
        and puts into arrays for each coil. This map can then be
        called at a later time, and quickly return the field
        """
        pgreen = {}
        for label, coil in self.coils:
            pgreen[label] = coil.createPsiGreens(R, Z)
        return pgreen

    def calcPsiFromGreens(self, pgreen):
        """
        Uses the object returned by createPsiGreens to quickly
        compute the plasma psi
        """
        psi_coils = 0.0
        for label, coil in self.coils:
            psi_coils += coil.calcPsiFromGreens(pgreen[label])

        return psi_coils

    def Br(self, R, Z):
        """
        Radial magnetic field at given points
        """
        Br = 0.0
        for label, coil in self.coils:
            Br += coil.Br(R, Z)

        return Br

    def Bz(self, R, Z):
        """
        Vertical magnetic field
        """
        Bz = 0.0
        for label, coil in self.coils:
            Bz += coil.Bz(R, Z)

        return Bz

    def controlBr(self, R, Z):
        """
        Returns a list of control responses for Br
        at the given (R,Z) location(s).
        """
        return torch.stack(
            [coil.controlBr(R, Z) for label, coil in self.coils if coil.control], 
            dim=0
            )

    def controlBz(self, R, Z):
        """
        Returns a list of control responses for Bz
        at the given (R,Z) location(s)
        """
        return torch.stack(
            [coil.controlBz(R, Z) for label, coil in self.coils if coil.control], 
            dim=0
            )

    def controlPsi(self, R, Z):
        """
        Returns a list of control responses for psi
        at the given (R,Z) location(s)
        """
        return torch.stack(
            [coil.controlPsi(R, Z) for label, coil in self.coils if coil.control],
            dim=0
            )

    def controlAdjust(self, current_change):
        """
        Add given currents to the controls.
        Given iterable must be the same length
        as the list returned by controlBr, controlBz
        """
        # Get list of coils being controlled
        controlcoils = [coil for label, coil in self.coils if coil.control]

        for coil, dI in zip(controlcoils, current_change):
            # Ensure that dI is a scalar
            coil.current += dI.item()

    def controlCurrents(self):
        """
        Return a list of coil currents for the coils being controlled
        """
        return [coil.current for label, coil in self.coils if coil.control]

    def setControlCurrents(self, currents):
        """
        Sets the currents in the coils being controlled.
        Input list must be of the same length as the list
        returned by controlCurrents
        """
        controlcoils = [coil for label, coil in self.coils if coil.control]
        for coil, current in zip(controlcoils, currents):
            coil.current = current

    def printCurrents(self):
        print("==========================")
        for label, coil in self.coils:
            print(label + " : " + str(coil))
        print("==========================")

    def takeMeasurements(self, eq=None):
        """
        Method calling the measure method of each sensor on the machine
        """
        for sensor in self.sensors:
            sensor.get_measure(self, eq)

    def printMeasurements(self, eq=None):
        """
        Method for calling the takeMeasurements method, then printing the results
        """
        print("==========================")
        self.takeMeasurements(eq=eq)
        for sensor in self.sensors:
            if sensor.name is not None:
                print(sensor.name + ' '+ str(sensor) + ", Measurement=" + str(
                    sensor.measurement))
            else:
                print(str(type(sensor)) + str(sensor) + " Measurement=" + str(
                    sensor.measurement))
        print("==========================")
        return

    def getForces(self, equilibrium=None):
        """
        Calculate forces on the coils, given the plasma equilibrium.
        If no plasma equilibrium given then the forces due to
        the coils alone will be calculated.

        Returns a dictionary of coil label -> force
        """

        if equilibrium is None:
            equilibrium = self

        forces = {}
        for label, coil in self.coils:
            forces[label] = coil.getForces(equilibrium)
        return forces

    def getCurrents(self):
        """
        Returns a dictionary of coil label -> current in Amps
        """
        currents = {}
        for label, coil in self.coils:
            currents[label] = coil.current
        return currents

    def plot(self, axis=None, show=True):
        """
        Plot the machine coils
        """
        for label, coil in self.coils:
            axis = coil.plot(axis=axis, show=False)
        if show:
            import matplotlib.pyplot as plt

            plt.show()
        return axis

class Wall:
    """
    Represents the wall of the device.
    Consists of an ordered list of (R,Z) points
    """

    def __init__(self, R: list, Z: list):
        assert len(R) == len(Z)
        self.R = R
        self.Z = Z

    def __repr__(self):
        return "Wall(R={R}, Z={Z})".format(R=self.R, Z=self.Z)

    def __eq__(self, other):
        return np.allclose(self.R, other.R) and np.allclose(self.Z, other.Z)

    def __ne__(self, other):
        return not self == other

def TestTokamak():
    """
    Create a simple tokamak
    """

    coils = [
        (
            "P1L",
            ShapedCoil([(0.95, -1.15), (0.95, -1.05), (1.05, -1.05), (1.05, -1.15)]),
        ),
        ("P1U", ShapedCoil([(0.95, 1.15), (0.95, 1.05), (1.05, 1.05), (1.05, 1.15)])),
        ("P2L", Coil(1.75, -0.6)),
        ("P2U", Coil(1.75, 0.6)),
    ]

    wall = Wall(
        [0.75, 0.75, 1.5, 1.8, 1.8, 1.5], [-0.85, 0.85, 0.85, 0.25, -0.25, -0.85]  # R
    )  # Z

    return Machine(coils, wall)
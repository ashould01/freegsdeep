from freegsdeep.freegs.gradshafranov import (
    Greens, GreensBr, GreensBz, dGreens_dR, dGreens_dZ, d2Greens_dR2,
    d2Greens_dZ2, d2Greens_dRdZ,
)
from freegsdeep.typing import *
from freegs.freegs import quadrature, polygons
import torch
import numbers
from shapely.geometry import Point, Polygon
import warnings

class AreaCurrentLimit:

    def __init__(self, current_density=3.5e9):
        self._current_density = current_density

    def __call__(self, coil):
        return abs(coil.current * coil.turns) / self._current_density

class Coil:
    """
    R, Z - Location of the coil
    current - current in each turn of the coil in Amps
    turns   - Number of turns
    control - enable or disable control system
    area    - Cross-section area in m^2

    The total toroidal current carried by the coil is current * turns
    """

    def __init__(
        self, R: Tensor, Z: Tensor, current=0.0, turns=1,
        control=True, area=AreaCurrentLimit()
    ):
        """
        R, Z - Location of the coil

        current - current in each turn of the coil in Amps
        turns   - Number of turns. Total coil current is current * turns
        control - enable or disable control system
        area    - Cross-section area in m^2

        To specify a current density limit, use:
            area = AreaCurrentLimit(current_density)
        where current_density is in A/m^2. The area of the coil will be recalculated
        as the coil current is changed.

        The most important effect of the area is on the coil self-force:
        The smaller the area the larger the hoop force for a given current.
        """
        self.R = R
        self.Z = Z

        self.current = current
        self.turns = turns
        self.control = control
        self.area = area
        self.mu0 = 4e-7 * torch.pi

    def psi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z)
        """
        return self.controlPsi(R, Z) * self.current

    def createPsiGreens(self, R, Z):
        """
        Calculate the Greens function at every point, and return
        array. This will be passed back to evaluate Psi in
        calcPsiFromGreens()
        """
        return self.controlPsi(R, Z)

    def calcPsiFromGreens(self, pgreen):
        """
        Calculate plasma psi from Greens functions and current
        """
        return self.current * pgreen

    def Br(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z)
        """
        return self.controlBr(R, Z) * self.current

    def Bz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z)
        """
        return self.controlBz(R, Z) * self.current
    
    def dpsi_dR(self, R, Z):
        return self.control_dpsi_dR(R, Z) * self.current
    
    def dpsi_dZ(self, R, Z):
        return self.control_dpsi_dZ(R, Z) * self.current
    
    def d2psi_dR2(self, R, Z):
        return self.control_d2psi_dR2(R, Z) * self.current
    
    def d2psi_dZ2(self, R, Z):
        return self.control_d2psi_dZ2(R, Z) * self.current
    
    def d2psi_dRdZ(self, R, Z):
        return self.control_d2psi_dRdZ(R, Z) * self.current
    
    def controlPsi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z) due to a unit current
        """
        return Greens(self.R, self.Z, R, Z) * self.turns

    def controlBr(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z) due to a unit current
        """
        return GreensBr(self.R, self.Z, R, Z) * self.turns

    def controlBz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z) due to a unit current
        """
        return GreensBz(self.R, self.Z, R, Z) * self.turns
    
    def control_dpsi_dR(self, R, Z):
        return dGreens_dR(self.R, self.Z, R, Z) * self.turns
    
    def control_dpsi_dZ(self, R, Z):
        return dGreens_dZ(self.R, self.Z, R, Z) * self.turns
    
    def control_d2psi_dR2(self, R, Z):
        return d2Greens_dR2(self.R, self.Z, R, Z) * self.turns
    
    def control_d2psi_dZ2(self, R, Z):
        return d2Greens_dZ2(self.R, self.Z, R, Z) * self.turns
    
    def control_d2psi_dRdZ(self, R, Z):
        return d2Greens_dRdZ(self.R, self.Z, R, Z) * self.turns

    def getForces(self, equilibrium):
        """
        Calculate forces on the coils in Newtons

        Returns an array of two elements: [ Fr, Fz ]


        Force on coil due to its own current:
            Lorentz self-forces on curved current loops
            Physics of Plasmas 1, 3425 (1998); https://doi.org/10.1063/1.870491
            David A. Garren and James Chen
        """
        current = self.current  # current per turn
        total_current = current * self.turns  # Total toroidal current

        # Calculate field at this coil due to all other coils
        # and plasma. Need to zero this coil's current
        self.current = 0.0
        Br = equilibrium.Br(self.R, self.Z)
        Bz = equilibrium.Bz(self.R, self.Z)
        self.current = current

        # Assume circular cross-section for hoop (self) force
        minor_radius = torch.sqrt(self.area / torch.pi)

        # Self inductance factor, depending on internal current
        # distribution. 0.5 for uniform current, 0 for surface current
        self_inductance = 0.5

        # Force per unit length.
        # In cgs units f = I^2/(c^2 * R) * (ln(8*R/a) - 1 + xi/2)
        # In SI units f = mu0 * I^2 / (4*pi*R) * (ln(8*R/a) - 1 + xi/2)
        self_fr = (self.mu0 * total_current ** 2 / (4.0 * torch.pi * self.R)) * (
            torch.log(8.0 * self.R / minor_radius) - 1 + self_inductance / 2.0
        )

        Ltor = 2 * torch.pi * self.R  # Length of coil
        return torch.array(
            [
                (total_current * Bz + self_fr)
                * Ltor,  # Jphi x Bz = Fr, self force always outwards
                -total_current * Br * Ltor,
            ]
        )  # Jphi x Br = - Fz

    def inShape(self, polygon):
        if polygon.contains(Point(self.R, self.Z)):
            return 1
        else:
            return 0

    def __repr__(self):
        return "Coil(R={0}, Z={1}, current={2:.1f}, turns={3}, control={4})".format(
            self.R, self.Z, self.current, self.turns, self.control
        )

    def __eq__(self, other):
        return (
            self.R == other.R
            and self.Z == other.Z
            and self.current == other.current
            and self.turns == other.turns
            and self.control == other.control
        )

    def __ne__(self, other):
        return not self == other

    def to_torch(self):
        """
        Helper method for writing output
        """
        return torch.tensor(
            (self.R, self.Z, self.current, self.turns, self.control), dtype=self.dtype
        )

    @classmethod
    def from_torch_tensor(cls, value):
        if value.dtype != cls.dtype:
            raise ValueError(
                "Can't create {this} from dtype: {got} (expected: {dtype})".format(
                    this=type(cls), got=value.dtype, dtype=cls.dtype
                )
            )
        return Coil(*value[()])

    @property
    def area(self):
        """
        The cross-section area of the coil in m^2
        """
        if isinstance(self._area, numbers.Number):
            if not self._area > 0:
                warnings.warn(f"Coil area {self._area:3.2f} <= 0")
            return self._area
        # Calculate using functor
        area = self._area(self)
        if not area > 0:
            warnings.warn(f"Coil area {area:3.2f} <= 0")
        return area

    @area.setter
    def area(self, area):
        self._area = area

    def plot(self, axis=None, show=False):
        """
        Plot the coil location, using axis if given

        The area of the coil is used to set the radius
        """
        minor_radius = torch.sqrt(self.area / torch.pi)

        import matplotlib.pyplot as plt

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)

        circle = plt.Circle((self.R, self.Z), minor_radius, color="gray")
        axis.add_artist(circle)
        return axis

class ShapedCoil(Coil):
    """
    Represents a coil with a specified shape

    public members
    --------------

    R, Z - Location of the point coil/Locations of coil filaments
    current - current in the coil(s) in Amps
    turns   - Number of turns if using point coils
    control - enable or disable control system
    area    - Cross-section area in m^2

    The total toroidal current carried by the coil block is current * turns
    """

    # A dtype for converting to Numpy array and storing in HDF5 files
    dtype = np.dtype(
        [
            (str("RZlen"), int),  # Length of the R and Z arrays
            (str("R"), "10f8"),  # Note: Up to 10 points
            (str("Z"), "10f8"),  # Note: Up to 10 points
            (str("current"), np.float64),
            (str("turns"), int),
            (str("control"), bool),
            (str("npoints"), int),
        ]
    )

    def __init__(self, shape, current=0.0, turns=1, control=True, npoints=6):
        """
        Inputs
        ------
        shape:
            Outline of the coil shape as a list of points ``[(r1,z1),
            (r2,z2), ...]``. Must have more than two points
        current:
            The current in the circuit. The total current is current * turns
        turns:
            Number of turns in point coil(s) block. Total block current is current * turns
        control:
            enable or disable control system
        npoints:
            Number of quadrature points per triangle. Valid choices: 1, 3, 6

        """
        assert len(shape) > 2

        # Find the geometric middle of the coil
        # The R,Z properties have accessor functions to handle modifications
        self._R_centre = sum(r for r, z in shape) / len(shape)
        self._Z_centre = sum(z for r, z in shape) / len(shape)

        self.current = current
        self.turns = turns
        self.control = control
        self._area = abs(polygons.area(shape))
        self.shape = shape

        # The quadrature points to be used
        self.npoints_per_triangle = npoints
        self._points = quadrature.polygon_quad(shape, n=npoints)

    def controlPsi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z) due to a unit current in the circuit
        """
        result = 0.0
        for R_fil, Z_fil, weight in self._points:
            result += Greens(R_fil, Z_fil, R, Z) * weight
        # Multiply by turns so that toroidal current is current * turns
        return result * self.turns

    def controlBr(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z) due to a unit current
        """
        result = 0.0
        for R_fil, Z_fil, weight in self._points:
            result += GreensBr(R_fil, Z_fil, R, Z) * weight
        return result * self.turns

    def controlBz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z) due to a unit current
        """
        result = 0.0
        for R_fil, Z_fil, weight in self._points:
            result += GreensBz(R_fil, Z_fil, R, Z) * weight
        return result * self.turns

    def inShape(self,polygon):
        Shaped_Coil = Polygon([shape for shape in self.shape])
        return (polygon.intersection(Shaped_Coil).area) / (self._area)

    def __repr__(self):
        return "ShapedCoil({0}, current={1:.1f}, turns={2}, control={3})".format(
            self.shape, self.current, self.turns, self.control
        )

    @property
    def R(self):
        """
        Major radius of the coil in m
        """
        return self._R_centre

    @R.setter
    def R(self, Rnew):
        # Need to shift all points
        Rshift = Rnew - self._R_centre
        self._points = [(r + Rshift, z, w) for r, z, w in self._points]
        self._R_centre = Rnew

    @property
    def Z(self):
        """
        Height of the coil in m
        """
        return self._Z_centre

    @Z.setter
    def Z(self, Znew):
        # Need to shift all points
        Zshift = Znew - self._Z_centre
        self._points = [(r, z + Zshift, w) for r, z, w in self._points]
        self._Z_centre = Znew

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, area):
        raise ValueError("Area of a ShapedCoil is fixed")

    def plot(self, axis=None, show=False):
        """
        Plot the coil shape, using axis if given
        """
        import matplotlib.pyplot as plt

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)

        r = [r for r, z in self.shape]
        z = [z for r, z in self.shape]
        axis.fill(r, z, color="gray")
        axis.plot(r, z, color="black")

        # Quadrature points
        # rquad = [r for r,z,w in self._points]
        # zquad = [z for r,z,w in self._points]
        # axis.plot(rquad, zquad, 'ro')

        return axis

    def to_numpy_array(self):
        """
        Helper method for writing output
        """
        RZlen = len(self.shape)
        R = np.zeros(10)
        Z = np.zeros(10)
        R[:RZlen] = [R for R, Z in self.shape]
        Z[:RZlen] = [Z for R, Z in self.shape]

        return np.array(
            (
                RZlen,
                R,
                Z,
                self.current,
                self.turns,
                self.control,
                self.npoints_per_triangle,
            ),
            dtype=self.dtype,
        )

    @classmethod
    def from_numpy_array(cls, value):
        if value.dtype != cls.dtype:
            raise ValueError(
                "Can't create {this} from dtype: {got} (expected: {dtype})".format(
                    this=type(cls), got=value.dtype, dtype=cls.dtype
                )
            )
        RZlen = value["RZlen"]
        R = value["R"][:RZlen]
        Z = value["Z"][:RZlen]
        current = value["current"]
        turns = value["turns"]
        control = value["control"]
        npoints = value["npoints"]

        return ShapedCoil(
            list(zip(R, Z)),
            current=current,
            turns=turns,
            control=control,
            npoints=npoints,
        )

    def __eq__(self, other):
        return (
            np.allclose(self.shape, other.shape)
            and self.current == other.current
            and self.turns == other.turns
            and self.control == other.control
            and self.npoints_per_triangle == other.npoints_per_triangle
        )

    def __ne__(self, other):
        return not self == other
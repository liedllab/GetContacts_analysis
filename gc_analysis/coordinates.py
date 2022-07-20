"""Module containing two classes for handeling 2D coordinates."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar
import numpy as np

T = TypeVar("T", bound="Coordinate")

class Coordinate(ABC):
    """Abstract class used for the classes Cartesian and Polar."""

    @property
    def values(self) -> np.array:
        """return coordinates as :class:`np.array`"""
        return self._values

    @abstractmethod
    def convert(self: T) -> T:
        """convert Coordinate to different Coordinate."""
        pass

@dataclass
class Cartesian(Coordinate):
    """Container class to store the coordinates of a 2D point in cartesian
    cordinates.

    The class contains one method :func:`convert`, which allows to conversion
    to polar coordinates and returns a :class:`Polar` object.

        :param x: int|float, the value of the x-component of the point.
        :param y: int|float, the value of the y-component of the point.
    """
    x: float
    y: float

    def __post_init__(self):
        self._values = np.array((self.x, self.y))

    def convert(self) -> Coordinate:
        """Method to convert from :class:`Cartesian` to :class:`Polar`."""
        r = np.sqrt(self.x**2+self.y**2)
        theta = np.arctan2(self.x,self.y)

        return Polar(theta=theta, r=r)

@dataclass
class Polar(Coordinate):
    """Container class to store the coordinates of a 2D point in polar
    cordinates.

    The class contains one method :func:`convert`, which allows to conversion
    to cartesian coordinates and returns a :class:`Cartesian` object.

        :param theta: float
                the angle in radians.
        :param r: float
                the distance from the origin to the point, the radius.
    """
    theta: float
    r: float

    def __post_init__(self):
        self.theta %= np.pi*2
        self._values =  np.array((self.theta, self.r))

    def convert(self) -> Coordinate:
        """Method to convert from :class:`Polar` to :class:`Cartesian`."""
        x = self.r*np.sin(self.theta)
        y = self.r*np.cos(self.theta)
        return Cartesian(x,y)

    def in_degrees(self) -> float:
        """Convenience method to convert :attr:`theta` from radians to degrees."""
        return self.theta*180/np.pi #, self.r 
    # check why a tuple was returned!


def bezier(starting_point: Coordinate, end_point: Coordinate,
            origin: Coordinate = Cartesian(0,0), n_points: int = 50):
    """Calculates a bezier curve connecting three points.

        :param starting_point: Coordinate
                start point of the bezier curve.
        :param end_point: Coordinate
                end point of the bezier curve.
        :param origin: Coordinate
                middle point for the bezier curve.
            default: Cartesian(0,0)
                sets zero as origin.
        :param n_points: int
                number of points on the bezier curve to calculate.
            default: 50

        :return: list[Cartesian]
                list of :class:`Cartesian` points.

        :raises: ValueError
                if not all arguments are Coordinate objects.
    """

    if not all(map(lambda x: isinstance(x, Coordinate), (starting_point,end_point,origin))):
        raise ValueError("Arguments need to be a Coordinate-Object")
    if isinstance(starting_point, Polar):
        starting_point = starting_point.convert()
    if isinstance(end_point, Polar):
        end_point = end_point.convert()
    if isinstance(origin, Polar):
        origin = origin.convert()

    bezier_curve = [Cartesian(
            *((1-t) * ( (1-t) * starting_point.values + t * origin.values ) + \
            t * ( (1-t) * origin.values + (t) * end_point.values ))
        ) for t in np.linspace(0,1, n_points)]

    return bezier_curve

if __name__ == "__main__":
    exit()

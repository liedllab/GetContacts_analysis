"""Tests for the coordinates classes."""

import pytest
import numpy as np
from GetContacts_analysis.coordinates import Cartesian, Polar

def test_cartesian_values() -> None:
    x, y = 0, 0
    cart = Cartesian(x, y)
    assert cart.values[0] == x
    assert cart.values[1] == y

def test_cartesian_convert() -> None:
    x, y = 1, 1
    cart = Cartesian(x, y)
    cart2 = Cartesian(x, y)

    assert np.allclose(cart.convert().convert().values, cart2.values)
    assert np.allclose(cart2.values, cart.convert().convert().values)

def test_polar_values() -> None:
    theta, r = 0, 0
    pol = Polar(theta, r)
    assert pol.values[0] == r 
    assert pol.values[1] == theta

def test_polar_convert() -> None:
    theta, r = 1, 1
    pol = Polar(theta, r)
    pol2 = Polar(theta, r)

    assert np.allclose(pol.convert().convert().values, pol2.values)
    assert np.allclose(pol2.values, pol.convert().convert().values)

def test_rad_to_deg() -> None:
    pol = Polar(2*np.pi, 1)
    deg = pol.in_degrees()
    assert deg == 0

def test_zero_point() -> None:
    cart = Cartesian(0, 0)
    pol = Polar(0, 0)

    assert np.allclose(cart.convert().values, pol.values)
    assert np.allclose(pol.values, cart.convert().values)
    assert np.allclose(pol.convert().values, cart.values)
    assert np.allclose(cart.values, pol.convert().values)


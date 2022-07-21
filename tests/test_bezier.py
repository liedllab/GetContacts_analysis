import pytest

from gc_analysis.coordinates import bezier, Polar, Cartesian

def test_bezier_len() -> None:
    n = 50

    cart = Cartesian(0, 0)
    bez = bezier(cart, cart, n_points = n)
    
    assert len(bez) == n

def test_bezier_types_cart() -> None:
    cart = Cartesian(0, 0)
    bez = bezier(cart, cart)
    
    assert all([isinstance(point, Cartesian) for point in bez])

def test_bezier_vertical() -> None:
    bez = bezier(Cartesian(1,0), Cartesian(-1,0))
    
    assert all([point.y == 0 for point in bez])

def test_bezier_horizontal() -> None:
    bez = bezier(Cartesian(0,1), Cartesian(0, -1))
    
    assert all([point.x == 0 for point in bez])

def test_bezier_types_polar() -> None:
    pol = Polar(0, 0)
    bez = bezier(pol, pol, pol)
    
    assert all([isinstance(point, Cartesian) for point in bez])

def test_wrong_input() -> None:
    with pytest.raises(ValueError):
        bezier((0,0), (1,1))

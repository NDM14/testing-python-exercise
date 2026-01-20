"""
Tests for functions in class SolveDiffusion2D
"""

from unittest.mock import MagicMock
from diffusion2d import SolveDiffusion2D
import pytest
import numpy



@pytest.mark.parametrize(("width", "height", "dx", "dy", "expected_dx", "expected_dy"), 
    [(11.0,12.0,0.5,0.3, 22, 40)]
)
def test_initialize_domain(width, height, dx, dy, expected_dx, expected_dy):
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(width, height, dx, dy)
    assert solver.nx == expected_dx
    assert solver.ny == expected_dy

@pytest.mark.parametrize(
    ("d", "T_cold", "T_hot", "expected_dt"), 
    [(5.0, 321.0, 723.0, 0.0005)]
)
def test_initialize_physical_parameters(d, T_cold, T_hot, expected_dt):
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    solver.w = 10.0
    solver.h = 10.0
    solver.dx = 0.1
    solver.dy = 0.1
    solver.nx = 100
    solver.ny = 100
    
    dt = solver.initialize_physical_parameters(d, T_cold, T_hot)
    
    solver.dt == pytest.approx(expected_dt, rel=1e-4)

    
@pytest.mark.parametrize(("T_cold", "T_hot"), 
                         [(302.0, 700.0)])
def test_set_initial_condition(T_cold, T_hot):
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    
    solver = SolveDiffusion2D()
    solver.T_cold = T_cold
    solver.T_hot = T_hot
    solver.w = 10.0
    solver.h = 10.0
    solver.dx = 0.1
    solver.dy = 0.1
    solver.nx = 100
    solver.ny = 100
    
    u = solver.set_initial_condition()
    assert u.shape == (solver.nx, solver.ny)
    assert numpy.all(u >= T_cold)
    assert numpy.all(u <= T_hot)
    assert u[50, 50] == T_hot

def test_time_step():
    assert True
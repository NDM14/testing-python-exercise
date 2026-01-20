"""
Tests for functionality checks in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import pytest
import numpy

@pytest.fixture
def solver():
    solver =  SolveDiffusion2D()
    solver.initialize_domain(10.0, 10.0, 0.1, 0.1)
    return solver

@pytest.mark.parametrize(
    ("d", "T_cold", "T_hot", "expected_dt"), 
    [(5.0, 321.0, 723.0, 0.0005)]
)
def test_initialize_physical_parameters(solver, d, T_cold, T_hot, expected_dt):
    """
    Checks function SolveDiffusion2D.initialize_domain
    """

    solver.initialize_physical_parameters(d, T_cold, T_hot)
    solver.dt == pytest.approx(expected_dt, rel=1e-4)


@pytest.mark.parametrize(("T_cold", "T_hot", "radius"), 
                         [(302.0, 700.0, 2.0)])
def test_set_initial_condition(solver, T_cold, T_hot, radius):
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver.initialize_physical_parameters(4.0, T_cold, T_hot)
    u = solver.set_initial_condition()
    
    cx = solver.w / radius
    cy = solver.h / radius
    
    assert u.shape == (solver.nx, solver.ny)

    assert numpy.all(u >= T_cold)
    assert numpy.all(u <= T_hot)
    assert u[int(cx/solver.dx), int(cy/solver.dy)] == T_hot

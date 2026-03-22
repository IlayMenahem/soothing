from pathlib import Path
from typing import Sequence

from sympy import Poly, Rational, Symbol, cos, sin, expand_trig

_SYSTEMS_DIR = Path(__file__).resolve().parent.parent / "systems"


def load_mathematica_system(filename: str) -> str:
    """Load raw Mathematica-style system text from ``systems/<filename>`` (repo root)."""
    path = _SYSTEMS_DIR / filename
    return path.read_text(encoding="utf-8")


def generate_robot_system(
    n_links: int,
    link_lengths: Sequence[Rational],
    target_pos: tuple[Rational, Rational] = (Rational(1, 1), Rational(1, 1)),
) -> tuple[list[Symbol], list[Poly]]:
    """
    Generates polynomial equations for an n-link planar robot arm.

    Args:
        n_links (int): Number of arm segments.
        link_lengths (Sequence[Rational]): Length of each segment (default: all 1.0).
        target_pos (tuple[Rational, Rational]): The (x, y) coordinate the end-effector must reach.

    Returns:
        variables (list[Symbol]): The symbols used (c1, s1, c2, s2, ...).
        equations (list[Poly]): The polynomial system F(x) = 0.
    """
    # Define variables: c_i = cos(theta_i), s_i = sin(theta_i)
    # We use relative angles (angle of link i relative to link i-1)
    thetas = sp.symbols(f"theta_:{n_links}")
    variables: list[Symbol] = []

    # We need algebraic variables for the solver (c, s) instead of trig functions
    algebraic_vars: dict[Symbol, tuple[Symbol, Symbol]] = {}
    for i in range(n_links):
        c = Symbol(f"c{i + 1}")
        s = Symbol(f"s{i + 1}")
        variables.extend([c, s])
        algebraic_vars[thetas[i]] = (c, s)

    # 1. Forward Kinematics (Trig form)
    # x = L1*cos(th1) + L2*cos(th1+th2) + ...
    x_expr = 0
    y_expr = 0
    current_angle = 0

    for i in range(n_links):
        current_angle += thetas[i]
        x_expr += link_lengths[i] * cos(current_angle)
        y_expr += link_lengths[i] * sin(current_angle)

    # 2. Convert Trig to Polynomials
    # expand_trig breaks cos(a+b) into cos(a)cos(b) - sin(a)sin(b)
    x_poly = expand_trig(x_expr)
    y_poly = expand_trig(y_expr)

    # Substitute cos(theta) -> c, sin(theta) -> s
    # We must substitute recursively or carefully to catch all terms
    for theta_sym, (c_sym, s_sym) in algebraic_vars.items():
        x_poly = x_poly.subs(cos(theta_sym), c_sym).subs(sin(theta_sym), s_sym)
        y_poly = y_poly.subs(cos(theta_sym), c_sym).subs(sin(theta_sym), s_sym)

    # 3. Assemble Equations
    system = []

    # Constraint A: Reach the target (Kinematics)
    system.append(x_poly - target_pos[0])
    system.append(y_poly - target_pos[1])

    # Constraint B: Pythagorean Identity (Geometry)
    # c^2 + s^2 - 1 = 0 for every joint
    for i in range(n_links):
        c, s = variables[2 * i], variables[2 * i + 1]
        system.append(c**2 + s**2 - 1)

    return variables, system


def generate_molecule_system(
    n_atoms: int, bond_length: Rational = Rational(1, 1)
) -> tuple[list[Symbol], list[Poly]]:
    """
    Generates polynomial equations for a cyclic molecule (ring) of n atoms.

    Args:
        n_atoms (int): Number of atoms in the ring.
        bond_length (Rational): Target distance between adjacent atoms.

    Returns:
        variables (list[Symbol]): The free variables (coordinates).
        equations (list[Poly]): The polynomial system.
    """
    # Create coordinates for all atoms: x0, y0, x1, y1, ...
    coords = []
    for i in range(n_atoms):
        coords.append((Symbol(f"x{i}"), Symbol(f"y{i}")))

    equations = []

    # 1. Distance Constraints (The "Bonds")
    # (x_i - x_j)^2 + (y_i - y_j)^2 - L^2 = 0
    for i in range(n_atoms):
        # Connect atom i to atom i+1 (wrapping around to 0 at the end)
        curr_atom = coords[i]
        next_atom = coords[(i + 1) % n_atoms]

        dist_sq = (curr_atom[0] - next_atom[0]) ** 2 + (
            curr_atom[1] - next_atom[1]
        ) ** 2
        equations.append(dist_sq - bond_length**2)

    # 2. Remove Rigid Body Motion (Pinning)
    # Without this, the solution space is 3 dimensions higher (x-trans, y-trans, rotation)

    # Fix Atom 0 to (0,0)
    # We add these as equations: x0 = 0, y0 = 0
    equations.append(coords[0][0])
    equations.append(coords[0][1])

    # Fix Atom 1 to lie on X-axis (removes rotation)
    # y1 = 0
    equations.append(coords[1][1])

    # Flatten variables list for output
    all_vars = [v for sub in coords for v in sub]

    return all_vars, equations


def generate_3d_graph_system(
    n_atoms: int,
    edges: list[tuple[int, int]],
    bond_lengths: dict[tuple[int, int], Rational] = {},
) -> tuple[list[Symbol], list[Poly]]:
    """
    Generates a 3D polynomial system for an arbitrary molecular graph.

    Args:
        n_atoms (int): Total number of atoms.
        edges (list of tuples): Connectivity, e.g., [(0,1), (1,2), (0,2)]
        bond_lengths (dict): Map of edge -> length. Defaults to 1.0 for all.

    Returns:
        variables (list): All free coordinate variables.
        equations (list): The polynomial system F(x) = 0.
    """
    # 1. Define Variables (x_i, y_i, z_i)
    # We store them in a structured list
    coords = []
    variables = []

    for i in range(n_atoms):
        x = Symbol(f"x{i}")
        y = Symbol(f"y{i}")
        z = Symbol(f"z{i}")
        coords.append((x, y, z))
        variables.extend([x, y, z])

    equations = []

    # 2. Distance Constraints (The Topology)
    # (xi - xj)^2 + (yi - yj)^2 + (zi - zj)^2 - L^2 = 0
    for i, j in edges:
        p1 = coords[i]
        p2 = coords[j]

        # Sort key to ensure we match the bond_lengths dictionary if order swapped
        key = tuple(sorted((i, j)))
        L = bond_lengths.get(key, Rational(1, 1))

        dist_sq = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
        equations.append(dist_sq - L**2)

    # 3. Remove Rigid Body Motion (Pinning)
    equations.append(coords[0][0])
    equations.append(coords[0][1])
    equations.append(coords[0][2])
    equations.append(coords[1][1])
    equations.append(coords[2][2])

    return variables, equations


if __name__ == "__main__":
    vars_robot, eqs_robot = generate_robot_system(n_links=5)
    print(f"Number of Variables: {len(vars_robot)}")
    print(f"Number of Equations: {len(eqs_robot)}")
    for eq in eqs_robot:
        print(eq)

    vars_molecule, eqs_molecule = generate_molecule_system(n_atoms=3)
    print(f"Number of Variables: {len(vars_molecule)}")
    print(f"Number of Equations: {len(eqs_molecule)}")
    for eq in eqs_molecule:
        print(eq)

    import numpy as np

    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]
    bond_lengths = {edge: np.random.rand() for edge in edges}
    vars_3d_graph, eqs_3d_graph = generate_3d_graph_system(5, edges, bond_lengths)
    print(f"Number of Variables: {len(vars_3d_graph)}")
    print(f"Number of Equations: {len(eqs_3d_graph)}")
    for eq in eqs_3d_graph:
        print(eq)

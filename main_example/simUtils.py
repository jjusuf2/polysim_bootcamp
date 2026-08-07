# MIT License

# Copyright (c) 2019 Massachusetts Institute of Technology 

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# The following is adapted from polychrom starting_conformations.create_constrained_random_walk()
# See https://github.com/open2c/polychrom/

import numpy as np
from typing import Callable, Tuple
from polychrom.starting_conformations import _random_points_sphere

def create_constrained_random_walk(
    N: int,
    constraint_f: Callable[[Tuple[float, float, float],Tuple[float, float, float]], bool],
    starting_point=(0, 0, 0),
    step_size=1.0,
    polar_fixed=None,
) -> bool:
    """
    Adapated from starting_conformations.create_constrained_random_walk()

    Creates a constrained freely joined chain of length N with step step_size.
    Each step of a random walk is tested with the constraint function and is
    rejected if the tried step lies outside of the constraint.
    This function is much less efficient than create_random_walk().

    Parameters
    ----------
    N : int
        The number of steps
    constraint_f : callable
        The constraint function.
        Must accept two tuples of 3 floats with the polymer start pt and the tentative position of a particle,
        and return True if the new position is accepted and False is it is forbidden.
    starting_point : a tuple of (float, float, float)
        The starting point of a random walk.
    step_size: float
        The size of a step of the random walk.
    polar_fixed: float, optional
        If specified, the random_walk is forced to fix the polar angle at polar_fixed.
        The implementation includes backtracking if no steps were possible, but if seriously overconstrained,
        the algorithm can get stuck in an infinite loop.
    """

    i = 1
    j = N
    n_reps = 0
    out = np.full((N, 3), np.nan)
    out[0] = starting_point

    while i < N:
        if j == N:
            theta, u = _random_points_sphere(N).T
            if polar_fixed is not None:
                # fixes the polar angle in uniform distribution on the sphere
                u = np.cos(polar_fixed) * np.ones(len(u))
            dx = step_size * np.sqrt(1.0 - u * u) * np.cos(theta)
            dy = step_size * np.sqrt(1.0 - u * u) * np.sin(theta)
            dz = step_size * u
            d = np.vstack([dx, dy, dz]).T
            n_reps += 1
            j = 0
        if polar_fixed is not None and i > 1:
            # Rotate the generated point to a coordinate system with the previous
            # displacement pointing along the z-axis

            past_displacement = out[i - 1] - out[i - 2]

            vec_to_rot = d[j]
            rot_axis = np.cross(past_displacement, np.array([0, 0, 1]))
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            rot_angle = -np.arccos(np.dot(past_displacement, np.array([0, 0, 1])) / np.linalg.norm(past_displacement))
            # Rotating with the Rodriques' rotation formula
            # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
            next_displacement = (
                vec_to_rot * np.cos(rot_angle)
                + np.cross(rot_axis, vec_to_rot) * np.sin(rot_angle)
                + rot_axis * np.dot(rot_axis, vec_to_rot) * (1 - np.cos(rot_angle))
            )
            # Add the rotated point
            new_p = out[i - 1] + next_displacement
        else:
            new_p = out[i - 1] + d[j]

        if constraint_f(starting_point,new_p):
            out[i] = new_p
            i += 1

        j += 1
        if n_reps > 2:
            if i != 1:
                # Backtracking if no moves are possible
                i -= 1
                n_reps = 0
            else:
                # If the first point is reached, there is nothing to do
                print("error: start, newpt:",starting_point, new_p)
                raise RuntimeError(
                    "The walk-generation cannot take the first step! Have another look at the"
                    " constraints and initial condition"
                )

    return out


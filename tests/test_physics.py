# Copyright 2026 Raúl Rodolfo Flores Mena
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Density and cell descriptors.

The density anchor is the experimental TMPA-An structure (CCDC 1493451):
Z = 4, M = 632.73 g/mol, V = 3364.2 A^3 must give the published 1.249 g/cm3.
Density is not cosmetic here -- it is one of the two coordinates the generator
consumes and one of the seven the surrogate is trained on, so a silent factor
error would misplace the entire search.
"""

import unittest

import numpy as np
from ase import Atoms

from cryal.utils import cell_params_dict, compute_density

TMPA_AN_MW = 632.73


def box(a, b, c, alpha=90.0, beta=90.0, gamma=90.0):
    atoms = Atoms("C", positions=[[0, 0, 0]], pbc=True)
    atoms.set_cell([a, b, c, alpha, beta, gamma])
    return atoms


class TestDensity(unittest.TestCase):

    def test_reproduces_the_experimental_density(self):
        atoms = box(18.493, 11.044, 16.785, beta=101.077)
        rho = compute_density(atoms, Z=4, mol_weight=TMPA_AN_MW)
        self.assertAlmostEqual(rho, 1.249, places=3)

    def test_scales_inversely_with_volume(self):
        small = compute_density(box(10, 10, 10), 4, TMPA_AN_MW)
        big = compute_density(box(10, 10, 20), 4, TMPA_AN_MW)
        self.assertAlmostEqual(small, 2 * big, places=9)

    def test_scales_linearly_with_Z(self):
        one = compute_density(box(10, 10, 10), 1, TMPA_AN_MW)
        four = compute_density(box(10, 10, 10), 4, TMPA_AN_MW)
        self.assertAlmostEqual(four, 4 * one, places=9)


class TestCellParams(unittest.TestCase):

    def test_round_trips_a_monoclinic_cell(self):
        cp = cell_params_dict(box(18.493, 11.044, 16.785, beta=101.077))
        self.assertAlmostEqual(cp["a"], 18.493, places=3)
        self.assertAlmostEqual(cp["b"], 11.044, places=3)
        self.assertAlmostEqual(cp["c"], 16.785, places=3)
        self.assertAlmostEqual(cp["alpha"], 90.0, places=6)
        self.assertAlmostEqual(cp["beta"], 101.077, places=3)
        self.assertAlmostEqual(cp["gamma"], 90.0, places=6)

    def test_volume_matches_the_experimental_cell(self):
        cp = cell_params_dict(box(18.493, 11.044, 16.785, beta=101.077))
        self.assertAlmostEqual(cp["volume"], 3364.2, delta=0.5)

    def test_orthorhombic_volume_is_the_product_of_the_axes(self):
        cp = cell_params_dict(box(7.0, 9.0, 11.0))
        self.assertAlmostEqual(cp["volume"], 7.0 * 9.0 * 11.0, places=6)

    def test_returns_every_documented_key(self):
        cp = cell_params_dict(box(10, 10, 10))
        self.assertEqual(set(cp), {"a", "b", "c", "alpha", "beta", "gamma",
                                   "volume"})


if __name__ == "__main__":
    unittest.main()

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

"""The integrity gate that runs before every relaxation.

This gate exists because of a real failure: the CH-pi optimizer used to return
structures with 0.1-0.37 A atomic overlaps, which made MACE-OFF produce forces
of order 1e5 eV/A and LAMMPS die with misleading errors ("lost atoms", "box has
tilted too far"). The diagnosis took a long time precisely because the symptom
pointed at the relaxation rather than at the geometry handed to it.

The important property is the periodic one: an overlap between an atom and the
periodic image of another must be caught. That is why the check reads
NeighborList offsets instead of minimum-image distances.
"""

import unittest

import numpy as np
from ase import Atoms

from cryal.utils import check_bond_integrity, check_cell_axes


class TestBondIntegrity(unittest.TestCase):

    def test_accepts_a_sane_molecule(self):
        # Ethane-like C-C at 1.54 A in a roomy cell.
        atoms = Atoms("C2", positions=[[5.0, 5.0, 5.0], [6.54, 5.0, 5.0]],
                      cell=np.eye(3) * 15.0, pbc=True)
        self.assertTrue(check_bond_integrity(atoms))

    def test_rejects_atoms_on_top_of_each_other(self):
        atoms = Atoms("C2", positions=[[5.0, 5.0, 5.0], [5.3, 5.0, 5.0]],
                      cell=np.eye(3) * 15.0, pbc=True)
        self.assertFalse(check_bond_integrity(atoms))

    def test_rejects_an_overlap_through_the_periodic_boundary(self):
        # The two atoms are 9.7 A apart inside the box but only 0.3 A apart
        # through the boundary. A check that ignored periodicity would pass.
        atoms = Atoms("C2", positions=[[0.15, 5.0, 5.0], [9.85, 5.0, 5.0]],
                      cell=np.eye(3) * 10.0, pbc=True)
        self.assertFalse(check_bond_integrity(atoms))

    def test_threshold_is_honoured(self):
        atoms = Atoms("C2", positions=[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]],
                      cell=np.eye(3) * 15.0, pbc=True)
        self.assertTrue(check_bond_integrity(atoms, min_bond=0.8))
        self.assertFalse(check_bond_integrity(atoms, min_bond=1.2))

    def test_survives_a_single_atom(self):
        atoms = Atoms("C", positions=[[5.0, 5.0, 5.0]],
                      cell=np.eye(3) * 15.0, pbc=True)
        self.assertTrue(check_bond_integrity(atoms))


class TestCellAxes(unittest.TestCase):

    def test_accepts_a_cell_above_the_minimum(self):
        atoms = Atoms("C", positions=[[0, 0, 0]], cell=np.eye(3) * 12.0, pbc=True)
        self.assertTrue(check_cell_axes(atoms, 9.0))

    def test_rejects_a_short_axis(self):
        atoms = Atoms("C", positions=[[0, 0, 0]],
                      cell=np.diag([12.0, 4.0, 12.0]), pbc=True)
        self.assertFalse(check_cell_axes(atoms, 9.0))

    def test_uses_lengths_not_components(self):
        # A triclinic cell whose b vector has a small y component but a
        # perfectly acceptable length.
        cell = np.array([[12.0, 0.0, 0.0],
                         [9.0, 7.0, 0.0],
                         [0.0, 0.0, 12.0]])
        atoms = Atoms("C", positions=[[0, 0, 0]], cell=cell, pbc=True)
        self.assertAlmostEqual(np.linalg.norm(cell[1]), 11.4018, places=3)
        self.assertTrue(check_cell_axes(atoms, 9.0))


if __name__ == "__main__":
    unittest.main()

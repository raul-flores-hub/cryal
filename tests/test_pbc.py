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

"""Rebuilding molecules across the periodic boundary.

Getting this wrong is not hypothetical: an early version of the analysis code
applied the lattice offset of each atom relative to the molecule's first atom
instead of accumulating it along the bond graph, which displaced atoms by
3-7 A and quietly corrupted every geometric descriptor computed from it.

The last test in this file records a limitation that is still present in
unwrap_molecule and is marked as an expected failure rather than asserted as
correct behaviour: it uses the minimum image convention against a single
reference atom, so it cannot rebuild a molecule longer than half the cell.
That is safe today only because its one caller, the CH-pi optimizer, is off by
design in production runs. If the function is ever fixed, this test turns into
an unexpected success and says so.
"""

import unittest

import numpy as np

from cryal.utils import mic_vector, unwrap_molecule


def cubic(length):
    return np.eye(3) * length


class TestUnwrapMolecule(unittest.TestCase):

    def test_contiguous_molecule_is_returned_unchanged(self):
        cell = cubic(20.0)
        pos = np.array([[5.0, 5.0, 5.0], [6.5, 5.0, 5.0], [8.0, 5.0, 5.0]])
        out = unwrap_molecule(pos, np.arange(3), cell)
        np.testing.assert_allclose(out, pos, atol=1e-12)

    def test_molecule_split_across_the_boundary_is_rejoined(self):
        # Three atoms 1.5 A apart, wrapped so the last one sits on the far side.
        cell = cubic(10.0)
        pos = np.array([[9.0, 5.0, 5.0], [0.5, 5.0, 5.0], [2.0, 5.0, 5.0]])
        out = unwrap_molecule(pos, np.arange(3), cell)
        bonds = np.linalg.norm(np.diff(out, axis=0), axis=1)
        np.testing.assert_allclose(bonds, [1.5, 1.5], atol=1e-9)

    def test_first_atom_is_the_reference_and_does_not_move(self):
        cell = cubic(10.0)
        pos = np.array([[9.0, 5.0, 5.0], [0.5, 5.0, 5.0]])
        out = unwrap_molecule(pos, np.arange(2), cell)
        np.testing.assert_allclose(out[0], pos[0], atol=1e-12)

    def test_works_in_a_skewed_triclinic_cell(self):
        # Skewed cells are where naive wrapping breaks first.
        cell = np.array([[10.0, 0.0, 0.0],
                         [3.0, 9.0, 0.0],
                         [2.0, 1.5, 8.0]])
        frac = np.array([[0.95, 0.5, 0.5], [0.02, 0.5, 0.5]])
        pos = frac @ cell
        out = unwrap_molecule(pos, np.arange(2), cell)
        d = np.linalg.norm(out[1] - out[0])
        # 0.07 of a cell vector along a, the short way round.
        self.assertAlmostEqual(d, 0.07 * 10.0, places=9)

    def test_indices_may_be_a_subset_of_the_frame(self):
        cell = cubic(10.0)
        pos = np.array([[0.0, 0.0, 0.0],      # another molecule
                        [9.0, 5.0, 5.0],
                        [0.5, 5.0, 5.0]])
        out = unwrap_molecule(pos, np.array([1, 2]), cell)
        self.assertEqual(out.shape, (2, 3))
        self.assertAlmostEqual(np.linalg.norm(out[1] - out[0]), 1.5, places=9)

    @unittest.expectedFailure
    def test_molecule_longer_than_half_the_cell(self):
        """Known limitation -- see the module docstring.

        A chain spanning 8 A in a 10 A cell: every atom more than half a cell
        away from the reference is folded back by the minimum image
        convention, so the chain collapses. Rebuilding this correctly needs
        the offset accumulated along the bond graph, not a single reference.
        """
        cell = cubic(10.0)
        pos = np.array([[[x, 5.0, 5.0] for x in (1.0, 3.0, 5.0, 7.0, 9.0)]][0])
        out = unwrap_molecule(pos, np.arange(5), cell)
        bonds = np.linalg.norm(np.diff(out, axis=0), axis=1)
        np.testing.assert_allclose(bonds, [2.0] * 4, atol=1e-9)


class TestMicVector(unittest.TestCase):

    def test_takes_the_short_way_round(self):
        cell = cubic(10.0)
        cell_T = cell.T
        v = mic_vector(np.array([9.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                       cell_T, np.linalg.inv(cell_T))
        np.testing.assert_allclose(v, [2.0, 0.0, 0.0], atol=1e-9)

    def test_is_antisymmetric(self):
        cell = cubic(10.0)
        cell_T = cell.T
        inv = np.linalg.inv(cell_T)
        a, b = np.array([1.0, 2.0, 3.0]), np.array([7.0, 8.0, 1.0])
        np.testing.assert_allclose(mic_vector(a, b, cell_T, inv),
                                   -mic_vector(b, a, cell_T, inv), atol=1e-9)


if __name__ == "__main__":
    unittest.main()

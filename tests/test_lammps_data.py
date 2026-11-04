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


"""The LAMMPS data file must describe every cell as a triclinic box.

The relaxation protocol starts with `fix npt ... tri`, which LAMMPS accepts
only on a triclinic box. ASE writes no tilt factors for a cell with right
angles, so before this check every orthorhombic candidate stopped with an
error at the start of its relaxation and was recorded as a failure -- all 38
of them in the search reported in the accompanying article.
"""

import os
import tempfile
import unittest

from ase import Atoms

from cryal.lammps_runner import _write_lammps_data


class TestRightAngleCellsReachLammpsAsTriclinic(unittest.TestCase):

    def _written(self, cellpar):
        atoms = Atoms("CH", positions=[[0, 0, 0], [1.1, 0, 0]],
                      cell=cellpar, pbc=True)
        fd, path = tempfile.mkstemp(suffix=".data")
        os.close(fd)
        try:
            _write_lammps_data(atoms, path, ["C", "H"])
            with open(path) as f:
                return f.read()
        finally:
            os.unlink(path)

    def test_tilt_factors_written_for_right_angles(self):
        self.assertIn("xy xz yz", self._written([8.0, 8.0, 11.0, 90, 90, 90]))

    def test_monoclinic_cell_unchanged(self):
        # A cell that already has a tilt was always written as triclinic.
        self.assertIn("xy xz yz", self._written([8.0, 9.0, 11.0, 90, 101, 90]))


if __name__ == "__main__":
    unittest.main()

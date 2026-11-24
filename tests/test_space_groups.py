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

"""Space groups outside the default eight.

The lattice sampler used to classify only the eight default groups, so any
other orthorhombic group -- Cmce (64), where rubrene crystallises -- was drawn
as a triclinic cell and every PyXtal attempt failed silently. Then it stopped
at 74; the tetragonal, trigonal, hexagonal and cubic cells cover the rest.
"""

import logging
import os
import random
import tempfile
import unittest

import numpy as np
from ase import Atoms

from cryal.backends.lammps_mace import _write_lammps_data
from cryal.config import Config, load_config
from cryal.gp_model import anisotropy
from cryal.structure_gen import generate_structures, random_lattice_for_sg

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestLatticeByCrystalSystem(unittest.TestCase):

    def setUp(self):
        random.seed(0)
        self.cfg = Config()

    def _cell(self, sg):
        return random_lattice_for_sg(sg, 2700.0, self.cfg)

    def test_every_orthorhombic_group_gets_right_angles(self):
        for sg in (16, 19, 33, 61, 64, 74):
            with self.subTest(sg=sg):
                self.assertEqual(self._cell(sg)[3:], (90.0, 90.0, 90.0))

    def test_every_monoclinic_group_keeps_alpha_and_gamma_right(self):
        for sg in (3, 4, 9, 12, 14, 15):
            with self.subTest(sg=sg):
                alpha, beta, gamma = self._cell(sg)[3:]
                self.assertEqual((alpha, gamma), (90.0, 90.0))
                self.assertGreaterEqual(beta, self.cfg.beta_min)

    def test_triclinic_groups_get_free_angles(self):
        for sg in (1, 2):
            with self.subTest(sg=sg):
                self.assertNotEqual(self._cell(sg)[3:], (90.0, 90.0, 90.0))

    def test_tetragonal_groups_get_a_equal_b_and_right_angles(self):
        for sg in (75, 88, 114, 142):
            with self.subTest(sg=sg):
                a, b, c, *angles = self._cell(sg)
                self.assertEqual(a, b)
                self.assertEqual(tuple(angles), (90.0, 90.0, 90.0))
                self.assertAlmostEqual(a * b * c, 2700.0, places=6)

    def test_trigonal_and_hexagonal_groups_get_gamma_120(self):
        # 146, 148, 167: rhombohedral, in the hexagonal setting
        for sg in (143, 146, 148, 167, 173, 194):
            with self.subTest(sg=sg):
                a, b, c, *angles = self._cell(sg)
                self.assertEqual(a, b)
                self.assertEqual(tuple(angles), (90.0, 90.0, 120.0))
                self.assertAlmostEqual(a * a * c * np.sqrt(3) / 2, 2700.0, places=6)

    def test_cubic_groups_get_three_equal_axes(self):
        for sg in (195, 205, 230):
            with self.subTest(sg=sg):
                a, b, c, *angles = self._cell(sg)
                self.assertEqual((a, b, tuple(angles)), (c, c, (90.0, 90.0, 90.0)))
                self.assertAlmostEqual(a ** 3, 2700.0, places=6)

    def test_fallback_keeps_hexagonal_gamma(self):
        self.cfg.min_cell_axis = 1e3     # no draw passes, so the fallback runs
        self.assertEqual(self._cell(185)[3:], (90.0, 90.0, 120.0))
        self.assertEqual(self._cell(88)[3:], (90.0, 90.0, 90.0))


class TestAnisotropyWithEqualAxes(unittest.TestCase):

    def setUp(self):
        random.seed(3)
        self.cfg = Config()
        self.cfg.min_cell_axis = 3.0

    def test_unique_axis_ratio_hits_the_target(self):
        for sg in (88, 167):
            longest = set()
            for _ in range(100):
                a, b, c = random_lattice_for_sg(sg, 1150.0, self.cfg,
                                                anisotropy_target=3.2)[:3]
                with self.subTest(sg=sg):
                    self.assertEqual(a, b)
                    self.assertGreaterEqual(anisotropy(a, b, c), 0.9 * 3.2 - 1e-9)
                    self.assertLessEqual(anisotropy(a, b, c), 1.1 * 3.2 + 1e-9)
                longest.add(c > a)
            # c is the long axis in some cells and the short one in others
            self.assertEqual(longest, {True, False})

    def test_cubic_ignores_the_target(self):
        a, b, c = random_lattice_for_sg(205, 1150.0, self.cfg,
                                        anisotropy_target=3.2)[:3]
        self.assertEqual(anisotropy(a, b, c), 1.0)


class TestConfigRejectsUnsupportedGroups(unittest.TestCase):

    def _load(self, groups, weights):
        table = ", ".join(f"{g} = {w}" for g, w in zip(groups.split(), weights.split()))
        fd, path = tempfile.mkstemp(suffix="_cryal.toml")
        with os.fdopen(fd, "w") as f:
            f.write('[general]\nmolecule = "examples/benzene.xyz"\nZ = 4\n'
                    f"[space_groups]\nweights = {{ {table} }}\n")
        try:
            return load_config(path)
        finally:
            os.unlink(path)

    def test_orthorhombic_group_outside_the_default_list_loads(self):
        self.assertIn(64, self._load("14 2 64", "1 1 1").space_groups)

    def test_groups_above_74_load(self):
        self.assertEqual(self._load("14 88 167 230", "1 1 1 1").space_groups,
                         [14, 88, 167, 230])

    def test_numbers_outside_1_230_are_refused(self):
        for groups in ("14 231", "0 14"):
            with self.subTest(groups=groups):
                with self.assertRaisesRegex(ValueError, "not a space group"):
                    self._load(groups, "1 1")


class TestRightAngleCellsReachLammpsAsTriclinic(unittest.TestCase):
    """`fix npt ... tri` needs a triclinic box; an orthogonal cell written
    without tilt factors is read as an orthogonal one and the stage dies."""

    def test_tilt_factors_written_for_right_angles(self):
        atoms = Atoms("CH", positions=[[0, 0, 0], [1.1, 0, 0]],
                      cell=[8.0, 8.0, 11.0, 90, 90, 90], pbc=True)
        fd, path = tempfile.mkstemp(suffix=".data")
        os.close(fd)
        try:
            _write_lammps_data(atoms, path, ["C", "H"])
            with open(path) as f:
                self.assertIn("xy xz yz", f.read())
        finally:
            os.unlink(path)


class TestGroupThatNeverFitsIsNamed(unittest.TestCase):

    def test_warning_names_the_group_and_pyxtal_reason(self):
        # Ia-3d (230): general position 96, smallest special position 16, so
        # Z = 4 fits no Wyckoff combination and PyXtal refuses every draw.
        # (It reports the molecule's symmetry first, Symm_CompatibilityError,
        # before it gets to the count, Comp_CompatibilityError.)
        cfg = Config()
        cfg.Z = 4
        cfg.mol_weight = 78.11
        cfg.target_volume_min, cfg.target_volume_max = 400.0, 800.0
        cfg.space_groups, cfg.sg_weights = [230], [1.0]
        logger = logging.getLogger("test_space_groups.never_fits")
        random.seed(0)
        with self.assertLogs(logger, level="WARNING") as logs:
            out = generate_structures(os.path.join(_ROOT, "examples", "benzene.xyz"),
                                      cfg, 1, logger=logger)
        self.assertEqual(out, [])
        self.assertTrue(any("SG 230: 0/" in m and "CompatibilityError" in m
                            for m in logs.output), logs.output)


if __name__ == "__main__":
    unittest.main()

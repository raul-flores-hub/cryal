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
as a triclinic cell and every PyXtal attempt failed silently.
"""

import os
import random
import tempfile
import unittest

from cryal.config import Config, load_config
from cryal.structure_gen import random_lattice_for_sg


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


class TestConfigRejectsUnsupportedGroups(unittest.TestCase):

    def _load(self, groups, weights):
        fd, path = tempfile.mkstemp(suffix="_INPUT.txt")
        with os.fdopen(fd, "w") as f:
            f.write("% GENERAL\nmoleculeFile = examples/benzene.xyz\nZ = 4\n"
                    f"% SPACE_GROUPS\nspaceGroups = {groups}\nsgWeights = {weights}\n")
        try:
            return load_config(path)
        finally:
            os.unlink(path)

    def test_orthorhombic_group_outside_the_default_list_loads(self):
        self.assertIn(64, self._load("14 2 64", "1 1 1").space_groups)

    def test_tetragonal_group_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not supported"):
            self._load("14 88", "1 1")


if __name__ == "__main__":
    unittest.main()

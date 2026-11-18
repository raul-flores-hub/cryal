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

"""The repository ships a runnable example, and no molecule of its own.

The molecule under study belongs to whoever runs the search: it is input data,
not part of the tool, and it does not travel with the code. The inputs of the
search reported in the accompanying article are in its data deposit. What is
shipped is a generic example (benzene) that anyone can run as-is, with the
relaxation protocol and the search settings of the article (release 1.1.0).
"""

import os
import tomllib
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestShippedExample(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(ROOT, "cryal.toml"), "rb") as f:
            self.toml = tomllib.load(f)

    def _script(self):
        path = os.path.join(ROOT, self.toml["backend"]["lammps"]["input"])
        with open(path) as f:
            return [l.split("#")[0].strip() for l in f.read().splitlines()]

    def test_the_example_molecule_exists_and_is_a_valid_xyz(self):
        path = os.path.join(ROOT, "examples", "benzene.xyz")
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        declared = int(lines[0])
        self.assertEqual(declared, len(lines) - 2)   # count, comment, atoms
        self.assertEqual(declared, 12)

    def test_the_example_input_points_at_the_example_molecule(self):
        molecule = self.toml["general"]["molecule"]
        self.assertTrue(molecule.startswith("examples/"), molecule)
        self.assertTrue(os.path.exists(os.path.join(ROOT, molecule)))

    def test_the_user_molecule_is_kept_out_of_the_repository(self):
        with open(os.path.join(ROOT, ".gitignore")) as f:
            self.assertIn("molecule.xyz", f.read().split())

    def test_species_match_the_lammps_script(self):
        # A mismatch here is a run that dies on the first structure: LAMMPS
        # maps atom types to elements by position in both lists.
        spec = self.toml["backend"]["lammps"]["species"]
        for line in self._script():
            if line.startswith("pair_coeff"):
                self.assertEqual(line.split()[3:], spec)
                break
        else:
            self.fail("no pair_coeff line in the LAMMPS script")

    def test_the_masses_cover_every_atom_type(self):
        spec = self.toml["backend"]["lammps"]["species"]
        masses = [l for l in self._script() if l.startswith("mass ")]
        self.assertEqual(len(masses), len(spec))

    def test_the_script_is_the_protocol_of_the_article(self):
        # Staged NPT anneal on a triclinic cell, then one minimization with the
        # cell relaxed by box/relax.
        script = [l for l in self._script() if l]
        self.assertIn("timestep 0.0004".split(), [l.split() for l in script])
        npt = [l.split() for l in script if l.startswith("fix") and " npt " in f" {l} "]
        self.assertEqual(len(npt), 3)
        self.assertTrue(all("tri" in f for f in npt))
        self.assertEqual([f[f.index("tri") + 1] for f in npt], ["3000.0", "1.0", "1.0"])
        runs = [int(l.split()[1]) for l in script if l.startswith("run ")]
        self.assertEqual(runs, [6000, 4000, 2000])
        self.assertTrue(any(l.startswith("minimize") for l in script))

    def test_the_example_matches_the_article_search_settings(self):
        # BACH, the CH-pi optimizer off and a budget of 20 cycles, as in that search.
        self.assertEqual(self.toml["general"].get("algorithm", "bach"), "bach")
        self.assertIs(self.toml["chpi"]["enabled"], False)
        self.assertEqual(self.toml["active_learning"]["cycles"], 20)


if __name__ == "__main__":
    unittest.main()

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

"""The structural fingerprint (cryal/fingerprint.py) and the [duplicates] rule.

A fingerprint is only a fingerprint if it does not see what does not change
the crystal -- its setting, its origin, its orientation, a supercell, whether
molecules are cut by the cell's faces -- and does see what does: the packing.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from ase.build import make_supercell, niggli_reduce
from ase.io import write

from cryal import genetic as G
from cryal.config import Config, load_config
from cryal.fingerprint import distance, fingerprint
from tests.test_genetic import BENZENE, CENTRES, SKEWED, crystal

MOL = G.Molecule.from_file(BENZENE)


def benzene(seed, cell=SKEWED):
    return crystal(BENZENE, cell, CENTRES, np.random.default_rng(seed))


class TestInvariance(unittest.TestCase):

    def setUp(self):
        self.atoms = benzene(1)
        self.f = fingerprint(self.atoms, MOL)

    def D(self, other):
        return distance(self.f, fingerprint(other, MOL))

    def test_identical(self):
        self.assertAlmostEqual(self.D(self.atoms.copy()), 0.0, places=12)

    def test_setting_orientation_and_origin(self):
        b = self.atoms.copy()
        niggli_reduce(b)                      # wraps atoms one by one: cut molecules
        b.rotate(37, 'z', rotate_cell=True)
        b.translate([1.3, -0.4, 2.2])
        self.assertLess(self.D(b), 1e-9)

    def test_supercell(self):
        # From whole molecules: a supercell of a crystal wrapped atom by atom
        # puts pieces of one molecule in different copies, which is no longer
        # laid out molecule by molecule -- something no database record is.
        whole = crystal(BENZENE, SKEWED, CENTRES, np.random.default_rng(1), wrap=False)
        # Not to 1e-9: in another cell a distance can round across a bin edge.
        self.assertLess(self.D(whole.repeat((2, 1, 1))), 1e-4)
        self.assertLess(self.D(make_supercell(whole, np.diag([1, 2, 1]))), 1e-4)

    def test_a_molecule_cut_by_the_cell_counts_as_one(self):
        whole = crystal(BENZENE, SKEWED, CENTRES, np.random.default_rng(1), wrap=False)
        self.assertLess(self.D(whole), 1e-9)


class TestDiscrimination(unittest.TestCase):

    def test_the_same_crystal_disturbed_is_close_and_others_are_far(self):
        # "Others" share the cell and the molecular centres and differ only in
        # how the molecules are turned: the nearest thing to two polymorphs
        # that agree in energy and density. "The same" is one minimum reached
        # twice: 0.01 Å of noise on the atoms, 0.2 % on the cell.
        rng = np.random.default_rng(0)
        same, other = [], []
        prints = [fingerprint(benzene(s), MOL) for s in range(8)]
        for s, f in enumerate(prints):
            b = benzene(s)
            b.rattle(0.01, seed=s)
            b.set_cell(b.cell.array @ (np.eye(3) + rng.normal(0, 0.002, (3, 3))),
                       scale_atoms=True)
            same.append(distance(f, fingerprint(b, MOL)))
            other += [distance(f, g) for g in prints[s + 1:]]
        default = Config().dup_fingerprint
        self.assertLess(max(same), default)
        self.assertGreater(min(other), default)

    def test_symmetric_and_bounded(self):
        f, g = fingerprint(benzene(1), MOL), fingerprint(benzene(2), MOL)
        self.assertAlmostEqual(distance(f, g), distance(g, f), places=12)
        self.assertTrue(0.0 <= distance(f, g) <= 1.0)


class TestDuplicates(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_fp_")
        G._FINGERPRINTS.clear()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def record(self, i, atoms, energy=-100.0, density=1.0):
        path = os.path.join(self.tmp, f"{i}.cif")
        write(path, atoms)
        return {'id': i, 'energy': energy, 'density': density, 'cif_path': path}

    def test_same_energy_and_density_different_packing_are_two(self):
        a, b = self.record(1, benzene(1)), self.record(2, benzene(2))
        with_fp = Config(Z=4)
        no_fp = Config(Z=4, dup_fingerprint=0.0)
        self.assertFalse(G.same_structure(a, b, with_fp, MOL))
        self.assertTrue(G.same_structure(a, b, no_fp, MOL))
        self.assertEqual(len(G.distinct_lowest([a, b], 5, with_fp, MOL)), 2)
        self.assertEqual(len(G.distinct_lowest([a, b], 5, no_fp, MOL)), 1)

    def test_one_minimum_found_twice_is_one(self):
        x = benzene(3)
        y = x.copy()
        y.rattle(0.01, seed=1)
        a = self.record(1, x)
        b = self.record(2, y, energy=-100.001, density=1.02)   # a density apart
        self.assertTrue(G.same_structure(a, b, Config(Z=4), MOL))
        self.assertFalse(G.same_structure(a, b, Config(Z=4, dup_fingerprint=0.0), MOL))

    def test_energy_still_decides_first(self):
        a = self.record(1, benzene(4))
        b = self.record(2, benzene(4), energy=-99.0)
        self.assertFalse(G.same_structure(a, b, Config(Z=4), MOL))

    def test_an_unreadable_cif_falls_back_to_density(self):
        a = self.record(1, benzene(1))
        b = {'id': 2, 'energy': -100.0, 'density': 1.0,
             'cif_path': os.path.join(self.tmp, "missing.cif")}
        self.assertTrue(G.same_structure(a, b, Config(Z=4), MOL))


class TestDistinctMinimaTool(unittest.TestCase):

    def test_counts_a_run(self):
        import json
        import logging
        from cryal.search import run_random
        from tests.test_genetic import run_config
        from tools import distinct_minima
        tmp = tempfile.mkdtemp(prefix="cryal_fp_tool_")
        try:
            out = os.path.join(tmp, "run")
            run_random(run_config(out, log_level="WARNING", num_cycles=1))
            for h in list(logging.getLogger("cryal").handlers):
                logging.getLogger("cryal").removeHandler(h)
                h.close()
            result = os.path.join(tmp, "minima.json")
            distinct_minima.main([out, "--window", "1000", "--json", result])
            with open(result) as f:
                got = json.load(f)[out]
            with open(os.path.join(out, "database.json")) as f:
                n = len(json.load(f))
            self.assertTrue(1 <= got['n_minima'] <= n)
            self.assertEqual(sum(m['found'] for m in got['minima']), n)
            self.assertEqual(got['minima'][0]['dE_kJ_mol'], 0.0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestConfig(unittest.TestCase):

    def test_table_and_the_old_keys(self):
        tmp = tempfile.mkdtemp(prefix="cryal_fp_cfg_")
        try:
            def load(body):
                path = os.path.join(tmp, "c.toml")
                with open(path, "w") as f:
                    f.write(f'[general]\nmolecule = "{BENZENE}"\n[backend]\nengine = "ase"\n'
                            '[backend.ase]\ncalculator = "tests.stub_calculator.StubCalculator"\n'
                            + body)
                return load_config(path)
            cfg = load("[duplicates]\nenergy = 0.001\nfingerprint = 0.03\n")
            self.assertEqual((cfg.dup_energy, cfg.dup_fingerprint), (0.001, 0.03))
            self.assertEqual(load("").dup_fingerprint, 0.02)
            with self.assertRaises(ValueError):
                load("[ga]\nduplicate_energy = 0.002\n")
            with self.assertRaises(ValueError):
                load("[duplicates]\nfingerprint = 2\n")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

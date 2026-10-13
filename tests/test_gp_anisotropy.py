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

"""The gpAnisotropy option: an eighth GP feature (longest/shortest axis) that
the GP-biased half of each batch reproduces.

The contract that matters most is that it is OFF unless asked for: every
input written before the key existed, the published run included, must keep
meaning the 7-feature surrogate, and the default lattice sampler must draw
exactly the same random numbers as before.
"""

import os
import random
import tempfile
import unittest

import numpy as np

from cryal.config import Config, load_config
from cryal.gp_model import (GPSurrogate, anisotropy, features_from_record,
                            maximize_ei)
from cryal.structure_gen import random_lattice_for_sg


def _load(extra=""):
    fd, path = tempfile.mkstemp(suffix="_INPUT.txt")
    with os.fdopen(fd, "w") as f:
        f.write("% GENERAL\nmoleculeFile = examples/benzene.xyz\nZ = 4\n" + extra)
    try:
        return load_config(path)
    finally:
        os.unlink(path)


def _records(n=30, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(n):
        a, b, c = rng.uniform(4, 20, 3)
        an = max(a, b, c) / min(a, b, c)
        recs.append({'a': a, 'b': b, 'c': c, 'alpha': 90.0,
                     'beta': rng.uniform(90, 120), 'gamma': 90.0,
                     'density': rng.uniform(1.0, 1.4),
                     'energy': -100.0 - 0.1 * an + rng.normal(0, 0.01)})
    return recs


class TestOffByDefault(unittest.TestCase):

    def test_default_is_off(self):
        self.assertFalse(Config().gp_anisotropy)

    def test_input_without_the_key_is_off(self):
        self.assertFalse(_load().gp_anisotropy)

    def test_key_switches_it_on(self):
        self.assertTrue(_load("% GP\ngpAnisotropy = true\n").gp_anisotropy)

    def test_seven_features_when_off(self):
        self.assertEqual(len(features_from_record(_records(1)[0])), 7)

    def test_default_sampler_draws_the_same_numbers_as_before(self):
        # The pre-gpAnisotropy monoclinic branch, written out by hand.
        cfg = Config()
        random.seed(7)
        got = random_lattice_for_sg(14, 2700.0, cfg)
        random.seed(7)
        beta = random.uniform(cfg.beta_min, cfg.beta_max)
        r1 = np.exp(random.gauss(0.0, 0.5))
        r2 = np.exp(random.gauss(0.0, 0.5))
        a = (2700.0 / (r1 * r2 * np.sin(np.radians(beta)))) ** (1 / 3)
        self.assertTrue(np.allclose(got, (a, a * r1, a * r2, 90.0, beta, 90.0)))


class TestFeature(unittest.TestCase):

    def test_eighth_feature_is_longest_over_shortest(self):
        r = {'a': 5.0, 'b': 13.0, 'c': 16.5, 'alpha': 90.0, 'beta': 90.0,
             'gamma': 90.0, 'density': 1.3}
        x = features_from_record(r, use_anisotropy=True)
        self.assertEqual(len(x), 8)
        self.assertAlmostEqual(x[7], 3.3)
        self.assertAlmostEqual(anisotropy(16.5, 5.0, 13.0), 3.3)

    def test_gp_fits_eight_length_scales_and_names_them(self):
        gp = GPSurrogate(n_restarts=0, use_anisotropy=True)
        gp.fit(_records())
        imp = gp.feature_importance()
        self.assertEqual(len(imp), 8)
        self.assertIn('anisotropy', imp)

    def test_suggestions_carry_an_anisotropy_consistent_with_their_axes(self):
        cfg = Config()
        cfg.num_ei_candidates = 2000
        cfg.num_ei_suggestions = 5
        cfg.mol_weight = 228.29
        recs = _records()
        gp = GPSurrogate(n_restarts=0, use_anisotropy=True)
        gp.fit(recs)
        sug = maximize_ei(gp, recs, cfg, np.random.default_rng(0))
        self.assertTrue(sug)
        for s in sug:
            self.assertAlmostEqual(s['anisotropy'], anisotropy(s['a'], s['b'], s['c']))

    def test_suggestions_have_no_anisotropy_when_off(self):
        cfg = Config()
        cfg.num_ei_candidates = 2000
        cfg.mol_weight = 228.29
        recs = _records()
        gp = GPSurrogate(n_restarts=0)
        gp.fit(recs)
        self.assertNotIn('anisotropy', maximize_ei(gp, recs, cfg, np.random.default_rng(0))[0])


class TestSamplerReproducesTarget(unittest.TestCase):

    def test_ratio_within_ten_percent_and_volume_kept(self):
        cfg = Config()
        cfg.min_cell_axis = 3.0
        random.seed(1)
        for sg in (19, 14, 2):
            for _ in range(200):
                a, b, c, al, be, ga = random_lattice_for_sg(sg, 1150.0, cfg,
                                                            anisotropy_target=3.2)
                with self.subTest(sg=sg):
                    self.assertGreaterEqual(anisotropy(a, b, c), 0.9 * 3.2 - 1e-9)
                    self.assertLessEqual(anisotropy(a, b, c), 1.1 * 3.2 + 1e-9)
                    if sg == 19:
                        self.assertAlmostEqual(a * b * c, 1150.0, places=6)

    def test_long_axis_is_not_always_the_same_one(self):
        cfg = Config()
        cfg.min_cell_axis = 3.0
        random.seed(2)
        longest = {int(np.argmax(random_lattice_for_sg(19, 1150.0, cfg,
                                                       anisotropy_target=3.2)[:3]))
                   for _ in range(60)}
        self.assertEqual(longest, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()

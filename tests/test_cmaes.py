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

"""CMA-ES (cryal/cmaes.py).

Three things are guarded: that the strategy is CMA-ES (it solves the problems
CMA-ES is known to solve, and learns an ill-conditioned one's shape), that a
vector is always a crystal (the starting structure at zero, a valid cell
anywhere), and that the state a run rebuilds from its database is exactly the
state it was in.
"""

import json
import logging
import os
import shutil
import tempfile
import unittest

import numpy as np

from cryal import cmaes as CM
from cryal import genetic as G
from cryal import search as S
from cryal.config import Config, load_config
from tests.test_genetic import BENZENE, CENTRES, SKEWED, crystal, run_config

MOL = G.Molecule.from_file(BENZENE)


def evolve(f, n, lam, gens, seed=0, x0=3.0):
    rng = np.random.default_rng(seed)
    es = CM.CMA(np.full(n, x0), 1.0, lam)
    for _ in range(gens):
        X = [es.ask(rng) for _ in range(lam)]
        es.tell(X, [f(x) for x in X])
    return es


class TestStrategy(unittest.TestCase):

    def test_sphere(self):
        es = evolve(lambda x: float(x @ x), 10, 10, 150)
        self.assertLess(float(es.mean @ es.mean), 1e-6)

    def test_learns_an_ill_conditioned_ellipsoid(self):
        def ellipsoid(x):
            return float(sum(10 ** (6 * i / (len(x) - 1)) * v * v for i, v in enumerate(x)))
        es = evolve(ellipsoid, 10, 10, 600)
        self.assertLess(ellipsoid(es.mean), 1e-6)
        _, D = es.axes()
        self.assertGreater(D.max() / D.min(), 300)          # sqrt(1e6) = 1000

    def test_rosenbrock(self):
        def rosen(x):
            return float(sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
        es = evolve(rosen, 5, 8, 800, x0=0.0)
        np.testing.assert_allclose(es.mean, np.ones(5), atol=1e-3)

    def test_a_generation_with_no_results_changes_only_the_count(self):
        es = CM.CMA(np.zeros(4), 1.0, 8)
        es.tell([], [])
        self.assertEqual(es.generation, 1)
        np.testing.assert_array_equal(es.mean, np.zeros(4))
        self.assertEqual(es.sigma, 1.0)

    def test_fewer_results_than_mu_still_update(self):
        es = CM.CMA(np.zeros(4), 1.0, 8)
        es.tell([np.ones(4), 2 * np.ones(4)], [1.0, 2.0])
        self.assertTrue(np.all(es.mean > 0))
        self.assertEqual(len(es.w), 4)                       # back to lambda // 2


class TestEncoding(unittest.TestCase):

    def setUp(self):
        atoms = crystal(BENZENE, SKEWED, CENTRES, np.random.default_rng(1))
        self.start = G.standardize(G.genome_from_atoms(atoms, MOL))
        self.enc = CM.Encoding(self.start, Config(Z=4))

    def test_zero_is_the_starting_structure(self):
        g = self.enc.decode(np.zeros(self.enc.n))
        np.testing.assert_allclose(g.cell, self.start.cell, atol=1e-12)
        np.testing.assert_allclose(g.frac, self.start.frac, atol=1e-12)
        np.testing.assert_allclose(g.shapes, self.start.shapes, atol=1e-12)

    def test_every_vector_is_a_valid_cell(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            g = self.enc.decode(rng.normal(0, 5, self.enc.n))
            self.assertGreater(np.linalg.det(g.cell), 0)
            self.assertTrue(np.all(np.isfinite(g.cell)))

    def test_cell_round_trip_and_scales(self):
        np.testing.assert_allclose(CM.u_to_cell(CM.cell_to_u(self.start.cell)),
                                   self.start.cell, atol=1e-12)
        self.assertEqual(self.enc.n, 6 + 6 * 4)
        x = np.zeros(self.enc.n)
        x[6] = 1.0                                           # one unit on centre 0, along a
        g = self.enc.decode(x)
        d = (g.frac[0] - self.start.frac[0]) @ self.start.cell
        self.assertAlmostEqual(np.linalg.norm(d), 0.5, places=9)   # shift_scale, Å


def fake_records(n_start=3):
    """Cycle-0 records without CIFs; fingerprints off, so none is read."""
    return [{'id': i + 1, 'cycle': 0, 'energy': -100.0 + 0.5 * i, 'density': 1.0 + 0.1 * i,
             'source': 'random_sg2'} for i in range(n_start)]


class TestReplay(unittest.TestCase):

    def cfg(self, **kw):
        c = Config(Z=4, structures_per_cycle=8, dup_fingerprint=0.0, cma_stall=100)
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def test_replaying_the_database_is_the_live_state(self):
        cfg = self.cfg()
        rng = np.random.default_rng(2)
        live = CM.CMA(np.zeros(30), cfg.cma_sigma, cfg.structures_per_cycle)
        records = fake_records()
        target = rng.normal(size=30)
        for cycle in range(1, 6):
            X = [live.ask(rng) for _ in range(8)]
            E = [float(np.sum((x - target) ** 2)) for x in X]
            kept = [k for k in range(8) if k != cycle % 8]   # one relaxation lost per cycle
            live.tell([X[k] for k in kept], [E[k] for k in kept])
            for k in kept:
                records.append({'id': len(records) + 1, 'cycle': cycle, 'energy': E[k],
                                'density': 1.0, 'source': 'cmaes',
                                'cma': {'instance': 1, 'x': [float(v) for v in X[k]]}})
        inst, starts = CM.replay(records, cfg, None, up_to=6)
        self.assertEqual(starts, [1])
        self.assertEqual(inst.cma.generation, live.generation)
        np.testing.assert_allclose(inst.cma.mean, live.mean, atol=1e-12)
        np.testing.assert_allclose(inst.cma.C, live.C, atol=1e-12)
        np.testing.assert_allclose(inst.cma.ps, live.ps, atol=1e-12)
        np.testing.assert_allclose(inst.cma.pc, live.pc, atol=1e-12)
        self.assertAlmostEqual(inst.cma.sigma, live.sigma, places=12)

    def test_a_stalled_instance_restarts_from_the_next_distinct_structure(self):
        cfg = self.cfg(cma_stall=2)
        records = fake_records()
        for cycle in (1, 2):                                 # never better than the start
            records.append({'id': len(records) + 1, 'cycle': cycle, 'energy': -50.0,
                            'density': 1.0, 'source': 'cmaes',
                            'cma': {'instance': 1, 'x': [0.1] * 30}})
        inst, starts = CM.replay(records, cfg, None, up_to=3)
        self.assertEqual(starts, [1, 2])
        self.assertEqual((inst.number, inst.start['id'], inst.cma.generation), (2, 2, 0))


def reset_logger():
    lg = logging.getLogger("cryal")
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()


class TestRun(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_cma_")
        reset_logger()

    def tearDown(self):
        reset_logger()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def records(self, out):
        with open(os.path.join(out, "database.json")) as f:
            return json.load(f)

    def test_runs_and_resumes_in_the_same_state(self):
        out = os.path.join(self.tmp, "cma")
        cfg = run_config(out, log_level="WARNING", structures_per_cycle=8, num_cycles=2)
        CM.run(cfg)
        first = self.records(out)
        samples = [r for r in first if r['cycle'] > 0 and 'cma' in r]
        self.assertTrue(samples)
        for r in samples:
            self.assertEqual(S.source_family(r['source']), "cmaes")
            self.assertEqual(len(r['cma']['x']), 6 + 6 * cfg.Z)
        inst, starts = CM.replay(first, cfg, MOL, up_to=2)
        for r in [r for r in samples if r['cycle'] == 1]:
            self.assertEqual(r['parents'], [starts[0]])
        before = CM.replay(first, cfg, MOL, up_to=3)[0]

        reset_logger()
        CM.run(run_config(out, log_level="WARNING", structures_per_cycle=8, num_cycles=3,
                          resume=True))
        again = self.records(out)
        self.assertEqual(again[:len(first)], first)
        self.assertEqual({r['cycle'] for r in again[len(first):]}, {3})
        # Generation 3 was drawn from the state the first run ended in.
        for r in [r for r in again if r['cycle'] == 3 and 'cma' in r]:
            self.assertEqual(r['cma']['instance'], before.number)
            self.assertEqual(r['parents'], [before.start['id']])
        with open(os.path.join(out, "final_report.json")) as f:
            self.assertEqual(json.load(f)['algorithm'], "cmaes")

    def test_config(self):
        path = os.path.join(self.tmp, "c.toml")
        with open(path, "w") as f:
            f.write(f'[general]\nmolecule = "{BENZENE}"\nalgorithm = "cmaes"\n'
                    '[backend]\nengine = "ase"\n'
                    '[backend.ase]\ncalculator = "tests.stub_calculator.StubCalculator"\n'
                    '[cmaes]\nsigma = 0.5\nstall = 3\n')
        cfg = load_config(path)
        self.assertEqual((cfg.algorithm, cfg.cma_sigma, cfg.cma_stall), ("cmaes", 0.5, 3))


if __name__ == "__main__":
    unittest.main()

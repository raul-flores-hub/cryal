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

"""The particle swarm (cryal/pso.py).

A swarm is only as good as its arithmetic on crystals: y - x must be zero when
y is x described differently (another origin, molecules in another order, a
molecule turned by one of its own symmetries), a full step along y - x must
land on y, and a velocity must turn with the crystal it belongs to.
"""

import json
import logging
import os
import shutil
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from cryal import genetic as G
from cryal import pso as P
from cryal import search as S
from cryal.config import Config, load_config
from tests.test_genetic import BENZENE, CENTRES, SKEWED, crystal, run_config

MOL = G.Molecule.from_file(BENZENE)
SHAPE = P.Shape(BENZENE)


def genome(seed):
    atoms = crystal(BENZENE, SKEWED, CENTRES, np.random.default_rng(seed))
    return G.standardize(G.genome_from_atoms(atoms, MOL))


def norms(v):
    return {k: float(np.abs(a).max()) for k, a in v.items()}


class TestOrientation(unittest.TestCase):

    def test_kabsch_recovers_a_rotation(self):
        R = Rotation.from_rotvec([0.3, -1.1, 0.4]).as_matrix()
        np.testing.assert_allclose(P.kabsch(SHAPE.ref, SHAPE.ref @ R.T), R, atol=1e-9)

    def test_benzene_has_twelve_proper_symmetry_rotations(self):
        self.assertEqual(len(SHAPE.symmetry), 12)        # D6h: 24 operations, 12 proper

    def test_a_turn_by_a_symmetry_of_the_molecule_is_no_turn(self):
        normal = np.cross(SHAPE.ref[0], SHAPE.ref[1])
        normal /= np.linalg.norm(normal)
        R0 = Rotation.from_rotvec([0.2, 0.5, -0.3]).as_matrix()
        for deg, expect in [(60, 0.0), (180, 0.0), (30, 30.0), (90, 30.0)]:
            R = Rotation.from_rotvec(np.radians(deg) * normal).as_matrix()
            turn = SHAPE.turn(R0, R0 @ R)
            self.assertAlmostEqual(np.degrees(np.linalg.norm(turn)), expect, places=6,
                                   msg=f"{deg} degrees about the ring normal")


class TestPull(unittest.TestCase):

    def test_x_minus_x_is_zero(self):
        x = genome(1)
        self.assertLess(max(norms(P.pull(x, x, SHAPE)).values()), 1e-9)

    def test_another_description_of_x_is_x(self):
        # Moved origin, molecules in another order, each turned by a symmetry
        # of benzene: the same crystal, so nothing to pull toward.
        x = genome(1)
        y = x.copy()
        order = [2, 0, 3, 1]
        normal = np.cross(SHAPE.ref[0], SHAPE.ref[1])
        normal /= np.linalg.norm(normal)
        y.frac = (x.frac[order] + [0.31, 0.12, 0.77]) % 1.0
        y.shapes = np.array([x.shapes[m] for m in order])
        for m in range(4):
            R0 = SHAPE.orientation(y.shapes[m])
            sym = R0 @ Rotation.from_rotvec(np.radians(120) * normal).as_matrix() @ R0.T
            y.shapes[m] = y.shapes[m] @ sym.T
        self.assertLess(max(norms(P.pull(x, y, SHAPE)).values()), 1e-6)

    def test_a_strained_copy_pulls_by_its_strain(self):
        x = genome(1)
        eps = np.array([[0.02, 0.01, 0.0], [0.01, -0.03, 0.005], [0.0, 0.005, 0.01]])
        y = x.copy()
        y.cell = x.cell @ (np.eye(3) + eps)
        v = P.pull(x, y, SHAPE)
        np.testing.assert_allclose(v['strain'], eps, atol=1e-9)
        self.assertLess(norms(v)['centres'], 1e-9)
        self.assertLess(norms(v)['turns'], 1e-9)

    def test_a_full_step_lands_on_the_target(self):
        x, y = genome(1), genome(2)
        z = P.step(x, P.pull(x, y, SHAPE))
        self.assertLess(max(norms(P.pull(z, y, SHAPE)).values()), 1e-6)

    def test_a_velocity_turns_with_its_crystal(self):
        x, y = genome(1), genome(2)
        v = P.pull(x, y, SHAPE)
        Q = Rotation.from_rotvec([0.4, 0.1, -0.7]).as_matrix()     # rows: r @ Q
        xq = G.Genome(x.cell @ Q, x.frac.copy(), x.shapes @ Q)
        a = P.step(xq, P.rotate_velocity(v, Q))
        b = P.step(x, v)
        np.testing.assert_allclose(a.cell, b.cell @ Q, atol=1e-9)
        np.testing.assert_allclose(a.shapes, b.shapes @ Q, atol=1e-9)
        np.testing.assert_allclose(a.frac, b.frac, atol=1e-9)


class TestVelocity(unittest.TestCase):

    def test_clip(self):
        rng = np.random.default_rng(0)
        v = {'strain': rng.normal(size=(3, 3)), 'centres': rng.normal(size=(4, 3)) * 5,
             'turns': rng.normal(size=(4, 3)) * 5}
        c = P.clip(v, 0.1, 1.0, 30.0)
        self.assertLessEqual(np.linalg.norm(c['strain']), 0.1 + 1e-12)
        self.assertTrue(np.all(np.linalg.norm(c['centres'], axis=1) <= 1.0 + 1e-12))
        self.assertTrue(np.all(np.degrees(np.linalg.norm(c['turns'], axis=1)) <= 30 + 1e-9))

    def test_inertia_falls_over_the_run(self):
        cfg = Config(num_cycles=5, pso_inertia=[0.9, 0.4])
        self.assertAlmostEqual(P.inertia(cfg, 1), 0.9)
        self.assertAlmostEqual(P.inertia(cfg, 5), 0.4)
        self.assertAlmostEqual(P.inertia(cfg, 3), 0.65)

    def test_json_round_trip(self):
        v = P.pull(genome(1), genome(2), SHAPE)
        back = P.from_json(json.loads(json.dumps(P.to_json(v))))
        for k in v:
            np.testing.assert_allclose(back[k], v[k], atol=1e-6)

    def test_swarm_size(self):
        self.assertEqual(P.swarm_size(Config(structures_per_cycle=50,
                                             pso_random_fraction=0.4)), 30)
        self.assertEqual(P.swarm_size(Config(structures_per_cycle=8,
                                             pso_random_fraction=0.0)), 8)


def reset_logger():
    lg = logging.getLogger("cryal")
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()


class TestRun(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_pso_")
        reset_logger()

    def tearDown(self):
        reset_logger()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def records(self, out):
        with open(os.path.join(out, "database.json")) as f:
            return json.load(f)

    def check(self, recs, cfg):
        """Each step starts where its particle was; the rest of a cycle is random."""
        start = {p: r['id'] for p, r in enumerate(P.starting_points(recs, cfg, MOL))}
        here = dict(start)
        n_particles = P.swarm_size(cfg)
        for cycle in sorted({r['cycle'] for r in recs if r['cycle'] > 0}):
            mine = [r for r in recs if r['cycle'] == cycle]
            steps = [r for r in mine if 'particle' in r]
            self.assertTrue(steps)
            for r in steps:
                self.assertEqual(S.source_family(r['source']), "pso")
                self.assertIn(r['particle'], start)
                self.assertEqual(r['parents'], [here[r['particle']]])
                v = P.from_json(r['velocity'])
                self.assertEqual(v['strain'].shape, (3, 3))
                self.assertEqual(v['centres'].shape, (cfg.Z, 3))
                self.assertEqual(v['turns'].shape, (cfg.Z, 3))
            for r in steps:
                here[r['particle']] = r['id']
            others = [r for r in mine if 'particle' not in r]
            for r in others:
                self.assertEqual(S.source_family(r['source']), "random")
            self.assertLessEqual(len(steps), n_particles)

    def test_runs_and_resumes(self):
        out = os.path.join(self.tmp, "pso")
        cfg = run_config(out, log_level="WARNING", structures_per_cycle=8, num_cycles=2)
        P.run(cfg)
        first = self.records(out)
        self.assertEqual({r['cycle'] for r in first}, {0, 1, 2})
        self.check(first, cfg)
        reset_logger()
        cfg = run_config(out, log_level="WARNING", structures_per_cycle=8, num_cycles=3,
                         resume=True)
        P.run(cfg)
        again = self.records(out)
        self.assertEqual(again[:len(first)], first)
        self.check(again, cfg)
        with open(os.path.join(out, "final_report.json")) as f:
            rep = json.load(f)
        self.assertEqual(rep['algorithm'], "pso")
        self.assertEqual(rep['config']['particles'], P.swarm_size(cfg))

    def test_config(self):
        path = os.path.join(self.tmp, "c.toml")
        with open(path, "w") as f:
            f.write(f'[general]\nmolecule = "{BENZENE}"\nalgorithm = "pso"\n'
                    '[backend]\nengine = "ase"\n'
                    '[backend.ase]\ncalculator = "tests.stub_calculator.StubCalculator"\n'
                    '[pso]\nrandom_fraction = 0.25\ninertia = [0.8, 0.3]\n')
        cfg = load_config(path)
        self.assertEqual((cfg.algorithm, cfg.pso_random_fraction, cfg.pso_inertia),
                         ("pso", 0.25, [0.8, 0.3]))
        with open(path, "a") as f:
            f.write("max_turns = 10\n")
        with self.assertRaises(ValueError) as ctx:
            load_config(path)
        self.assertIn("did you mean 'max_turn'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

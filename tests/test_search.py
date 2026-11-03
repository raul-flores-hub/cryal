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

"""The searches of cryal/search.py other than the GA, which test_genetic covers.

A baseline is only worth having if it is the baseline it says it is: pure
random search must learn nothing from what it has evaluated -- no GP, no
perturbed incumbents, no parents -- while spending the same budget through the
same pipeline as everything it is compared with.
"""

import json
import logging
import os
import shutil
import tempfile
import unittest

from cryal import basin_hopping as BH
from cryal import genetic as G
from cryal import search as S
from cryal.config import load_config
from tests.test_genetic import BENZENE, run_config


def reset_logger():
    lg = logging.getLogger("cryal")
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()


class SearchCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_search_")
        reset_logger()

    def tearDown(self):
        reset_logger()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def records(self, out):
        with open(os.path.join(out, "database.json")) as f:
            return json.load(f)

    def report(self, out):
        with open(os.path.join(out, "final_report.json")) as f:
            return json.load(f)

    def toml(self, body):
        path = os.path.join(self.tmp, "cryal.toml")
        with open(path, "w") as f:
            f.write(f'[general]\nmolecule = "{BENZENE}"\n' + body +
                    '[backend]\nengine = "ase"\n'
                    '[backend.ase]\ncalculator = "tests.stub_calculator.StubCalculator"\n')
        return path


class TestSourceFamily(unittest.TestCase):

    def test_families(self):
        for source, family in [("biased_sg14_b101", "biased"), ("random_sg2", "random"),
                               ("perturb_x.cif", "perturb"), ("seed_x.cif", "seed"),
                               ("ga_crossover", "crossover"), ("ga_immigrant_sg2", "immigrant"),
                               ("hop_rotation", "hop"), ("unknown", "other")]:
            self.assertEqual(S.source_family(source), family, source)


class TestRandomSearch(SearchCase):

    def test_config(self):
        self.assertEqual(load_config(self.toml('algorithm = "random"\n')).algorithm, "random")
        with self.assertRaises(ValueError) as ctx:
            load_config(self.toml('algorithm = "random"\n'
                                  '[active_learning]\nga_fraction = 0.5\n'))
        self.assertIn('algorithm = "bach"', str(ctx.exception))

    def test_learns_nothing_and_spends_the_budget(self):
        out = os.path.join(self.tmp, "random")
        S.run_random(run_config(out, log_level="WARNING", structures_per_cycle=5))
        recs = self.records(out)
        self.assertEqual({r['cycle'] for r in recs}, {0, 1, 2})
        for r in recs:
            self.assertEqual(S.source_family(r['source']), "random", r['source'])
            self.assertNotIn('parents', r)
        # Every cycle asks for structures_per_cycle and, with a stub that
        # rejects nothing, keeps them: no pool perturbations on top.
        for cycle in (1, 2):
            self.assertEqual(sum(r['cycle'] == cycle for r in recs), 5)
        rep = self.report(out)
        self.assertEqual(rep['algorithm'], "random")
        self.assertEqual(rep['source_stats']['random']['kept'], len(recs))

    def test_resume_continues(self):
        out = os.path.join(self.tmp, "random")
        S.run_random(run_config(out, log_level="WARNING", num_cycles=1))
        first = self.records(out)
        reset_logger()
        S.run_random(run_config(out, log_level="WARNING", num_cycles=2, resume=True))
        again = self.records(out)
        self.assertEqual(again[:len(first)], first)
        self.assertEqual({r['cycle'] for r in again[len(first):]}, {2})



class TestBasinHopping(SearchCase):

    def test_metropolis(self):
        self.assertTrue(BH.accept(-0.1, 0.0, 0.99))         # downhill, even cold
        self.assertTrue(BH.accept(0.0, 0.0, 0.99))
        self.assertFalse(BH.accept(0.001, 0.0, 0.0))        # T = 0: never uphill
        self.assertTrue(BH.accept(0.01, 0.01, 0.36))        # exp(-1) = 0.368
        self.assertFalse(BH.accept(0.01, 0.01, 0.37))

    def test_shares(self):
        self.assertEqual(BH.shares(10, 3, 0), [4, 3, 3])
        self.assertEqual(BH.shares(10, 3, 1), [3, 4, 3])    # the remainder rotates
        self.assertEqual(BH.shares(2, 3, 0), [1, 1, 0])
        self.assertEqual(sum(BH.shares(50, 7, 5)), 50)

    def test_config(self):
        cfg = load_config(self.toml('algorithm = "basin_hopping"\n[basin_hopping]\n'
                                    'walkers = 4\nmoves = { rotation = 1 }\n'))
        self.assertEqual(cfg.bh_walkers, 4)
        self.assertEqual(cfg.bh_moves, dict(strain=0.0, rotation=1.0, translation=0.0))
        with self.assertRaises(ValueError) as ctx:
            load_config(self.toml('[basin_hopping]\nmoves = { crossover = 1 }\n'))
        self.assertIn("unknown move", str(ctx.exception))

    def run_bh(self, out, **kw):
        cfg = run_config(out, log_level="WARNING", bh_walkers=3,
                         structures_per_cycle=6, **kw)
        BH.run(cfg)
        reset_logger()
        return cfg

    def check_walks(self, recs, cfg):
        """Every trial starts where its walker was; at most one move per cycle."""
        start = {w: r['id'] for w, r in enumerate(BH.starting_points(recs, cfg, G.Molecule.from_file(BENZENE)))}
        by_id = {r['id']: r for r in recs}
        here = dict(start)
        for cycle in sorted({r['cycle'] for r in recs if r['cycle'] > 0}):
            trials = [r for r in recs if r['cycle'] == cycle and 'walker' in r]
            for r in trials:
                self.assertEqual(S.source_family(r['source']), "hop")
                self.assertEqual(r['parents'], [here[r['walker']]])
                self.assertIn('accepted', r)
            for w in {r['walker'] for r in trials}:
                won = [r for r in trials if r['walker'] == w and r['accepted']]
                self.assertLessEqual(len(won), 1)
                if won:
                    mine = [r for r in trials if r['walker'] == w]
                    self.assertEqual(won[0]['energy'], min(r['energy'] for r in mine))
                    here[w] = won[0]['id']
        return here, by_id

    def test_walks(self):
        out = os.path.join(self.tmp, "bh")
        cfg = self.run_bh(out)
        recs = self.records(out)
        self.assertEqual({r['cycle'] for r in recs}, {0, 1, 2})
        hops = [r for r in recs if r['cycle'] > 0 and 'walker' in r]
        self.assertTrue(hops)
        # One walker per distinct starting structure, up to the 3 asked for.
        # (The stub's energy depends on volume alone, so two of cycle 0's
        # structures can count as one, and leave fewer than 3.)
        n_start = len(BH.starting_points(recs, cfg, G.Molecule.from_file(BENZENE)))
        self.assertTrue(1 <= n_start <= 3)
        self.assertEqual({r['walker'] for r in hops}, set(range(n_start)))
        self.check_walks(recs, cfg)
        rep = self.report(out)
        self.assertEqual(rep['algorithm'], "basin_hopping")
        self.assertEqual(rep['config']['walkers'], 3)

    def test_a_cold_walk_never_goes_uphill(self):
        out = os.path.join(self.tmp, "bh")
        cfg = self.run_bh(out, bh_temperature=0.0)
        recs = self.records(out)
        here, by_id = self.check_walks(recs, cfg)
        for r in recs:
            if r.get('accepted'):
                self.assertLessEqual(r['energy'], by_id[r['parents'][0]]['energy'])

    def test_an_unsettled_cycle_is_settled_on_resume(self):
        out = os.path.join(self.tmp, "bh")
        self.run_bh(out, num_cycles=1)
        path = os.path.join(out, "database.json")
        with open(path) as f:
            recs = json.load(f)
        for r in recs:                      # as if it stopped before settling
            r.pop('accepted', None)
        with open(path, "w") as f:
            json.dump(recs, f)
        cfg = self.run_bh(out, num_cycles=2, resume=True)
        again = self.records(out)
        self.assertTrue(all('accepted' in r for r in again if 'walker' in r))
        self.assertEqual({r['cycle'] for r in again}, {0, 1, 2})
        self.check_walks(again, cfg)


if __name__ == "__main__":
    unittest.main()

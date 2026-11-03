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


if __name__ == "__main__":
    unittest.main()
